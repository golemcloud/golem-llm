mod client;
mod conversions;

use crate::client::{
    Content, ContentBlockDelta, ErrorResponse, MessagesApi, MessagesRequest, StopReason, Usage,
};
use crate::conversions::{
    convert_usage, messages_to_request, process_response, stop_reason_to_finish_reason,
    tool_results_to_messages,
};
use golem_llm::chat_stream::{LlmChatStream, LlmChatStreamState};
use golem_llm::config::with_config_key;
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, ContentPart, Error, ErrorCode, Guest, Message, ResponseMetadata,
    Role, StreamDelta, StreamEvent, ToolCall, ToolResult,
};
use golem_llm::LOGGING_STATE;
use golem_rust::wasm_rpc::Pollable;
use log::trace;
use std::cell::{Ref, RefCell, RefMut};
use std::collections::HashMap;

#[derive(Default)]
struct JsonFragment {
    id: String,
    name: String,
    json: String,
}

struct AnthropicChatStream {
    stream: RefCell<Option<EventSource>>,
    failure: Option<Error>,
    finished: RefCell<bool>,
    json_fragments: RefCell<HashMap<u64, JsonFragment>>,
    response_metadata: RefCell<ResponseMetadata>,
}

impl AnthropicChatStream {
    pub fn new(stream: EventSource) -> LlmChatStream<Self> {
        LlmChatStream::new(AnthropicChatStream {
            stream: RefCell::new(Some(stream)),
            failure: None,
            finished: RefCell::new(false),
            json_fragments: RefCell::new(HashMap::new()),
            response_metadata: RefCell::new(ResponseMetadata {
                finish_reason: None,
                usage: None,
                provider_id: None,
                timestamp: None,
                provider_metadata_json: None,
            }),
        })
    }

    pub fn failed(error: Error) -> LlmChatStream<Self> {
        LlmChatStream::new(AnthropicChatStream {
            stream: RefCell::new(None),
            failure: Some(error),
            finished: RefCell::new(false),
            json_fragments: RefCell::new(HashMap::new()),
            response_metadata: RefCell::new(ResponseMetadata {
                finish_reason: None,
                usage: None,
                provider_id: None,
                timestamp: None,
                provider_metadata_json: None,
            }),
        })
    }
}

impl LlmChatStreamState for AnthropicChatStream {
    fn failure(&self) -> &Option<Error> {
        &self.failure
    }

    fn is_finished(&self) -> bool {
        *self.finished.borrow()
    }

    fn set_finished(&self) {
        *self.finished.borrow_mut() = true;
    }

    fn stream(&self) -> Ref<Option<EventSource>> {
        self.stream.borrow()
    }

    fn stream_mut(&self) -> RefMut<Option<EventSource>> {
        self.stream.borrow_mut()
    }

    fn decode_message(&self, raw: &str) -> Result<Option<StreamEvent>, String> {
        trace!("Received raw stream event: {raw}");
        let json: serde_json::Value = serde_json::from_str(raw)
            .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;

        let typ = json
            .as_object()
            .and_then(|obj| obj.get("type"))
            .and_then(|v| v.as_str());
        match typ {
            Some("error") => {
                let error = serde_json::from_value::<ErrorResponse>(json)
                    .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;
                Ok(Some(StreamEvent::Error(Error {
                    code: ErrorCode::InternalError,
                    message: error.error.message,
                    provider_error_json: None,
                })))
            }
            Some("content_block_start") => {
                let index = json
                    .as_object()
                    .and_then(|obj| obj.get("index"))
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        "Unexpected stream event format, does not have 'index' field".to_string()
                    })?;

                let raw_content_block = json
                    .as_object()
                    .and_then(|obj| obj.get("content_block"))
                    .ok_or_else(|| {
                    "Unexpected stream event format, does not have 'content_block' field"
                        .to_string()
                })?;

                let content_block = serde_json::from_value::<Content>(raw_content_block.clone())
                    .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;

                if let Content::ToolUse { id, name, .. } = content_block {
                    self.json_fragments.borrow_mut().insert(
                        index,
                        JsonFragment {
                            id,
                            name,
                            json: String::new(),
                        },
                    );
                }

                Ok(None)
            }
            Some("content_block_delta") => {
                let raw_delta = json
                    .as_object()
                    .and_then(|obj| obj.get("delta"))
                    .ok_or_else(|| {
                        "Unexpected stream event format, does not have 'delta' field".to_string()
                    })?;
                let delta = serde_json::from_value::<ContentBlockDelta>(raw_delta.clone())
                    .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;

                match delta {
                    ContentBlockDelta::TextDelta { text } => {
                        Ok(Some(StreamEvent::Delta(StreamDelta {
                            content: Some(vec![ContentPart::Text(text)]),
                            tool_calls: None,
                        })))
                    }
                    ContentBlockDelta::InputJsonDelta { partial_json } => {
                        let index = json
                            .as_object()
                            .and_then(|obj| obj.get("index"))
                            .and_then(|v| v.as_u64())
                            .ok_or_else(|| {
                                "Unexpected stream event format, does not have 'index' field"
                                    .to_string()
                            })?;

                        let mut json_fragments = self.json_fragments.borrow_mut();
                        let fragment = json_fragments.entry(index).or_default();
                        fragment.json.push_str(&partial_json);

                        Ok(None)
                    }
                }
            }
            Some("content_block_stop") => {
                let index = json
                    .as_object()
                    .and_then(|obj| obj.get("index"))
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        "Unexpected stream event format, does not have 'index' field".to_string()
                    })?;

                if let Some(tool_use) = self.json_fragments.borrow_mut().remove(&index) {
                    Ok(Some(StreamEvent::Delta(StreamDelta {
                        content: None,
                        tool_calls: Some(vec![ToolCall {
                            id: tool_use.id,
                            name: tool_use.name,
                            arguments_json: tool_use.json,
                        }]),
                    })))
                } else {
                    Ok(None)
                }
            }
            Some("message_delta") => {
                let stop_reason = json
                    .as_object()
                    .and_then(|obj| obj.get("delta"))
                    .and_then(|v| v.as_object())
                    .and_then(|obj| obj.get("stop_reason"))
                    .and_then(|v| serde_json::from_value::<StopReason>(v.clone()).ok());
                let usage = json
                    .as_object()
                    .and_then(|obj| obj.get("usage"))
                    .and_then(|v| serde_json::from_value::<Usage>(v.clone()).ok());

                if let Some(stop_reason) = stop_reason {
                    self.response_metadata.borrow_mut().finish_reason =
                        Some(stop_reason_to_finish_reason(stop_reason));
                }
                if let Some(usage) = usage {
                    self.response_metadata.borrow_mut().usage = Some(convert_usage(usage));
                }
                Ok(None)
            }
            Some("message_stop") => {
                let response_metadata = self.response_metadata.borrow().clone();
                Ok(Some(StreamEvent::Finish(response_metadata)))
            }
            Some(_) => Ok(None),
            None => Err("Unexpected stream event format, does not have 'type' field".to_string()),
        }
    }
}

struct AnthropicComponent;

impl AnthropicComponent {
    const ENV_VAR_NAME: &'static str = "ANTHROPIC_API_KEY";

    fn request(client: MessagesApi, request: MessagesRequest) -> ChatEvent {
        match client.send_messages(request) {
            Ok(response) => process_response(response),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn streaming_request(
        client: MessagesApi,
        mut request: MessagesRequest,
    ) -> LlmChatStream<AnthropicChatStream> {
        request.stream = true;
        match client.stream_send_messages(request) {
            Ok(stream) => AnthropicChatStream::new(stream),
            Err(err) => AnthropicChatStream::failed(err),
        }
    }
}

impl Guest for AnthropicComponent {
    type ChatStream = LlmChatStream<AnthropicChatStream>;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |anthropic_api_key| {
            let client = MessagesApi::new(anthropic_api_key);

            match messages_to_request(messages, config) {
                Ok(request) => Self::request(client, request),
                Err(err) => ChatEvent::Error(err),
            }
        })
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |anthropic_api_key| {
            let client = MessagesApi::new(anthropic_api_key);

            match messages_to_request(messages, config) {
                Ok(mut request) => {
                    request
                        .messages
                        .extend(tool_results_to_messages(tool_results));
                    Self::request(client, request)
                }
                Err(err) => ChatEvent::Error(err),
            }
        })
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for AnthropicComponent {
    fn unwrapped_stream(
        messages: Vec<Message>,
        config: Config,
    ) -> LlmChatStream<AnthropicChatStream> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(
            Self::ENV_VAR_NAME,
            AnthropicChatStream::failed,
            |anthropic_api_key| {
                let client = MessagesApi::new(anthropic_api_key);

                match messages_to_request(messages, config) {
                    Ok(request) => Self::streaming_request(client, request),
                    Err(err) => AnthropicChatStream::failed(err),
                }
            },
        )
    }

    fn retry_prompt(original_messages: &[Message], partial_result: &[StreamDelta]) -> Vec<Message> {
        let mut extended_messages = Vec::new();
        extended_messages.push(Message {
            role: Role::System,
            name: None,
            content: vec![
                ContentPart::Text(
                    "You were asked the same question previously, but the response was interrupted before completion. \
                     Please continue your response from where you left off. \
                     Do not include the part of the response that was already seen.".to_string()),
            ],
        });
        extended_messages.push(Message {
            role: Role::User,
            name: None,
            content: vec![ContentPart::Text(
                "Here is the original question:".to_string(),
            )],
        });
        extended_messages.extend_from_slice(original_messages);

        let mut partial_result_as_content = Vec::new();
        for delta in partial_result {
            if let Some(contents) = &delta.content {
                partial_result_as_content.extend_from_slice(contents);
            }
            if let Some(tool_calls) = &delta.tool_calls {
                for tool_call in tool_calls {
                    partial_result_as_content.push(ContentPart::Text(format!(
                        "<tool-call id=\"{}\" name=\"{}\" arguments=\"{}\"/>",
                        tool_call.id, tool_call.name, tool_call.arguments_json,
                    )));
                }
            }
        }

        extended_messages.push(Message {
            role: Role::User,
            name: None,
            content: vec![ContentPart::Text(
                "Here is the partial response that was successfully received:".to_string(),
            )]
            .into_iter()
            .chain(partial_result_as_content)
            .collect(),
        });
        extended_messages
    }

    fn subscribe(stream: &Self::ChatStream) -> Pollable {
        stream.subscribe()
    }
}

type DurableAnthropicComponent = DurableLLM<AnthropicComponent>;

golem_llm::export_llm!(DurableAnthropicComponent with_types_in golem_llm);
