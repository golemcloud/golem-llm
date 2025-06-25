mod client;
mod conversions;

use crate::client::{BedrockClient, ConverseRequest};
use crate::conversions::{
    convert_usage, messages_to_request, process_response, stop_reason_to_finish_reason,
    tool_results_to_messages,
};
use golem_llm::chat_stream::{LlmChatStream, LlmChatStreamState};
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, ContentPart, Error, ErrorCode, Guest, Message, ResponseMetadata,
    Role, StreamDelta, StreamEvent, ToolCall, ToolResult,
};
use golem_llm::LOGGING_STATE;
use golem_rust::wasm_rpc::Pollable;
use log::trace;
use serde_json::Value;
use std::cell::{Ref, RefCell, RefMut};

struct BedrockChatStream {
    stream: RefCell<Option<EventSource>>,
    failure: Option<Error>,
    finished: RefCell<bool>,
    response_metadata: RefCell<ResponseMetadata>,
}

impl BedrockChatStream {
    pub fn new(stream: EventSource) -> LlmChatStream<Self> {
        LlmChatStream::new(BedrockChatStream {
            stream: RefCell::new(Some(stream)),
            failure: None,
            finished: RefCell::new(false),
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
        LlmChatStream::new(BedrockChatStream {
            stream: RefCell::new(None),
            failure: Some(error),
            finished: RefCell::new(false),
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

impl LlmChatStreamState for BedrockChatStream {
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
        
        let json: Value = serde_json::from_str(raw)
            .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;

        if let Some(content_block_delta) = json.get("contentBlockDelta") {
            if let Some(delta) = content_block_delta.get("delta") {
                if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                    return Ok(Some(StreamEvent::Delta(StreamDelta {
                        content: Some(vec![ContentPart::Text(text.to_string())]),
                        tool_calls: None,
                    })));
                }
            }
        }

        if let Some(content_block_start) = json.get("contentBlockStart") {
            if let Some(start) = content_block_start.get("start") {
                if let Some(tool_use) = start.get("toolUse") {
                    if let (Some(tool_use_id), Some(name)) = (
                        tool_use.get("toolUseId").and_then(|v| v.as_str()),
                        tool_use.get("name").and_then(|v| v.as_str()),
                    ) {
                        if let Some(input) = tool_use.get("input") {
                            return Ok(Some(StreamEvent::Delta(StreamDelta {
                                content: None,
                                tool_calls: Some(vec![ToolCall {
                                    id: tool_use_id.to_string(),
                                    name: name.to_string(),
                                    arguments_json: serde_json::to_string(input).unwrap(),
                                }]),
                            })));
                        }
                    }
                }
            }
        }

        if let Some(metadata) = json.get("metadata") {
            if let Some(usage) = metadata.get("usage") {
                if let Ok(bedrock_usage) = serde_json::from_value::<crate::client::Usage>(usage.clone()) {
                    self.response_metadata.borrow_mut().usage = Some(convert_usage(bedrock_usage));
                }
            }
        }

        if let Some(message_stop) = json.get("messageStop") {
            if let Some(stop_reason) = message_stop.get("stopReason").and_then(|v| v.as_str()) {
                let stop_reason = match stop_reason {
                    "end_turn" => crate::client::StopReason::EndTurn,
                    "tool_use" => crate::client::StopReason::ToolUse,
                    "max_tokens" => crate::client::StopReason::MaxTokens,
                    "stop_sequence" => crate::client::StopReason::StopSequence,
                    "guardrail_intervened" => crate::client::StopReason::GuardrailIntervened,
                    "content_filtered" => crate::client::StopReason::ContentFiltered,
                    _ => crate::client::StopReason::EndTurn,
                };
                self.response_metadata.borrow_mut().finish_reason = Some(stop_reason_to_finish_reason(stop_reason));
            }

            let response_metadata = self.response_metadata.borrow().clone();
            return Ok(Some(StreamEvent::Finish(response_metadata)));
        }

        Ok(None)
    }
}

struct BedrockComponent;

impl BedrockComponent {
    const ACCESS_KEY_ID_ENV_VAR: &'static str = "AWS_ACCESS_KEY_ID";
    const SECRET_ACCESS_KEY_ENV_VAR: &'static str = "AWS_SECRET_ACCESS_KEY";
    const REGION_ENV_VAR: &'static str = "AWS_REGION";

    fn get_client() -> Result<BedrockClient, Error> {
        let access_key_id = std::env::var(Self::ACCESS_KEY_ID_ENV_VAR)
            .map_err(|_| Error {
                code: ErrorCode::AuthenticationFailed,
                message: format!("Missing environment variable: {}", Self::ACCESS_KEY_ID_ENV_VAR),
                provider_error_json: None,
            })?;

        let secret_access_key = std::env::var(Self::SECRET_ACCESS_KEY_ENV_VAR)
            .map_err(|_| Error {
                code: ErrorCode::AuthenticationFailed,
                message: format!("Missing environment variable: {}", Self::SECRET_ACCESS_KEY_ENV_VAR),
                provider_error_json: None,
            })?;

        let region = std::env::var(Self::REGION_ENV_VAR)
            .unwrap_or_else(|_| "us-east-1".to_string());

        Ok(BedrockClient::new(access_key_id, secret_access_key, region))
    }

    fn request(client: BedrockClient, model_id: &str, request: ConverseRequest) -> ChatEvent {
        match client.converse(model_id, request) {
            Ok(response) => process_response(response),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn streaming_request(
        client: BedrockClient,
        model_id: &str,
        request: ConverseRequest,
    ) -> LlmChatStream<BedrockChatStream> {
        match client.converse_stream(model_id, request) {
            Ok(stream) => BedrockChatStream::new(stream),
            Err(err) => BedrockChatStream::failed(err),
        }
    }
}

impl Guest for BedrockComponent {
    type ChatStream = LlmChatStream<BedrockChatStream>;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        
        let client = match Self::get_client() {
            Ok(client) => client,
            Err(err) => return ChatEvent::Error(err),
        };

        match messages_to_request(messages, config.clone()) {
            Ok(request) => Self::request(client, &config.model, request),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let client = match Self::get_client() {
            Ok(client) => client,
            Err(err) => return ChatEvent::Error(err),
        };

        match messages_to_request(messages, config.clone()) {
            Ok(mut request) => {
                request.messages.extend(tool_results_to_messages(tool_results));
                Self::request(client, &config.model, request)
            }
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for BedrockComponent {
    fn unwrapped_stream(
        messages: Vec<Message>,
        config: Config,
    ) -> LlmChatStream<BedrockChatStream> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let client = match Self::get_client() {
            Ok(client) => client,
            Err(err) => return BedrockChatStream::failed(err),
        };

        match messages_to_request(messages, config.clone()) {
            Ok(request) => Self::streaming_request(client, &config.model, request),
            Err(err) => BedrockChatStream::failed(err),
        }
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

type DurableBedrockComponent = DurableLLM<BedrockComponent>;

golem_llm::export_llm!(DurableBedrockComponent with_types_in golem_llm);