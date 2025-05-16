use std::cell::{Ref, RefCell, RefMut};

use client::{CompletionsRequest, CompletionsResponse, MessageRequest, OllamaApi};
use conversions::{messages_to_request, process_response};
use golem_llm::{
    chat_stream::{LlmChatStream, LlmChatStreamState},
    durability::{DurableLLM, ExtendedGuest},
    event_source::EventSource,
    golem::llm::llm::{
        ChatEvent, ChatStream, Config, ContentPart, Error, FinishReason, Guest, Message,
        ResponseMetadata, Role, StreamDelta, StreamEvent, ToolCall, ToolResult,
    },
    LOGGING_STATE,
};
use golem_rust::wasm_rpc::Pollable;
use log::trace;

mod client;
mod conversions;

struct OllamaChatStream {
    stream: RefCell<Option<EventSource>>,
    failure: Option<Error>,
    finished: RefCell<bool>,
}

impl OllamaChatStream {
    pub fn new(stream: EventSource) -> LlmChatStream<Self> {
        LlmChatStream::new(OllamaChatStream {
            stream: RefCell::new(Some(stream)),
            failure: None,
            finished: RefCell::new(false),
        })
    }

    pub fn failed(error: Error) -> LlmChatStream<Self> {
        LlmChatStream::new(OllamaChatStream {
            stream: RefCell::new(None),
            failure: Some(error),
            finished: RefCell::new(false),
        })
    }
}

impl LlmChatStreamState for OllamaChatStream {
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

        if json.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
            let done_reasone = match json
                .get("done_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("")
            {
                "stop" => Some(FinishReason::Stop),
                _ => Some(FinishReason::Other),
            };

            return Ok(Some(StreamEvent::Finish(ResponseMetadata {
                finish_reason: done_reasone,
                usage: None,
                provider_id: None,
                timestamp: None,
                provider_metadata_json: None,
            })));
        }

        let response = serde_json::from_value::<CompletionsResponse>(json.clone())
            .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;
        let mut content: Vec<golem_llm::exports::golem::llm::llm::ContentPart> = Vec::new();
        let mut tool_calls: Vec<golem_llm::exports::golem::llm::llm::ToolCall> = Vec::new();
        if response.messages.is_some() {
            for message in response.messages.unwrap() {
                if message.content.is_some() {
                    content.push(ContentPart::Text(message.content.unwrap_or_default()));
                }

                if message.tool_calls.is_some() {
                    let mut message_tool_calls: Vec<golem_llm::exports::golem::llm::llm::ToolCall> =
                        message
                            .tool_calls
                            .unwrap()
                            .iter()
                            .map(|tc| golem_llm::exports::golem::llm::llm::ToolCall {
                                id: tc.id.clone(),
                                name: tc.function.as_ref().unwrap().name.clone(),
                                arguments_json: tc.function.as_ref().unwrap().arguments.to_string(),
                            })
                            .collect();
                    tool_calls.append(&mut message_tool_calls);
                }
            }
        }

        Ok(Some(StreamEvent::Delta(StreamDelta {
            content: Some(content),
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
        })))
    }
}

struct OllamaComponent;

impl OllamaComponent {
    fn request(client: &OllamaApi, request: CompletionsRequest) -> ChatEvent {
        match client.send_chat(request) {
            Ok(response) => process_response(response),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn streaming_request(
        client: &OllamaApi,
        mut request: CompletionsRequest,
    ) -> LlmChatStream<OllamaChatStream> {
        request.stream = Some(true);
        match client.send_chat_stream(request) {
            Ok(stream) => OllamaChatStream::new(stream),
            Err(err) => OllamaChatStream::failed(err),
        }
    }
}

impl Guest for OllamaComponent {
    type ChatStream = LlmChatStream<OllamaChatStream>;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let client = OllamaApi::new(config.model.clone(), None);
        match messages_to_request(messages, config.clone()) {
            Ok(request) => Self::request(&client, request),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        let client = OllamaApi::new(config.model.clone(), None);
        match messages_to_request(messages, config.clone()) {
            Ok(request) => Self::request(&client, request),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config.clone()))
    }
}

impl ExtendedGuest for OllamaComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> LlmChatStream<OllamaChatStream> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let client = OllamaApi::new(config.model.clone(), None);
        match messages_to_request(messages, config.clone()) {
            Ok(request) => Self::streaming_request(&client, request),
            Err(err) => OllamaChatStream::failed(err),
        }
    }

    fn retry_prompt(original_messages: &[Message], partial_result: &[StreamDelta]) -> Vec<Message> {
        let mut extended_messages = Vec::new();

        extended_messages.push(Message {
            role: Role::System,
            name: None,
            content: vec![ContentPart::Text(
                "You were asked the same question previously, but the response was interrupted before completion. \
                 Please continue your response from where you left off. \
                 Do not include the part of the response that was already seen."
                    .to_string(),
            )],
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

type DurableOllamaComponent = DurableLLM<OllamaComponent>;

golem_llm::export_llm!(DurableOllamaComponent with_types_in golem_llm);
