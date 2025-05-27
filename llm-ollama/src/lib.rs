use std::cell::{Ref, RefCell, RefMut};

use client::{CompletionsRequest, OllamaApi};
use conversions::{messages_to_request, process_response};
use golem_llm::{
    chat_stream::{LlmChatStream, LlmChatStreamState},
    durability::{DurableLLM, ExtendedGuest},
    event_source::EventSource,
    golem::llm::llm::{
        ChatEvent, ChatStream, Config, ContentPart, Error, FinishReason, Guest, Message,
        ResponseMetadata, Role, StreamDelta, StreamEvent, ToolCall, ToolResult, Usage,
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
        trace!("Parsing NDJSON line: {raw}");
        let json: serde_json::Value =
            serde_json::from_str(raw.trim()).map_err(|e| format!("JSON parse error: {e}"))?;

        if json.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
            let input_tokens = json
                .get("prompt_eval_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let output_tokens = json.get("eval_count").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let timestamp = json
                .get("created_at")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let usage = Usage {
                input_tokens: Some(input_tokens),
                output_tokens: Some(input_tokens),
                total_tokens: Some(input_tokens + output_tokens),
            };

            let total_duration = json
                .get("total_duration")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let load_duration = json
                .get("load_duration")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let prompt_eval_duration = json
                .get("prompt_eval_duration")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let eval_duration = json
                .get("eval_duration")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let context = json
                .get("context")
                .cloned()
                .unwrap_or(serde_json::json!(null));

            let provider_metadata = serde_json::json!({
                "total_duration": total_duration,
                "load_duration": load_duration,
                "prompt_eval_duration": prompt_eval_duration,
                "eval_duration": eval_duration,
                "context": context
            })
            .to_string();

            return Ok(Some(StreamEvent::Finish(ResponseMetadata {
                finish_reason: Some(FinishReason::Stop),
                usage: Some(usage),
                provider_id: Some("ollama".to_string()),
                timestamp,
                provider_metadata_json: Some(provider_metadata),
            })));
        }

        if let Some(message) = json.get("message") {
            let mut content = Vec::new();
            let mut tool_calls = Vec::new();

            if let Some(text) = message.get("content").and_then(|c| c.as_str()) {
                if !text.is_empty() {
                    content.push(ContentPart::Text(text.to_string()));
                }
            }

            if let Some(calls) = message.get("tool_calls").and_then(|tc| tc.as_array()) {
                for call in calls {
                    if let Some(function) = call.get("function") {
                        let name = function
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let args_json = function
                            .get("arguments")
                            .cloned()
                            .unwrap_or(serde_json::json!({}));
                        let id = format!(
                            "ollama-{}",
                            json.get("created_at")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                                .unwrap_or_default()
                        );
                        tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments_json: args_json.to_string(),
                        });
                    }
                }
            }

            return Ok(Some(StreamEvent::Delta(StreamDelta {
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            })));
        }
        Ok(None)
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

        let client = OllamaApi::new(config.model.clone());
        match messages_to_request(messages, config.clone(), None) {
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

        let client = OllamaApi::new(config.model.clone());

        match messages_to_request(messages, config.clone(), Some(tool_results)) {
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

        let client = OllamaApi::new(config.model.clone());
        match messages_to_request(messages, config.clone(), None) {
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
