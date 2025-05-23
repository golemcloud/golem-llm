mod client;
mod conversions;

use crate::client::{ChatApi, ChatRequest, ChatResponse};
use crate::conversions::{messages_to_request, process_response, tool_results_to_messages};
use golem_llm::chat_stream::{LlmNdjsonChatStream, LlmNdjsonStreamState};
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, ContentPart, Error, ErrorCode, FinishReason, Guest, Message,
    ResponseMetadata, Role, StreamDelta, StreamEvent, ToolCall, ToolResult, Usage,
};
use golem_llm::ndjson_source::NdjsonSource;
use golem_llm::LOGGING_STATE;
use golem_rust::wasm_rpc::Pollable;
use log::trace;
use std::cell::{Ref, RefCell, RefMut};

// Environment variable to configure the Ollama API base URL
// Maybe Add a way to validate the URL
const BASE_URL_ENV: &str = "OLLAMA_BASE_URL";
// Default URL for local Ollama instance
const DEFAULT_BASE_URL: &str = "http://localhost:11434";

struct OllamaNdjsonChatStream {
    stream: RefCell<Option<NdjsonSource>>,
    failure: Option<Error>,
    finished: RefCell<bool>,
    response_metadata: RefCell<ResponseMetadata>,
}

impl OllamaNdjsonChatStream {
    pub fn new(stream: NdjsonSource) -> LlmNdjsonChatStream<Self> {
        LlmNdjsonChatStream::new(OllamaNdjsonChatStream {
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

    pub fn failed(error: Error) -> LlmNdjsonChatStream<Self> {
        LlmNdjsonChatStream::new(OllamaNdjsonChatStream {
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

impl LlmNdjsonStreamState for OllamaNdjsonChatStream {
    fn failure(&self) -> &Option<Error> {
        &self.failure
    }

    fn is_finished(&self) -> bool {
        *self.finished.borrow()
    }

    fn set_finished(&self) {
        *self.finished.borrow_mut() = true;
    }

    fn stream(&self) -> Ref<Option<NdjsonSource>> {
        self.stream.borrow()
    }

    fn stream_mut(&self) -> RefMut<Option<NdjsonSource>> {
        self.stream.borrow_mut()
    }

    fn decode_message(&self, raw: &str) -> Result<Option<StreamEvent>, String> {
        trace!("Received raw NDJSON event: {raw}");

        if raw.trim().is_empty() {
            return Ok(None);
        }

        // Parse as NDJSON - each line is a complete JSON object
        let response: ChatResponse = serde_json::from_str(raw)
            .map_err(|err| format!("Failed to parse NDJSON response: {err}"))?;

        // Handle streaming delta
        if !response.done {
            if !response.message.content.is_empty() {
                return Ok(Some(StreamEvent::Delta(StreamDelta {
                    content: Some(vec![ContentPart::Text(response.message.content)]),
                    tool_calls: None,
                })));
            }

            // Handle tool calls in streaming
            if let Some(tool_calls) = response.message.tool_calls {
                let mut llm_tool_calls = Vec::new();
                for tool_call in tool_calls {
                    llm_tool_calls.push(ToolCall {
                        id: generate_tool_call_id(),
                        name: tool_call.function.name,
                        arguments_json: serde_json::to_string(&tool_call.function.arguments)
                            .unwrap(),
                    });
                }
                return Ok(Some(StreamEvent::Delta(StreamDelta {
                    content: None,
                    tool_calls: Some(llm_tool_calls),
                })));
            }

            Ok(None)
        } else {
            // Final response - set metadata and finish
            let finish_reason = response
                .done_reason
                .as_ref()
                .map(|reason| match reason.as_str() {
                    "stop" => FinishReason::Stop,
                    "length" => FinishReason::Length,
                    _ => FinishReason::Other,
                });

            let usage = Usage {
                input_tokens: response.prompt_eval_count,
                output_tokens: response.eval_count,
                total_tokens: response
                    .prompt_eval_count
                    .zip(response.eval_count)
                    .map(|(input, output)| input + output),
            };

            let metadata = ResponseMetadata {
                finish_reason,
                usage: Some(usage),
                provider_id: Some(response.model),
                timestamp: Some(response.created_at),
                provider_metadata_json: None,
            };

            self.response_metadata.replace(metadata.clone());
            Ok(Some(StreamEvent::Finish(metadata)))
        }
    }
}

struct OllamaComponent;

impl OllamaComponent {
    fn get_base_url() -> String {
        std::env::var(BASE_URL_ENV).unwrap_or_else(|_| DEFAULT_BASE_URL.to_string())
    }

    fn request(client: ChatApi, request: ChatRequest) -> ChatEvent {
        match client.send_messages(request) {
            Ok(response) => process_response(response),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn streaming_request(
        client: ChatApi,
        mut request: ChatRequest,
    ) -> LlmNdjsonChatStream<OllamaNdjsonChatStream> {
        request.stream = Some(true);
        match client.stream_send_messages(request) {
            Ok(response) => {
                // Create NDJSON source for Ollama streaming
                match NdjsonSource::new(response) {
                    Ok(ndjson_source) => OllamaNdjsonChatStream::new(ndjson_source),
                    Err(err) => OllamaNdjsonChatStream::failed(Error {
                        code: ErrorCode::InternalError,
                        message: format!("Failed to create NDJSON source: {}", err),
                        provider_error_json: None,
                    }),
                }
            }
            Err(err) => OllamaNdjsonChatStream::failed(err),
        }
    }
}

impl Guest for OllamaComponent {
    type ChatStream = LlmNdjsonChatStream<OllamaNdjsonChatStream>;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let base_url = Self::get_base_url();
        let client = ChatApi::new(base_url);

        match messages_to_request(messages, config, &client) {
            Ok(request) => Self::request(client, request),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let base_url = Self::get_base_url();
        let client = ChatApi::new(base_url);

        match messages_to_request(messages, config, &client) {
            Ok(mut request) => {
                request
                    .messages
                    .extend(tool_results_to_messages(tool_results));
                Self::request(client, request)
            }
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for OllamaComponent {
    fn unwrapped_stream(
        messages: Vec<Message>,
        config: Config,
    ) -> LlmNdjsonChatStream<OllamaNdjsonChatStream> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let base_url = Self::get_base_url();
        let client = ChatApi::new(base_url);

        match messages_to_request(messages, config, &client) {
            Ok(request) => Self::streaming_request(client, request),
            Err(err) => OllamaNdjsonChatStream::failed(err),
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

fn generate_tool_call_id() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut hasher = DefaultHasher::new();
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .hash(&mut hasher);
    format!("call_{:x}", hasher.finish())
}

type DurableOllamaComponent = DurableLLM<OllamaComponent>;

golem_llm::export_llm!(DurableOllamaComponent with_types_in golem_llm);
