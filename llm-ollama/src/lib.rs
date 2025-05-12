use crate::client::{OllamaApi, OllamaChatCompletionChunk, OllamaChatCompletionRequest};
use crate::conversions::{messages_to_request, process_response, tool_results_to_messages};
use golem_llm::chat_stream::{LlmChatStream, LlmChatStreamState};
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, ContentPart, Error, FinishReason, Guest, Message,
    ResponseMetadata, Role, StreamDelta, StreamEvent, ToolCall, ToolResult, Usage,
};
use golem_llm::LOGGING_STATE;
use golem_rust::wasm_rpc::Pollable;
use log::trace;
use std::cell::{Ref, RefCell, RefMut};

mod client;
mod conversions;

// This implementation uses Server-Sent Events (SSE) from the OpenAI compatibility Ollama API
// and converts them to the format expected by golem-llm.
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
        trace!("Received raw Ollama stream event: {raw}");

        // With the OpenAI compatibility API, we parse the chunks as OllamaChatCompletionChunk
        let chunk: OllamaChatCompletionChunk = serde_json::from_str(raw).map_err(|err| {
            format!("Failed to deserialize Ollama stream chunk: {err} - raw: {raw}")
        })?;

        // Process the choices in the chunk
        if let Some(choice) = chunk.choices.first() {
            // Handle content delta
            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    return Ok(Some(StreamEvent::Delta(StreamDelta {
                        content: Some(vec![ContentPart::Text(content.clone())]),
                        tool_calls: None,
                    })));
                }
            }

            // Handle tool call delta
            // We have implemented tool calls in the OpenAI compatibility API, but they are not supported in Ollama yet
            // The code is here for when the support is added to Ollama, this should work as expected
            if let Some(tool_calls) = &choice.delta.tool_calls {
                if !tool_calls.is_empty() {
                    let golem_tool_calls = tool_calls
                        .iter()
                        .map(|tc| ToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments_json: tc.function.arguments.clone(),
                        })
                        .collect();

                    return Ok(Some(StreamEvent::Delta(StreamDelta {
                        content: None,
                        tool_calls: Some(golem_tool_calls),
                    })));
                }
            }

            // Handle finish reason
            if let Some(finish_reason) = &choice.finish_reason {
                let finish_reason_enum = match finish_reason.as_str() {
                    "stop" => FinishReason::Stop,
                    "length" => FinishReason::Length,
                    "tool_calls" => FinishReason::ToolCalls,
                    "content_filter" => FinishReason::ContentFilter,
                    _ => FinishReason::Other,
                };

                // Placeholder for usage, not available in Ollama yet
                let usage = Usage {
                    input_tokens: None,
                    output_tokens: None,
                    total_tokens: None,
                };

                // Generate provider metadata for durability
                let provider_metadata_json = Some(format!(
                    r#"{{"id":"{}","created":{}}}"#,
                    chunk.id, chunk.created
                ));

                return Ok(Some(StreamEvent::Finish(ResponseMetadata {
                    finish_reason: Some(finish_reason_enum),
                    usage: Some(usage),
                    provider_id: Some(chunk.id),
                    timestamp: Some(chunk.created.to_string()),
                    provider_metadata_json,
                })));
            }
        }

        // If we didn't find anything to process, return None
        Ok(None)
    }
}

struct OllamaComponent;

impl OllamaComponent {
    // Environment variable to configure the Ollama API base URL
    const BASE_URL_ENV_VAR: &'static str = "OLLAMA_BASE_URL";

    // Default URL for local Ollama instance with OpenAI compatibility API
    const DEFAULT_BASE_URL: &'static str = "http://localhost:11434/v1";

    fn request(client: OllamaApi, request: OllamaChatCompletionRequest) -> ChatEvent {
        match client.create_chat_completion(request) {
            Ok(response) => process_response(response),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn streaming_request(
        client: OllamaApi,
        mut request: OllamaChatCompletionRequest,
    ) -> LlmChatStream<OllamaChatStream> {
        request.stream = true;
        match client.stream_chat_completion(request) {
            Ok(stream) => OllamaChatStream::new(stream),
            Err(err) => OllamaChatStream::failed(err),
        }
    }
}

impl Guest for OllamaComponent {
    type ChatStream = LlmChatStream<OllamaChatStream>;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        // Initialize logging
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        // Get base URL from environment variable or use default
        let base_url = std::env::var(Self::BASE_URL_ENV_VAR)
            .unwrap_or_else(|_| Self::DEFAULT_BASE_URL.to_string());
        let client = OllamaApi::with_base_url(base_url);

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
        // Initialize logging
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        // Get base URL from environment variable or use default
        let base_url = std::env::var(Self::BASE_URL_ENV_VAR)
            .unwrap_or_else(|_| Self::DEFAULT_BASE_URL.to_string());
        let client = OllamaApi::with_base_url(base_url);

        match messages_to_request(messages, config, &client) {
            Ok(mut request) => {
                // Add tool results as additional messages
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

// Implementation of the ExtendedGuest trait which is required for
// the DurableLLM wrapper to work correctly.
impl ExtendedGuest for OllamaComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> LlmChatStream<OllamaChatStream> {
        // Initialize logging
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        // Get base URL from environment variable or use default
        let base_url = std::env::var(Self::BASE_URL_ENV_VAR)
            .unwrap_or_else(|_| Self::DEFAULT_BASE_URL.to_string());
        let client = OllamaApi::with_base_url(base_url);

        match messages_to_request(messages, config, &client) {
            Ok(request) => Self::streaming_request(client, request),
            Err(err) => OllamaChatStream::failed(err),
        }
    }

    fn subscribe(stream: &Self::ChatStream) -> Pollable {
        stream.subscribe()
    }

    fn retry_prompt(original_messages: &[Message], partial_result: &[StreamDelta]) -> Vec<Message> {
        let mut extended_messages = Vec::new();

        // Add a system message explaining the interruption
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

        // Include the original conversation
        extended_messages.extend_from_slice(original_messages);

        // Convert the partial result to content that can be included in the prompt
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

        // Add a message showing the partial response received
        extended_messages.push(Message {
            role: Role::System,
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
}

type DurableOllamaComponent = DurableLLM<OllamaComponent>;
golem_llm::export_llm!(DurableOllamaComponent with_types_in golem_llm);
