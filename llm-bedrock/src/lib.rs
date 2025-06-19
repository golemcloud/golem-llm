mod client;
mod conversions;
mod stream_bridge;

use golem_llm::chat_stream::{LlmChatStream, LlmChatStreamState};
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, Error as LlmError, ErrorCode, Guest, Message, StreamEvent,
    ToolCall, ToolResult,
};
use golem_llm::LOGGING_STATE;
use log::{debug, error, info, trace, warn};
use std::cell::{Ref, RefCell, RefMut};

use client::{AwsCredentials, BedrockClient, BedrockRequest};
use conversions::{
    bedrock_converse_to_chat_event, extract_model_config, messages_to_bedrock_converse,
    tool_results_to_bedrock_messages,
};
use stream_bridge::BedrockSdkStreamWrapper;

// Bedrock stream implementation
pub struct BedrockChatStreamState {
    // keep a dummy EventSource for framework compatibility
    dummy_stream: RefCell<Option<EventSource>>,
    failure: Option<LlmError>,
    finished: RefCell<bool>,
    // Background task handle
    _stream_wrapper: RefCell<Option<BedrockSdkStreamWrapper>>,
}

impl BedrockChatStreamState {
    pub fn new(stream_wrapper: BedrockSdkStreamWrapper) -> LlmChatStream<Self> {
        LlmChatStream::new(BedrockChatStreamState {
            dummy_stream: RefCell::new(None),
            failure: None,
            finished: RefCell::new(false),
            _stream_wrapper: RefCell::new(Some(stream_wrapper)),
        })
    }

    pub fn failed(error: LlmError) -> LlmChatStream<Self> {
        LlmChatStream::new(BedrockChatStreamState {
            dummy_stream: RefCell::new(None),
            failure: Some(error),
            finished: RefCell::new(true),
            _stream_wrapper: RefCell::new(None),
        })
    }
}

impl LlmChatStreamState for BedrockChatStreamState {
    fn failure(&self) -> &Option<LlmError> {
        &self.failure
    }

    fn is_finished(&self) -> bool {
        *self.finished.borrow()
    }

    fn set_finished(&self) {
        *self.finished.borrow_mut() = true;
    }

    fn stream(&self) -> Ref<Option<EventSource>> {
        // Return the dummy stream - for now this won't be used
        self.dummy_stream.borrow()
    }

    fn stream_mut(&self) -> RefMut<Option<EventSource>> {
        // Return the dummy stream - for now this won't be used
        self.dummy_stream.borrow_mut()
    }

    fn decode_message(&self, raw: &str) -> Result<Option<StreamEvent>, String> {
        // this method is used for JSON parsing of events generate ourselves
        trace!("Bedrock decode_message: {raw}");

        if raw.trim().is_empty() {
            return Ok(None);
        }

        // Try to parse as JSON and convert back to StreamEvent
        match serde_json::from_str::<serde_json::Value>(raw) {
            Ok(json) => {
                if let Some(event_type) = json.get("type").and_then(|t| t.as_str()) {
                    match event_type {
                        "delta" => {
                            let mut content = None;
                            let mut tool_calls = None;

                            if let Some(text) = json.get("content").and_then(|c| c.as_str()) {
                                content =
                                    Some(vec![golem_llm::golem::llm::llm::ContentPart::Text(
                                        text.to_string(),
                                    )]);
                            }

                            if let Some(tc_array) =
                                json.get("tool_calls").and_then(|tc| tc.as_array())
                            {
                                let parsed_tool_calls: Vec<ToolCall> = tc_array
                                    .iter()
                                    .filter_map(|tc| {
                                        if let (Some(id), Some(name), Some(args)) = (
                                            tc.get("id").and_then(|i| i.as_str()),
                                            tc.get("name").and_then(|n| n.as_str()),
                                            tc.get("arguments").and_then(|a| a.as_str()),
                                        ) {
                                            Some(ToolCall {
                                                id: id.to_string(),
                                                name: name.to_string(),
                                                arguments_json: args.to_string(),
                                            })
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();
                                if !parsed_tool_calls.is_empty() {
                                    tool_calls = Some(parsed_tool_calls);
                                }
                            }

                            Ok(Some(StreamEvent::Delta(
                                golem_llm::golem::llm::llm::StreamDelta {
                                    content,
                                    tool_calls,
                                },
                            )))
                        }
                        "finish" => {
                            let finish_reason = json
                                .get("finish_reason")
                                .and_then(|fr| fr.as_str())
                                .map(|fr_str| match fr_str {
                                    "stop" => golem_llm::golem::llm::llm::FinishReason::Stop,
                                    "length" => golem_llm::golem::llm::llm::FinishReason::Length,
                                    "tool_calls" => {
                                        golem_llm::golem::llm::llm::FinishReason::ToolCalls
                                    }
                                    "content_filter" => {
                                        golem_llm::golem::llm::llm::FinishReason::ContentFilter
                                    }
                                    "error" => golem_llm::golem::llm::llm::FinishReason::Error,
                                    _ => golem_llm::golem::llm::llm::FinishReason::Other,
                                });

                            let usage =
                                json.get("usage")
                                    .map(|u| golem_llm::golem::llm::llm::Usage {
                                        input_tokens: u
                                            .get("input_tokens")
                                            .and_then(|it| it.as_u64())
                                            .map(|v| v as u32),
                                        output_tokens: u
                                            .get("output_tokens")
                                            .and_then(|ot| ot.as_u64())
                                            .map(|v| v as u32),
                                        total_tokens: u
                                            .get("total_tokens")
                                            .and_then(|tt| tt.as_u64())
                                            .map(|v| v as u32),
                                    });

                            let provider_id = json
                                .get("provider_id")
                                .and_then(|pid| pid.as_str())
                                .map(|s| s.to_string());

                            Ok(Some(StreamEvent::Finish(
                                golem_llm::golem::llm::llm::ResponseMetadata {
                                    finish_reason,
                                    usage,
                                    provider_id,
                                    timestamp: None,
                                    provider_metadata_json: None,
                                },
                            )))
                        }
                        "error" => {
                            let message = json
                                .get("message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("Unknown error")
                                .to_string();

                            Ok(Some(StreamEvent::Error(LlmError {
                                code: ErrorCode::InternalError,
                                message,
                                provider_error_json: None,
                            })))
                        }
                        _ => {
                            warn!("Unknown event type in decode_message: {event_type}");
                            Ok(None)
                        }
                    }
                } else {
                    warn!("No event type in JSON: {raw}");
                    Ok(None)
                }
            }
            Err(e) => {
                warn!("Failed to parse JSON in decode_message: {raw} - Error: {e}");
                Ok(None)
            }
        }
    }
}

impl Drop for BedrockChatStreamState {
    fn drop(&mut self) {
        debug!("BedrockChatStreamState dropped");
    }
}

struct BedrockComponent;

impl BedrockComponent {
    fn get_aws_region(config: &Config) -> Option<String> {
        config
            .provider_options
            .iter()
            .find(|kv| {
                kv.key.eq_ignore_ascii_case("AWS_REGION") || kv.key.eq_ignore_ascii_case("REGION")
            })
            .map(|kv| kv.value.clone())
    }

    fn get_aws_credentials(config: &Config) -> Option<AwsCredentials> {
        let access_key_id = config
            .provider_options
            .iter()
            .find(|kv| kv.key.eq_ignore_ascii_case("AWS_ACCESS_KEY_ID"))
            .map(|kv| kv.value.clone())?;

        let secret_access_key = config
            .provider_options
            .iter()
            .find(|kv| kv.key.eq_ignore_ascii_case("AWS_SECRET_ACCESS_KEY"))
            .map(|kv| kv.value.clone())?;

        let session_token = config
            .provider_options
            .iter()
            .find(|kv| kv.key.eq_ignore_ascii_case("AWS_SESSION_TOKEN"))
            .map(|kv| kv.value.clone());

        Some(AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token,
        })
    }

    fn create_client(config: &Config) -> Result<BedrockClient, LlmError> {
        let region = Self::get_aws_region(config);
        let credentials = Self::get_aws_credentials(config);
        BedrockClient::new(region, credentials)
    }

    fn create_bedrock_request(
        messages: &[Message],
        config: &Config,
    ) -> Result<BedrockRequest, LlmError> {
        let bedrock_messages = messages_to_bedrock_converse(messages)?;
        let (max_tokens, temperature, system_prompt) = extract_model_config(config);

        Ok(BedrockRequest {
            model_id: config.model.clone(),
            messages: bedrock_messages,
            system_prompt,
            max_tokens,
            temperature,
        })
    }

    fn request(client: BedrockClient, request: BedrockRequest) -> ChatEvent {
        let model_id = request.model_id.clone();
        match client.converse(request) {
            Ok(response) => {
                bedrock_converse_to_chat_event(response, &model_id).unwrap_or_else(ChatEvent::Error)
            }
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn streaming_request(
        client: BedrockClient,
        request: BedrockRequest,
    ) -> LlmChatStream<BedrockChatStreamState> {
        match client.converse_stream(request) {
            Ok(aws_stream_output) => {
                debug!("Successfully created AWS Bedrock stream");
                let stream_wrapper = stream_bridge::BedrockSdkStreamWrapper::new(aws_stream_output);
                BedrockChatStreamState::new(stream_wrapper)
            }
            Err(err) => {
                error!("Failed to create Bedrock stream: {err:?}");
                BedrockChatStreamState::failed(err)
            }
        }
    }
}

impl Guest for BedrockComponent {
    type ChatStream = LlmChatStream<BedrockChatStreamState>;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        info!("Bedrock: send called. Model: {}", config.model);

        let client = match Self::create_client(&config) {
            Ok(client) => client,
            Err(err) => return ChatEvent::Error(err),
        };

        let request = match Self::create_bedrock_request(&messages, &config) {
            Ok(request) => request,
            Err(err) => return ChatEvent::Error(err),
        };

        Self::request(client, request)
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        info!("Bedrock: continue_ called. Model: {}", config.model);

        let client = match Self::create_client(&config) {
            Ok(client) => client,
            Err(err) => return ChatEvent::Error(err),
        };

        // Convert original messages to Bedrock format
        let mut bedrock_messages = match messages_to_bedrock_converse(&messages) {
            Ok(msgs) => msgs,
            Err(err) => return ChatEvent::Error(err),
        };

        // Add tool results as additional messages
        let tool_result_messages = tool_results_to_bedrock_messages(tool_results);
        bedrock_messages.extend(tool_result_messages);

        let (max_tokens, temperature, system_prompt) = extract_model_config(&config);

        let request = BedrockRequest {
            model_id: config.model.clone(),
            messages: bedrock_messages,
            system_prompt,
            max_tokens,
            temperature,
        };

        Self::request(client, request)
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for BedrockComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> Self::ChatStream {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        info!("Bedrock: stream called. Model: {}", config.model);

        let client = match Self::create_client(&config) {
            Ok(client) => client,
            Err(err) => return BedrockChatStreamState::failed(err),
        };

        let request = match Self::create_bedrock_request(&messages, &config) {
            Ok(request) => request,
            Err(err) => return BedrockChatStreamState::failed(err),
        };

        Self::streaming_request(client, request)
    }
}

type DurableBedrockComponent = DurableLLM<BedrockComponent>;

golem_llm::export_llm!(DurableBedrockComponent with_types_in golem_llm);
