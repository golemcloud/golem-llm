mod client;
mod conversions;

use std::cell::{Ref, RefCell, RefMut};

use golem_llm::chat_stream::{LlmChatStream, LlmChatStreamState};
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, Error as LlmError, ErrorCode, Guest, Message, ResponseMetadata, StreamEvent, ToolCall, ToolResult,
};
use golem_llm::LOGGING_STATE;
use log::{debug, info, error};

use client::BedrockClient;
use conversions::{
    bedrock_response_to_chat_event,
    messages_to_bedrock_body,
};
use wit_bindgen_rt::async_support::block_on;

const BEDROCK_DEFAULT_ACCEPT: &str = "application/json";
const BEDROCK_DEFAULT_CONTENT_TYPE: &str = "application/json";


// Custom stream implementation for Bedrock
pub struct BedrockChatStream {
    stream: RefCell<Option<EventSource>>,
    failure: Option<LlmError>,
    finished: RefCell<bool>,
}

impl BedrockChatStream {
    pub fn new(stream: EventSource) -> LlmChatStream<Self> {
        LlmChatStream::new(BedrockChatStream {
            stream: RefCell::new(Some(stream)),
            failure: None,
            finished: RefCell::new(false),
        })
    }

    pub fn failed(error: LlmError) -> LlmChatStream<Self> {
        LlmChatStream::new(BedrockChatStream {
            stream: RefCell::new(None),
            failure: Some(error),
            finished: RefCell::new(false),
        })
    }
}

impl LlmChatStreamState for BedrockChatStream {
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
        self.stream.borrow()
    }

    fn stream_mut(&self) -> RefMut<Option<EventSource>> {
        self.stream.borrow_mut()
    }

    fn decode_message(&self, _raw: &str) -> Result<Option<StreamEvent>, String> {
        // TODO: Implement Bedrock-specific message decoding
        Ok(None)
    }
}

impl Drop for BedrockChatStream {
    fn drop(&mut self) {
        debug!("BedrockChatStream dropped");
    }
}


struct BedrockComponent;

impl BedrockComponent {
    // Helper to get region from provider_options or default AWS SDK behavior
    fn get_aws_region(config: &Config) -> Option<String> {
        config.provider_options.iter()
            .find(|kv| kv.key.eq_ignore_ascii_case("AWS_REGION") || kv.key.eq_ignore_ascii_case("REGION"))
            .map(|kv| kv.value.clone())
    }
}

impl Guest for BedrockComponent {
    type ChatStream = LlmChatStream<BedrockChatStream>;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        info!("Bedrock: send called. Model: {}", config.model);

        block_on(async move {
            let aws_region = Self::get_aws_region(&config);
            match BedrockClient::new(aws_region).await {
                Ok(client) => {
                    match messages_to_bedrock_body(&messages, None, &config) {
                        Ok(body) => {
                            match client
                                .invoke_model(
                                    config.model.clone(),
                                    body,
                                    BEDROCK_DEFAULT_ACCEPT.to_string(),
                                    BEDROCK_DEFAULT_CONTENT_TYPE.to_string(),
                                )
                                .await
                            {
                                Ok(response_json) => bedrock_response_to_chat_event(
                                    response_json,
                                    &config.model,
                                    ResponseMetadata {
                                        finish_reason: None,
                                        usage: None,
                                        provider_id: None,
                                        timestamp: None,
                                        provider_metadata_json: None,
                                    },
                                ),
                                Err(e) => ChatEvent::Error(e),
                            }
                        }
                        Err(e) => ChatEvent::Error(e),
                    }
                }
                Err(e) => ChatEvent::Error(e),
            }
        })
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        info!("Bedrock: continue_ called. Model: {}", config.model);
        block_on(async move {
            let aws_region = Self::get_aws_region(&config);
            match BedrockClient::new(aws_region).await {
                Ok(client) => {
                     match messages_to_bedrock_body(&messages, Some(&tool_results), &config) {
                        Ok(body) => {
                            match client
                                .invoke_model(
                                    config.model.clone(),
                                    body,
                                    BEDROCK_DEFAULT_ACCEPT.to_string(),
                                    BEDROCK_DEFAULT_CONTENT_TYPE.to_string(),
                                )
                                .await
                            {
                                Ok(response_json) => bedrock_response_to_chat_event(
                                    response_json,
                                    &config.model,
                                    ResponseMetadata {
                                        finish_reason: None,
                                        usage: None,
                                        provider_id: None,
                                        timestamp: None,
                                        provider_metadata_json: None,
                                    },
                                ),
                                Err(e) => ChatEvent::Error(e),
                            }
                        }
                        Err(e) => ChatEvent::Error(e),
                    }
                }
                Err(e) => ChatEvent::Error(e),
            }
        })
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for BedrockComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> Self::ChatStream {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        info!("Bedrock: stream called. Model: {}", config.model);

        // The async block will attempt to set up the stream and resolve
        // to Result<EventSource, LlmError>.
        // However, due to the mismatch between AWS SDK stream and EventSource requirements,
        // it will currently return an Err.
        let result_for_event_source: Result<EventSource, LlmError> = block_on(async move {
            let aws_region = Self::get_aws_region(&config);
            let client = BedrockClient::new(aws_region).await?; // Returns LlmError on failure

            let body_json = messages_to_bedrock_body(&messages, None, &config)?; // Returns LlmError

            debug!("Requesting stream from Bedrock. Model: {}, Body: {}", config.model, body_json.to_string());

            let sdk_stream_output = client
                .invoke_model_with_response_stream(
                    config.model.clone(),
                    body_json,
                    BEDROCK_DEFAULT_ACCEPT.to_string(), // "application/json"
                    BEDROCK_DEFAULT_CONTENT_TYPE.to_string(), // "application/json"
                )
                .await?; // This is LlmError from client.rs with mapped SDK errors

            // sdk_stream_output is InvokeModelWithResponseStreamOutput
            // - sdk_stream_output.body is EventReceiver<PayloadChunk, _>
            // - sdk_stream_output.content_type is Option<String> (expected to be "application/json")

            // PROBLEM POINT:
            // golem_llm::event_source::EventSource::new() expects a `reqwest::Response`
            // and internally checks for `Content-Type: text/event-stream`.
            // The AWS SDK's `sdk_stream_output` does not directly provide a `reqwest::Response`.
            // Also, Bedrock's `InvokeModelWithResponseStream` sends a stream of JSON objects,
            // (matching the `Accept: application/json` header) not a standard SSE text/event-stream.
            // This means EventSource's SSE parser would not be suitable for these JSON objects directly.

            // This step requires a significant adaptation layer or a change in how
            // LlmChatStreamState consumes streams (i.e., not being tied to EventSource for SSE).
            // For now, we return an error indicating this unimplemented/mismatched part.
            Err(LlmError {
                code: ErrorCode::InternalError, // Or perhaps ErrorCode::Unsupported if this adaptation is deemed out of scope
                message: "Streaming from Bedrock SDK to golem-llm EventSource not yet fully implemented due to API/type mismatch.".to_string(),
                provider_error_json: Some(format!(
                    "Bedrock stream content_type: {:?}. EventSource expects 'text/event-stream' and a reqwest::Response.",
                    sdk_stream_output.content_type
                )),
            })

            // If adaptation was possible, it would look something like:
            // let adapted_response_for_event_source = adapt_sdk_stream_to_reqwest_response(sdk_stream_output)?;
            // EventSource::new(adapted_response_for_event_source)
            //     .map_err(|e| LlmError { /* convert event_source::error::Error to LlmError */ })
        });

        match result_for_event_source {
            Ok(event_source) => BedrockChatStream::new(event_source),
            Err(llm_error) => {
                error!("Failed to setup Bedrock stream: {}", llm_error.message);
                BedrockChatStream::failed(llm_error)
            }
        }
    }

    // Default retry_prompt from golem-llm/src/durability.rs is fine unless Bedrock needs a very specific format.
}

// Wrap with DurableLLM
type DurableBedrockComponent = DurableLLM<BedrockComponent>;

// Export the durable component
golem_llm::export_llm!(DurableBedrockComponent with_types_in golem_llm);