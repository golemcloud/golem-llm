// Placeholder for Bedrock LLM provider library

// TODO: Implement the golem:llm/llm@1.0.0 interface
// using types from the `golem_llm::golem::llm::llm` module (or generated bindings)

// mod client;
// mod conversions;

// struct BedrockComponent;

// TODO: Implement Guest and ExtendedGuest traits for BedrockComponent

// TODO: Export the component using golem_llm::export_llm! macro 

use std::cell::{RefCell, Ref, RefMut};
use std::sync::OnceLock;

use async_trait::async_trait;
use log::{debug, error, trace, warn};

use aws_config::SdkConfig as AwsSdkConfig;
use aws_sdk_bedrockruntime::{Client as BedrockClient};

// Assuming these types are re-exported or accessible from golem_llm crate
// use golem_llm::chat_stream::{LlmChatStream, LlmChatStreamState};
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, Error as LlmError, ErrorCode as LlmErrorCode, Guest, Message,
    StreamDelta, StreamEvent, ToolCall, ToolResult, ContentPart,
};
use golem_llm::event_source::EventSource; // May not be directly used if AWS SDK stream is different
use golem_llm::LOGGING_STATE; // For initializing logging

// Local modules
mod client;
mod conversions;

// Global static for AWS SDK Config and Bedrock Client (initialized once)
// This is a common pattern for expensive-to-create clients.
// However, `aws_config::load_from_env()` is async.
// We need a way to initialize this, potentially blocking on first use,
// or using a Golem-specific mechanism if available for async initialization of statics.
// For now, let's assume client::get_bedrock_client() handles this.
// static AWS_CONFIG: OnceLock<AwsSdkConfig> = OnceLock::new();
// static BEDROCK_CLIENT: OnceLock<BedrockClient> = OnceLock::new();

// Placeholder for the actual stream from AWS SDK
// This will likely be `aws_sdk_bedrockruntime::output::ConverseStreamOutput`
// or the stream type it produces.
type AwsBedrockStream = aws_sdk_bedrockruntime::output::converse_stream::Receiver;

// HYPOTHETICAL Golem utility to bridge sync WIT calls to async Rust functions
fn sync_await<F, T>(future: F) -> Result<T, LlmError>
where
    F: std::future::Future<Output = Result<T, LlmError>> + Send + 'static,
    T: Send + 'static,
{
    warn!("Using HYPOTHETICAL sync_await. This needs to be replaced with a Golem-specific async bridge.");
    // This is a HACK placeholder for compilation and conceptual design.
    // It CANNOT be the actual implementation in a Golem component.
    #[cfg(not(target_arch = "wasm32"))] 
    {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().map_err(|e| LlmError { code: LlmErrorCode::InternalError, message: format!("Failed to build tokio runtime for sync_await: {}",e), provider_error_json: None})?;
        rt.block_on(future)
    }
    #[cfg(target_arch = "wasm32")]
    {
        panic!("sync_await placeholder reached on wasm32. Golem runtime bridge needed.");
    }
}

pub struct BedrockChatStream {
    stream_receiver: RefCell<Option<client::AwsBedrockStreamReceiver>>,
    failure: RefCell<Option<LlmError>>,
    finished: RefCell<bool>,
    buffered_events: RefCell<Vec<StreamEvent>>,
    // We need a pollable for blocking_get_next if we don't use EventSource
    // This is tricky because the AWS SDK stream doesn't directly expose a wasi pollable.
    // We might need to create a synthetic one or have sync_await handle timed polling.
    // For now, let's assume blocking_get_next will be inefficient or needs more thought.
    // Or, a dummy pollable that is always ready, relying on get_next to eventually return something.
    dummy_pollable: golem_rust::wasm_rpc::bindings::wasi::io::poll::Pollable, 
    last_seen_usage: RefCell<Option<golem_llm::golem::llm::llm::Usage>>, // Added to store usage from MetadataEvent
}

impl BedrockChatStream {
    pub fn new(receiver: client::AwsBedrockStreamReceiver) -> Self {
        let dummy_pollable = golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0); // immediately ready
        Self {
            stream_receiver: RefCell::new(Some(receiver)),
            failure: RefCell::new(None),
            finished: RefCell::new(false),
            buffered_events: RefCell::new(Vec::new()),
            dummy_pollable,
            last_seen_usage: RefCell::new(None),
        }
    }

    pub fn new_failed(error: LlmError) -> Self {
        let dummy_pollable = golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0);
        Self {
            stream_receiver: RefCell::new(None),
            failure: RefCell::new(Some(error)),
            finished: RefCell::new(true),
            buffered_events: RefCell::new(Vec::new()),
            dummy_pollable,
            last_seen_usage: RefCell::new(None),
        }
    }

    // Internal helper to poll the stream and process one event, used by get_next
    fn poll_and_decode_next_event(&self) -> Result<Option<StreamEvent>, LlmError> {
        if *self.finished.borrow() {
            return Ok(None);
        }

        if self.stream_receiver.borrow().is_none() {
            *self.finished.borrow_mut() = true;
            return self.failure.borrow().clone().map_or(Ok(None), Err);
        }

        let mut receiver_mut = self.stream_receiver.borrow_mut();
        let receiver = receiver_mut.as_mut().expect("Stream receiver gone");

        match sync_await(client::receive_next_stream_event(receiver)) {
            Ok(Some(aws_event)) => {
                // Directly handle MetadataEvent here for usage, before passing to stateless conversion fn
                if let aws_sdk_bedrockruntime::types::ConverseStreamOutput::Metadata(ref metadata_event) = aws_event {
                    if let Some(usage) = metadata_event.usage() {
                        *self.last_seen_usage.borrow_mut() = conversions::bedrock_usage_to_golem_usage(Some(usage));
                        // Metadata event itself doesn't directly map to a Golem StreamEvent to be returned now,
                        // its data is used later.
                        return Ok(None); 
                    }
                }

                match conversions::bedrock_converse_stream_output_to_golem_stream_event(aws_event) {
                    Ok(Some(mut golem_event)) => {
                        if let StreamEvent::Finish(ref mut metadata) = golem_event {
                            // Populate usage if we have seen it from a MetadataEvent
                            if metadata.usage.is_none() {
                                metadata.usage = self.last_seen_usage.borrow_mut().take();
                            }
                            *self.finished.borrow_mut() = true;
                        } else if matches!(golem_event, StreamEvent::Error(_)) {
                            *self.finished.borrow_mut() = true;
                        }
                        Ok(Some(golem_event))
                    }
                    Ok(None) => Ok(None), 
                    Err(e) => {
                        *self.finished.borrow_mut() = true;
                        *self.failure.borrow_mut() = Some(e.clone());
                        Err(e)
                    }
                }
            }
            Ok(None) => {
                *self.finished.borrow_mut() = true;
                // If stream ends and we have pending usage, create a final Finish event with it.
                // This case might be hit if MessageStop didn't come but stream ended.
                if let Some(usage) = self.last_seen_usage.borrow_mut().take() {
                    Ok(Some(StreamEvent::Finish(golem_llm::golem::llm::llm::ResponseMetadata{
                        finish_reason: Some(golem_llm::golem::llm::llm::FinishReason::Stop), // Assuming Stop if not specified
                        usage: Some(usage),
                        provider_id: None,
                        timestamp: Some(chrono::Utc::now().to_rfc3339()),
                        provider_metadata_json: None,
                    })))
                } else {
                    Ok(None)
                }
            }
            Err(e) => {
                *self.finished.borrow_mut() = true;
                *self.failure.borrow_mut() = Some(e.clone());
                Err(e)
            }
        }
    }
}

impl golem_llm::golem::llm::llm::GuestChatStream for BedrockChatStream {
    fn get_next(&self) -> Option<Vec<StreamEvent>> {
        if *self.finished.borrow() && self.buffered_events.borrow().is_empty() {
            return Some(vec![]); // Finished and buffer empty
        }

        if !self.buffered_events.borrow().is_empty() {
            return Some(self.buffered_events.borrow_mut().drain(..).collect());
        }
        
        // If buffer is empty, try to poll for a new event
        match self.poll_and_decode_next_event() {
            Ok(Some(event)) => Some(vec![event]),
            Ok(None) => {
                // Stream might have ended, or it was a non-event.
                // If finished flag is set by poll_and_decode, return empty to signal end.
                if *self.finished.borrow() { Some(vec![]) } else { None } // None means try again
            }
            Err(err) => {
                // Failure already set by poll_and_decode_next_event
                Some(vec![StreamEvent::Error(err)])
            }
        }
    }

    fn blocking_get_next(&self) -> Vec<StreamEvent> {
        loop {
            match self.get_next() {
                Some(events) => {
                    if !events.is_empty() || *self.finished.borrow() {
                        return events;
                    }
                    // If events is empty but not finished, means get_next() returned None effectively
                    // and we should poll again.
                }
                None => { 
                    // get_next() returning None means try polling again.
                }
            }
            // If get_next() returned Some([]) and not finished, or None, we need to block/yield.
            // This dummy pollable will always be ready immediately.
            // A real implementation would need a way to get a pollable from the AWS stream
            // or have sync_await integrate with a host-provided pollable.
            self.dummy_pollable.block(); 
        }
    }
}

struct BedrockComponent;

impl BedrockComponent {
    fn make_sync_request<Fut, T>(
        future: Fut,
    ) -> Result<T, LlmError>
    where
        Fut: std::future::Future<Output = Result<T, LlmError>> + Send + 'static,
        T: Send + 'static,
    {
        sync_await(future)
    }
}

impl golem_llm::golem::llm::llm::Guest for BedrockComponent {
    type ChatStream = BedrockChatStream;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        trace!("BedrockComponent Guest::send called with config: {:?}", config);

        match Self::make_sync_request(client::perform_converse_request(
            messages,
            config,
            None,
        )) {
            Ok(chat_event) => chat_event,
            Err(err) => {
                error!("Error in Bedrock send request: {:?}", err);
                ChatEvent::Error(err)
            }
        }
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        trace!("BedrockComponent Guest::continue_ called with config: {:?}", config);
        match Self::make_sync_request(client::perform_converse_request(
            messages,
            config,
            Some(tool_results),
        )) {
            Ok(chat_event) => chat_event,
            Err(err) => {
                error!("Error in Bedrock continue_ request: {:?}", err);
                ChatEvent::Error(err)
            }
        }
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        trace!("BedrockComponent Guest::stream called with config: {:?}", config);
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl golem_llm::golem::llm::llm::ExtendedGuest for BedrockComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> Self::ChatStream {
        trace!("BedrockComponent ExtendedGuest::unwrapped_stream called");
        match Self::make_sync_request(client::perform_converse_stream_request(messages, config)) {
            Ok(stream_receiver) => BedrockChatStream::new(stream_receiver),
            Err(err) => {
                error!("Error in Bedrock streaming request: {:?}", err);
                BedrockChatStream::new_failed(err)
            }
        }
    }

    fn create_retry_prompt(
        original_messages: &[Message],
        original_config: &Config, 
        partial_response: Option<&[StreamDelta]>,
    ) -> (Vec<Message>, Config) {
        debug!("BedrockComponent ExtendedGuest::create_retry_prompt called");
        let mut new_messages = original_messages.to_vec();
        let mut new_config = original_config.clone();

        if let Some(deltas) = partial_response {
            let mut assistant_reply_parts: Vec<ContentPart> = Vec::new();
            // TODO: This logic needs to be robust for different kinds of deltas (text, tool_call start/delta/end)
            // and how Bedrock models expect continuation prompts (e.g. Claude).
            for delta in deltas {
                if let Some(content_parts) = &delta.content {
                    assistant_reply_parts.extend(content_parts.iter().cloned());
                }
            }

            if !assistant_reply_parts.is_empty() {
                new_messages.push(Message {
                    role: golem_llm::golem::llm::llm::Role::Assistant,
                    name: None,
                    content: assistant_reply_parts,
                });
            }
        }
        (new_messages, new_config)
    }
}

// Export the component, wrapped in DurableLLM
// The `with_types_in golem_llm` part maps the WIT types.
type DurableBedrockComponent = DurableLLM<BedrockComponent>;
golem_llm::export_llm!(DurableBedrockComponent with_types_in golem_llm); 

use chrono::Utc; // Add Utc for timestamp 