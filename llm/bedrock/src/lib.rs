mod client;
mod conversions;

use crate::client::{BedrockClient, ConverseRequest};
use crate::conversions::{
    convert_usage, messages_to_request, process_response, stop_reason_to_finish_reason,
    tool_results_to_messages,
};
use golem_llm::chat_stream::{LlmChatStream, LlmChatStreamState};
use golem_llm::config::with_config_keys;
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, ContentPart, Error, Guest, Message, ResponseMetadata,
    Role, StreamDelta, StreamEvent, ToolCall, ToolResult,
};
use golem_llm::LOGGING_STATE;
use golem_rust::wasm_rpc::Pollable;
use log::trace;
use serde::Deserialize;
use serde_json::Value;
use std::cell::{Ref, RefCell, RefMut};

struct BedrockChatStream {
    stream: RefCell<Option<EventSource>>,
    failure: Option<Error>,
    finished: RefCell<bool>,
    response_metadata: RefCell<ResponseMetadata>,
}


/// [2025-06-29T18:11:10.458Z] [TRACE   ] [golem_llm_bedrock] llm/bedrock/src/lib.rs:84: Received raw stream event: 
/// {
/// "contentBlockIndex":1,
/// "delta":{
/// "toolUse":{
/// "input":" 10
/// }"}},
/// "p":"abcdefghijklmnopqrstuvwxyzAB"
/// }
/// {
/// "contentBlockIndex":0,
/// "delta":
/// {
/// "text":" German"
/// },
/// "p":"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX"
/// }
/// 

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Delta {
    ToolUse {
        #[serde(rename = "toolUse")]
         tool_use: ToolUse,
    },
    Text {
         text: String,
    },
}

#[derive(Debug, Deserialize)]
pub struct ToolUse {
    pub input: String,
}

// Additional structs for different message types
#[derive(Debug, Deserialize)]
pub struct MessageStart {
    pub p: String,
    pub role: String,
}

#[derive(Debug, Deserialize)]
pub struct MessageStop {
    pub p: String,
    #[serde(rename = "stopReason")]
    pub stop_reason: String,
}

#[derive(Debug, Deserialize)]
pub struct MetadataMessage {
    pub p: String,
    pub usage: Option<crate::client::Usage>,
    pub metrics: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct EventContentBlock {
    #[serde(rename = "contentBlockIndex")]
    pub content_block_index: u32,
    pub delta : Delta,
    pub p: String,
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
        
        // 1. Handle content block delta messages (contentBlockIndex + delta)
        if json.get("contentBlockIndex").is_some() && json.get("delta").is_some() {
            match serde_json::from_value::<EventContentBlock>(json.clone()) {
                Ok(event_content_block) => {
                    match event_content_block.delta {
                        Delta::Text { text } => {
                            return Ok(Some(StreamEvent::Delta(StreamDelta {
                                content: Some(vec![ContentPart::Text(text)]),
                                tool_calls: None,
                            })));
                        }
                        Delta::ToolUse { tool_use } => {
                            // Handle tool use delta - this would need tool call ID and name from earlier message
                            // For now, just return the input as text
                            return Ok(Some(StreamEvent::Delta(StreamDelta {
                                content: Some(vec![ContentPart::Text(tool_use.input)]),
                                tool_calls: None,
                            })));
                        }
                    }
                }
                Err(err) => {
                    trace!("Failed to parse as EventContentBlock: {}", err);
                    // Continue to other parsing attempts
                }
            }
        }

        // 3. Handle message start (role + p)
        if json.get("role").is_some() {
            if let Ok(_message_start) = serde_json::from_value::<MessageStart>(json.clone()) {
                // Message start event - just metadata, no content to return
                return Ok(None);
            }
        }

        // 4. Handle message stop with stopReason
        if json.get("stopReason").is_some() {
            if let Ok(message_stop) = serde_json::from_value::<MessageStop>(json.clone()) {
                let stop_reason = match message_stop.stop_reason.as_str() {
                    "end_turn" => crate::client::StopReason::EndTurn,
                    "tool_use" => crate::client::StopReason::ToolUse,
                    "max_tokens" => crate::client::StopReason::MaxTokens,
                    "stop_sequence" => crate::client::StopReason::StopSequence,
                    "guardrail_intervened" => crate::client::StopReason::GuardrailIntervened,
                    "content_filtered" => crate::client::StopReason::ContentFiltered,
                    _ => crate::client::StopReason::EndTurn,
                };
                self.response_metadata.borrow_mut().finish_reason =
                    Some(stop_reason_to_finish_reason(stop_reason));

                let response_metadata = self.response_metadata.borrow().clone();
                return Ok(Some(StreamEvent::Finish(response_metadata)));
            }
        }

        // 5. Handle metadata messages with usage/metrics
        if json.get("usage").is_some() || json.get("metrics").is_some() {
            if let Ok(metadata) = serde_json::from_value::<MetadataMessage>(json.clone()) {
                if let Some(usage) = metadata.usage {
                    self.response_metadata.borrow_mut().usage = Some(convert_usage(usage));
                }
                // Metadata processed, no event to return
                return Ok(None);
            }
        }
        Ok(None)
    }
}

struct BedrockComponent;

impl BedrockComponent {
    const ACCESS_KEY_ID_ENV_VAR: &'static str = "AWS_ACCESS_KEY_ID";
    const SECRET_ACCESS_KEY_ENV_VAR: &'static str = "AWS_SECRET_ACCESS_KEY";
    const REGION_ENV_VAR: &'static str = "AWS_REGION";

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
        with_config_keys(
            &[
                Self::ACCESS_KEY_ID_ENV_VAR,
                Self::SECRET_ACCESS_KEY_ENV_VAR,
                Self::REGION_ENV_VAR,
            ],
            ChatEvent::Error,
            |bedrock_api_keys| {
                let client = BedrockClient::new(
                    bedrock_api_keys[Self::ACCESS_KEY_ID_ENV_VAR].clone(),
                    bedrock_api_keys[Self::SECRET_ACCESS_KEY_ENV_VAR].clone(),
                    bedrock_api_keys[Self::REGION_ENV_VAR].clone(),
                );

                match messages_to_request(messages, config.clone()) {
                    Ok(request) => Self::request(client, &config.model, request),
                    Err(err) => ChatEvent::Error(err),
                }
            },
        )
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_keys(
            &[
                Self::ACCESS_KEY_ID_ENV_VAR,
                Self::SECRET_ACCESS_KEY_ENV_VAR,
                Self::REGION_ENV_VAR,
            ],
            ChatEvent::Error,
            |bedrock_api_keys| {
                let client = BedrockClient::new(
                    bedrock_api_keys[Self::ACCESS_KEY_ID_ENV_VAR].clone(),
                    bedrock_api_keys[Self::SECRET_ACCESS_KEY_ENV_VAR].clone(),
                    bedrock_api_keys[Self::REGION_ENV_VAR].clone(),
                );

                match messages_to_request(messages, config.clone()) {
                    Ok(mut request) => {
                        request
                            .messages
                            .extend(tool_results_to_messages(tool_results));
                        Self::request(client, &config.model, request)
                    }
                    Err(err) => ChatEvent::Error(err),
                }
            },
        )
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

        with_config_keys(
            &[
                Self::ACCESS_KEY_ID_ENV_VAR,
                Self::SECRET_ACCESS_KEY_ENV_VAR,
                Self::REGION_ENV_VAR,
            ],
            BedrockChatStream::failed,
            |bedrock_api_keys| {
                let client = BedrockClient::new(
                    bedrock_api_keys[Self::ACCESS_KEY_ID_ENV_VAR].clone(),
                    bedrock_api_keys[Self::SECRET_ACCESS_KEY_ENV_VAR].clone(),
                    bedrock_api_keys[Self::REGION_ENV_VAR].clone(),
                );

                match messages_to_request(messages, config.clone()) {
                    Ok(request) => Self::streaming_request(client, &config.model, request),
                    Err(err) => BedrockChatStream::failed(err),
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

type DurableBedrockComponent = DurableLLM<BedrockComponent>;

golem_llm::export_llm!(DurableBedrockComponent with_types_in golem_llm);
