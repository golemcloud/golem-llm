use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamOutput as AwsConverseStreamOutput;
use aws_sdk_bedrockruntime::types::ConverseStreamOutput as AwsConverseStreamEventVariant;
use golem_llm::golem::llm::llm::{
    ContentPart, Error as LlmError, ErrorCode, FinishReason, ResponseMetadata, StreamDelta,
    StreamEvent as GolemStreamEvent, ToolCall, Usage,
};
use log::{debug, error, trace, warn};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// State for tracking tool call fragments during streaming
#[derive(Default, Debug, Clone)]
struct ToolCallFragment {
    id: String,
    name: String,
    arguments_json: String,
}

// This struct will wrap the AWS SDK's event receiver.
pub struct BedrockSdkStreamWrapper {
    // Store the whole stream output to work with its public interface
    stream_output: AwsConverseStreamOutput,
    // State to accumulate usage information from Metadata events
    accumulated_usage: Option<Usage>,
    // State for message ID from MessageStart events
    message_id: Option<String>,
    // State for tracking tool calls by content block index
    tool_call_fragments: HashMap<i32, ToolCallFragment>,
}

impl BedrockSdkStreamWrapper {
    pub fn new(aws_stream_output: AwsConverseStreamOutput) -> Self {
        BedrockSdkStreamWrapper {
            stream_output: aws_stream_output,
            accumulated_usage: None,
            message_id: None,
            tool_call_fragments: HashMap::new(),
        }
    }

    // This method will attempt to pull the next event from the AWS stream
    // and convert it into a GolemStreamEvent.
    // It needs to be async to call stream methods.
    pub async fn next_golem_event(&mut self) -> Option<Result<GolemStreamEvent, LlmError>> {
        // Use the stream() method to get access to the receiver
        match self.stream_output.stream.recv().await {
            Ok(Some(aws_event)) => {
                trace!("Received AWS SDK stream event: {aws_event:?}");
                self.convert_aws_event_to_golem_stream_event(aws_event)
            }
            Ok(None) => {
                debug!("AWS SDK stream ended.");
                None // Stream ended
            }
            Err(sdk_err) => {
                error!("AWS SDK stream error: {sdk_err:?}");
                let code = ErrorCode::InternalError;
                let message = format!("SDK Stream Error: {sdk_err:?}");
                Some(Err(LlmError {
                    code,
                    message,
                    provider_error_json: Some(format!("{sdk_err:?}")),
                }))
            }
        }
    }

    fn convert_aws_event_to_golem_stream_event(
        &mut self,
        aws_event: AwsConverseStreamEventVariant,
    ) -> Option<Result<GolemStreamEvent, LlmError>> {
        match aws_event {
            AwsConverseStreamEventVariant::ContentBlockStart(start_event) => {
                let content_block_index = start_event.content_block_index();

                // Check if this is a tool use start
                if let Some(start) = start_event.start() {
                    if let Ok(tool_use_start) = start.as_tool_use() {
                        let tool_id = tool_use_start.tool_use_id().to_string();
                        let tool_name = tool_use_start.name().to_string();

                        trace!(
                            "Tool use started: id={tool_id}, name={tool_name}, index={content_block_index}"
                        );

                        // Store the tool call fragment
                        self.tool_call_fragments.insert(
                            content_block_index,
                            ToolCallFragment {
                                id: tool_id,
                                name: tool_name,
                                arguments_json: String::new(),
                            },
                        );
                    }
                }

                None // Don't emit an event for ContentBlockStart
            }
            AwsConverseStreamEventVariant::ContentBlockDelta(delta) => {
                match delta.delta() {
                    Some(aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(text_delta)) => {
                        Some(Ok(GolemStreamEvent::Delta(StreamDelta {
                            content: Some(vec![ContentPart::Text(text_delta.to_string())]),
                            tool_calls: None,
                        })))
                    }
                    Some(aws_sdk_bedrockruntime::types::ContentBlockDelta::ToolUse(tool_delta)) => {
                        // Get the content block index from the delta event
                        let content_block_index = delta.content_block_index();

                        if let Some(fragment) =
                            self.tool_call_fragments.get_mut(&content_block_index)
                        {
                            // Accumulate the partial JSON input
                            let input_delta = tool_delta.input();
                            fragment.arguments_json.push_str(input_delta);

                            trace!(
                                "Accumulated tool input for index {}: {} chars",
                                content_block_index,
                                fragment.arguments_json.len()
                            );
                        } else {
                            warn!(
                                "Received ToolUse delta for unknown content block index: {content_block_index}"
                            );
                        }

                        None // Don't emit an event for partial tool input
                    }
                    Some(_) => {
                        warn!("Unhandled ContentBlockDelta variant");
                        None
                    }
                    None => {
                        warn!("ContentBlockDelta has no delta content");
                        None
                    }
                }
            }
            AwsConverseStreamEventVariant::ContentBlockStop(stop_event) => {
                let content_block_index = stop_event.content_block_index();

                // check for a completed tool call
                if let Some(fragment) = self.tool_call_fragments.remove(&content_block_index) {
                    trace!(
                        "Tool call completed: id={}, name={}, args={}",
                        fragment.id,
                        fragment.name,
                        fragment.arguments_json
                    );

                    // Emit the completed tool call
                    Some(Ok(GolemStreamEvent::Delta(StreamDelta {
                        content: None,
                        tool_calls: Some(vec![ToolCall {
                            id: fragment.id,
                            name: fragment.name,
                            arguments_json: fragment.arguments_json,
                        }]),
                    })))
                } else {
                    None // No tool call to complete
                }
            }
            AwsConverseStreamEventVariant::MessageStop(stop_event) => {
                // Extract stop reason and map to golem-llm FinishReason
                let finish_reason = match stop_event.stop_reason() {
                    aws_sdk_bedrockruntime::types::StopReason::EndTurn => Some(FinishReason::Stop),
                    aws_sdk_bedrockruntime::types::StopReason::StopSequence => {
                        Some(FinishReason::Stop)
                    }
                    aws_sdk_bedrockruntime::types::StopReason::MaxTokens => {
                        Some(FinishReason::Length)
                    }
                    aws_sdk_bedrockruntime::types::StopReason::ToolUse => {
                        Some(FinishReason::ToolCalls)
                    }
                    aws_sdk_bedrockruntime::types::StopReason::ContentFiltered => {
                        Some(FinishReason::ContentFilter)
                    }
                    aws_sdk_bedrockruntime::types::StopReason::GuardrailIntervened => {
                        Some(FinishReason::ContentFilter)
                    }
                    _ => {
                        warn!(
                            "Unknown stop reason from Bedrock: {:?}",
                            stop_event.stop_reason()
                        );
                        Some(FinishReason::Other)
                    }
                };

                // Extract additional metadata if available
                let provider_metadata_json = stop_event
                    .additional_model_response_fields()
                    .map(|doc| format!("{doc:?}")); // Convert Document to string representation

                // Generate timestamp
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs().to_string())
                    .ok();

                Some(Ok(GolemStreamEvent::Finish(ResponseMetadata {
                    finish_reason,
                    usage: self.accumulated_usage.take(), // Use accumulated usage and clear it
                    provider_id: self.message_id.clone(), // Use message ID from MessageStart
                    timestamp,
                    provider_metadata_json,
                })))
            }
            AwsConverseStreamEventVariant::MessageStart(start_event) => {
                // Extract role information from MessageStart event
                let role = start_event.role().as_str();
                self.message_id = Some(format!("bedrock_{role}_message"));

                trace!("Message started with role: {role}");
                None // Don't emit an event for MessageStart
            }
            AwsConverseStreamEventVariant::Metadata(metadata_event) => {
                // Extract and store usage information for later use in MessageStop
                if let Some(usage) = metadata_event.usage() {
                    self.accumulated_usage = Some(Usage {
                        input_tokens: Some(usage.input_tokens() as u32),
                        output_tokens: Some(usage.output_tokens() as u32),
                        total_tokens: Some(usage.total_tokens() as u32),
                    });
                }
                None // Don't emit an event for Metadata
            }
            _ => {
                warn!("Unknown or unhandled AWS Bedrock stream event variant encountered.");
                None
            }
        }
    }
}
