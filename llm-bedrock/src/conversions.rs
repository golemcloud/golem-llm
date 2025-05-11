// Placeholder for data conversion logic

// TODO: Implement functions to convert between Golem LLM interface types
// (e.g., Message, Config, ToolCall, StreamEvent)
// and AWS Bedrock API specific types. 

use crate::LlmError, LlmErrorCode; // Assuming these are pub use from lib.rs or golem_llm
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::types::{ 
    ConversationRole as BedrockRole,
    Message as BedrockMessage,
    ContentBlock as BedrockContentBlock,
    ToolConfiguration as BedrockToolConfiguration,
    ToolSpecification as BedrockToolSpecification,
    ToolInputSchema as BedrockToolInputSchema,
    // For Converse API specifically
    ConverseRequest,
    InferenceConfiguration as BedrockInferenceConfiguration,
    // For Converse Stream Output
    ConverseStreamOutput,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageStartEvent,
    MessageStopEvent,
    MetadataEvent,
    ToolResultBlock,
    ToolUseBlock,
    ContentBlockDelta as BedrockContentBlockDelta,
    ContentBlockStart as BedrockContentBlockStart,
    TokenUsage as BedrockTokenUsage,
    StopReason as BedrockStopReason,
};
use aws_sdk_bedrockruntime::output::ConverseOutput;
use golem_llm::golem::llm::llm::{
    ChatEvent, Config as LlmConfig, ContentPart as GolemContentPart, FinishReason,
    ImageDetail as GolemImageDetail, ImageUrl as GolemImageUrl, Message as GolemMessage,
    ResponseMetadata, Role as GolemRole, StreamDelta, StreamEvent, ToolCall as GolemToolCall,
    ToolDefinition as GolemToolDefinition, ToolResult as GolemToolResult,
    // Potentially Kv for provider_options, Usage, etc.
    Usage as GolemUsage,
};
use serde_json::json; // For constructing JSON blobs for Bedrock
use std::collections::HashMap;

// Helper to convert Golem Role to Bedrock ConversationRole string
fn golem_to_bedrock_role(role: &GolemRole) -> BedrockRole {
    match role {
        GolemRole::User => BedrockRole::User,
        GolemRole::Assistant => BedrockRole::Assistant,
        // Bedrock's Converse API uses User/Assistant. System prompts go elsewhere.
        // Tool roles are handled via specific message structures for tool use/results.
        _ => {
            log::warn!("Unsupported GolemRole {:?} for Bedrock ConversationRole, defaulting to User", role);
            BedrockRole::User 
        }
    }
}

// Helper to convert Golem ContentPart to Bedrock ContentBlock
// This might be more complex if Bedrock requires specific structuring for image content etc.
fn golem_to_bedrock_content_blocks(parts: &[GolemContentPart]) -> Result<Vec<BedrockContentBlock>, LlmError> {
    let mut blocks = Vec::new();
    for part in parts {
        match part {
            GolemContentPart::Text(text) => {
                blocks.push(BedrockContentBlock::Text(text.clone()));
            }
            GolemContentPart::Image(image_url) => {
                // Bedrock's `ContentBlock::Image` needs an `ImageBlock` which has format and source.
                // Source can be bytes or s3.
                // This requires more specific handling based on how GolemImageUrl provides image data.
                // For now, this is a simplified placeholder assuming URL directly maps if supported,
                // or this path needs significant enhancement.
                // Anthropic models via Bedrock expect base64 encoded images typically.
                // Example: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQSkZJRg..."}}
                return Err(LlmError {
                    code: LlmErrorCode::InvalidRequest,
                    message: "Image content part for Bedrock needs specific formatting (e.g., base64) not yet implemented".to_string(),
                    provider_error_json: None,
                });
            }
        }
    }
    Ok(blocks)
}

fn golem_tools_to_bedrock_tool_config(tools: &[GolemToolDefinition]) -> Result<Option<BedrockToolConfiguration>, LlmError> {
    if tools.is_empty() {
        return Ok(None);
    }
    let mut bedrock_tools = Vec::new();
    for tool_def in tools {
        let input_schema_doc = match serde_json::from_str::<serde_json::Value>(&tool_def.parameters_schema) {
            Ok(json_schema_val) => aws_sdk_bedrockruntime::types::Document::Object(json_schema_val.as_object().ok_or_else(|| LlmError{code: LlmErrorCode::InvalidRequest, message: "Tool schema must be a JSON object".to_string(), provider_error_json: None })?.clone()),
            Err(e) => return Err(LlmError { 
                code: LlmErrorCode::InvalidRequest, 
                message: format!("Failed to parse tool parameters schema for tool '{}': {}", tool_def.name, e), 
                provider_error_json: Some(tool_def.parameters_schema.clone())
            })
        };
        let input_schema = BedrockToolInputSchema::Json(input_schema_doc);

        bedrock_tools.push(aws_sdk_bedrockruntime::types::Tool::ToolSpecification(
            BedrockToolSpecification::builder()
                .name(tool_def.name.clone())
                .description(tool_def.description.clone().unwrap_or_default()) 
                .input_schema(input_schema)
                .build()
                .map_err(|e| LlmError{code: LlmErrorCode::InternalError, message: format!("Failed to build ToolSpecification: {}", e), provider_error_json: None})?
        ));
    }
    Ok(Some(BedrockToolConfiguration::builder().set_tools(Some(bedrock_tools)).build().map_err(|e| LlmError{code: LlmErrorCode::InternalError, message: format!("Failed to build ToolConfiguration: {}", e), provider_error_json: None})?))
}

/// Converts Golem LLM request parts into a Bedrock Converse API request body Blob.
/// This function constructs the JSON body that Bedrock expects for models like Claude.
pub(crate) fn golem_to_bedrock_converse_request_body(
    messages: Vec<GolemMessage>,
    llm_config: &LlmConfig,
    tool_results: Option<Vec<(GolemToolCall, GolemToolResult)>>,
) -> Result<Blob, LlmError> {
    // System prompt handling: Bedrock Converse API takes a top-level `system` field.
    let mut system_prompts = Vec::new();
    let mut user_assistant_messages = Vec::new();

    for msg in messages {
        if msg.role == GolemRole::System {
            // Assuming single text part for system prompt for now
            if let Some(GolemContentPart::Text(text)) = msg.content.first() {
                system_prompts.push(BedrockContentBlock::Text(text.clone()));
            } else {
                return Err(LlmError { 
                    code: LlmErrorCode::InvalidRequest, 
                    message: "System message must contain a single text part".to_string(), 
                    provider_error_json: None 
                });
            }
        } else {
            user_assistant_messages.push(msg);
        }
    }
    
    // Construct Bedrock messages from user/assistant Golem messages
    let mut bedrock_messages: Vec<BedrockMessage> = user_assistant_messages
        .into_iter()
        .map(|msg| {
            Ok(BedrockMessage::builder()
                .role(golem_to_bedrock_role(&msg.role))
                .set_content(Some(golem_to_bedrock_content_blocks(&msg.content)?))
                .build()
                .map_err(|e| LlmError{code: LlmErrorCode::InternalError, message: format!("Failed to build BedrockMessage: {}", e), provider_error_json: None})?)
        })
        .collect::<Result<Vec<_>, LlmError>>()?;

    // Handle tool results by converting them to appropriate Bedrock messages
    if let Some(results) = tool_results {
        for (tool_call_info, tool_call_result) in results {
            // Bedrock expects tool results to follow a tool_use block from the assistant.
            // The user message then contains ContentBlock::ToolResult.
            let tool_id = tool_call_info.id; // Bedrock ToolResultBlock needs tool_use_id
            
            let tool_result_block = match tool_call_result {
                GolemToolResult::Success(s) => {
                    let output_json_val: serde_json::Value = serde_json::from_str(&s.result_json)
                        .map_err(|e| LlmError { 
                            code: LlmErrorCode::InvalidRequest, 
                            message: format!("Tool result for '{}' is not valid JSON: {}", s.name, e), 
                            provider_error_json: Some(s.result_json.clone()) 
                        })?;
                    ToolResultBlock::builder()
                        .tool_use_id(tool_id)
                        .set_content(Some(vec![BedrockContentBlock::Json(aws_sdk_bedrockruntime::types::Document::Object(output_json_val.as_object().unwrap().clone()))]))
                        .build()
                        .map_err(|e| LlmError{code: LlmErrorCode::InternalError, message: format!("Failed to build ToolResultBlock: {}", e), provider_error_json: None})?
                }
                GolemToolResult::Error(e) => {
                    // TODO: How to represent tool error status in Bedrock if it has specific field?
                    // For now, putting error message as text content.
                    let error_content = BedrockContentBlock::Text(format!("Tool '{}' failed: {}", e.name, e.error_message));
                    ToolResultBlock::builder()
                        .tool_use_id(tool_id)
                        .set_content(Some(vec![error_content]))
                        .build()
                        .map_err(|e| LlmError{code: LlmErrorCode::InternalError, message: format!("Failed to build ToolResultBlock (error): {}", e), provider_error_json: None})?
                }
            };
            // Add a user message containing the tool result block
            bedrock_messages.push(
                BedrockMessage::builder()
                    .role(BedrockRole::User) // Tool results are sent as user messages
                    .set_content(Some(vec![BedrockContentBlock::ToolResult(tool_result_block)]))
                    .build()
                    .map_err(|e| LlmError{code: LlmErrorCode::InternalError, message: format!("Failed to build tool result BedrockMessage: {}", e), provider_error_json: None})?
            );
        }
    }

    let mut inference_config_builder = BedrockInferenceConfiguration::builder();
    if let Some(mt) = llm_config.max_tokens { inference_config_builder = inference_config_builder.max_tokens(mt as i32); }
    if let Some(temp) = llm_config.temperature { inference_config_builder = inference_config_builder.temperature(temp); }
    if let Some(ss) = &llm_config.stop_sequences { if !ss.is_empty() { inference_config_builder = inference_config_builder.set_stop_sequences(Some(ss.clone())); }}
    let inference_config = inference_config_builder.build();

    let tool_config = golem_tools_to_bedrock_tool_config(&llm_config.tools)?;

    // Construct the final ConverseRequest manually for now to ensure correct structure
    // as send_with_body in the SDK takes a Blob directly.
    let mut request_map = serde_json::Map::new();
    request_map.insert("modelId".to_string(), json!(llm_config.model)); // modelId is outside the body for the direct client call, but needed if constructing JSON for a generic endpoint.
    request_map.insert("messages".to_string(), serde_json::to_value(&bedrock_messages).map_err(|e| LlmError{code: LlmErrorCode::InternalError, message: format!("Failed to serialize Bedrock messages: {}", e), provider_error_json: None})?);
    if !system_prompts.is_empty() {
         request_map.insert("system".to_string(), serde_json::to_value(&system_prompts).map_err(|e| LlmError{code: LlmErrorCode::InternalError, message: format!("Failed to serialize system prompts: {}", e), provider_error_json: None})?);
    }
    request_map.insert("inferenceConfiguration".to_string(), serde_json::to_value(&inference_config).map_err(|e| LlmError{code: LlmErrorCode::InternalError, message: format!("Failed to serialize inference config: {}", e), provider_error_json: None})?);
    if let Some(tc) = tool_config {
        request_map.insert("toolConfiguration".to_string(), serde_json::to_value(&tc).map_err(|e| LlmError{code: LlmErrorCode::InternalError, message: format!("Failed to serialize tool config: {}", e), provider_error_json: None})?);
    }
    
    // Additional Bedrock specific parameters from llm_config.provider_options if any
    let mut additional_model_request_fields = HashMap::new();
    for kv in &llm_config.provider_options {
        // Bedrock Converse API has specific top-level fields for inference parameters.
        // For other provider_options, they might go into a generic `additionalModelRequestFields`
        // if the model supports it (e.g., for Anthropic models).
        // This needs to be model-specific.
        // For now, assuming they might be Anthropic specific additional fields.
        additional_model_request_fields.insert(kv.key.clone(), json!(kv.value)); 
    }
    if !additional_model_request_fields.is_empty() {
        // Anthropic model example: request_map.insert("additional_model_request_fields".to_string(), json!(additional_config));
        // General Bedrock: request_map.insert("additionalModelRequestFields".to_string(), json!(additional_config));
        // We will assume these are top-level for now and let user specify full key if needed.
        request_map.insert("additionalModelRequestFields".to_string(), json!(additional_model_request_fields));
    }

    let json_body_val = serde_json::Value::Object(request_map);
    Ok(Blob::new(json_body_val.to_string()))
}

fn bedrock_content_to_golem_parts(bedrock_blocks: Option<&[BedrockContentBlock]>) -> Vec<GolemContentPart> {
    match bedrock_blocks {
        Some(blocks) => blocks.iter().filter_map(|block| {
            match block {
                BedrockContentBlock::Text(text) => Some(GolemContentPart::Text(text.clone())),
                BedrockContentBlock::Image(_image_block) => {
                    log::warn!("Image block in Bedrock response not yet convertible to GolemContentPart.");
                    None
                }
                BedrockContentBlock::ToolUse(_) | BedrockContentBlock::ToolResult(_) => None, // Handled separately
                BedrockContentBlock::Json(json_val) => Some(GolemContentPart::Text(serde_json::to_string_pretty(json_val).unwrap_or_else(|_| "<json data>".to_string()))),
                _ => None,
            }
        }).collect(),
        None => Vec::new(),
    }
}

fn bedrock_tool_use_to_golem_tool_call(tool_use: &ToolUseBlock) -> GolemToolCall {
    GolemToolCall {
        id: tool_use.tool_use_id().unwrap_or("").to_string(),
        name: tool_use.name().unwrap_or("").to_string(),
        arguments_json: tool_use.input().map_or("{}".to_string(), |doc| serde_json::to_string(doc).unwrap_or_else(|_| "{}".to_string())),
    }
}

fn bedrock_stop_reason_to_golem_finish_reason(bedrock_reason: Option<&BedrockStopReason>) -> Option<FinishReason> {
    bedrock_reason.map(|reason| match reason {
        BedrockStopReason::EndTurn => FinishReason::Stop,
        BedrockStopReason::ToolUse => FinishReason::ToolCalls,
        BedrockStopReason::MaxTokens => FinishReason::Length,
        BedrockStopReason::StopSequence => FinishReason::Stop, 
        BedrockStopReason::ContentFiltered => FinishReason::ContentFilter,
        _ => FinishReason::Other, 
    })
}

fn bedrock_usage_to_golem_usage(bedrock_usage: Option<&BedrockTokenUsage>) -> Option<GolemUsage> {
    bedrock_usage.map(|u| GolemUsage {
        input_tokens: Some(u.input_tokens() as u32), // Assuming i32 can be cast to u32 safely here
        output_tokens: Some(u.output_tokens() as u32),
        total_tokens: Some(u.total_tokens() as u32),
    })
}

pub(crate) fn bedrock_converse_output_to_golem_chat_event(
    output: ConverseOutput
) -> Result<ChatEvent, LlmError> {
    let bedrock_message_output = output.output.as_ref().and_then(|o| o.as_message());
    let golem_content = bedrock_message_output.map_or(Vec::new(), |m| bedrock_content_to_golem_parts(m.content()));

    let golem_metadata = ResponseMetadata {
        finish_reason: bedrock_stop_reason_to_golem_finish_reason(output.stop_reason().as_ref()),
        usage: bedrock_usage_to_golem_usage(output.usage().as_ref()),
        provider_id: output.model_id().map(String::from),
        timestamp: Some(chrono::Utc::now().to_rfc3339()),
        provider_metadata_json: output.additional_model_response_fields().and_then(|doc| serde_json::to_string(doc).ok()),
    };

    if output.stop_reason() == Some(&BedrockStopReason::ToolUse) {
        let mut golem_tool_calls = Vec::new();
        if let Some(msg_output) = bedrock_message_output {
            for content_block in msg_output.content() {
                if let BedrockContentBlock::ToolUse(tool_use_block) = content_block {
                    golem_tool_calls.push(bedrock_tool_use_to_golem_tool_call(tool_use_block));
                }
            }
        }
        if golem_tool_calls.is_empty() {
            log::warn!("Bedrock reported ToolUse stop reason but no ToolUse blocks found in output message.");
            Ok(ChatEvent::Message(golem_llm::golem::llm::llm::CompleteResponse {
                id: output.id().unwrap_or_default().to_string(),
                content: golem_content,
                tool_calls: vec![], 
                metadata: golem_metadata,
            }))
        } else {
            Ok(ChatEvent::ToolRequest(golem_tool_calls))
        }
    } else {
        Ok(ChatEvent::Message(golem_llm::golem::llm::llm::CompleteResponse {
            id: output.id().unwrap_or_default().to_string(),
            content: golem_content,
            tool_calls: vec![],
            metadata: golem_metadata,
        }))
    }
}

pub(crate) fn bedrock_converse_stream_output_to_golem_stream_event(
    stream_output: ConverseStreamOutput
) -> Result<Option<StreamEvent>, LlmError> {
    match stream_output {
        ConverseStreamOutput::MessageStart(event) => {
            log::trace!("Bedrock Stream: MessageStartEvent - Role: {:?}", event.role);
            Ok(None)
        }
        ConverseStreamOutput::ContentBlockStart(event) => {
            log::trace!("Bedrock Stream: ContentBlockStartEvent - Index: {}", event.content_block_index);
            if let Some(BedrockContentBlockStart::ToolUse(tool_use_block)) = event.start {
                 Ok(Some(StreamEvent::Delta(StreamDelta {
                    content: None,
                    tool_calls: Some(vec![bedrock_tool_use_to_golem_tool_call(&tool_use_block)]),
                })))
            } else {
                Ok(None) 
            }
        }
        ConverseStreamOutput::ContentBlockDelta(event) => {
            log::trace!("Bedrock Stream: ContentBlockDeltaEvent - Index: {}", event.content_block_index);
            match event.delta {
                Some(BedrockContentBlockDelta::Text(text)) => {
                    Ok(Some(StreamEvent::Delta(StreamDelta {
                        content: Some(vec![GolemContentPart::Text(text)]),
                        tool_calls: None,
                    })))
                }
                Some(BedrockContentBlockDelta::ToolUse(tool_use_delta)) => {
                    log::warn!("Bedrock Stream: ContentBlockDelta::ToolUse - argument delta: {}. Full tool call aggregation might be needed.",
                        tool_use_delta.input().unwrap_or_default());
                    Ok(None) 
                }
                _ => Ok(None),
            }
        }
        ConverseStreamOutput::ContentBlockStop(_event) => {
            log::trace!("Bedrock Stream: ContentBlockStopEvent - Index: {}", _event.content_block_index);
            Ok(None)
        }
        ConverseStreamOutput::MessageStop(event) => {
            log::trace!("Bedrock Stream: MessageStopEvent - Stop Reason: {:?}", event.stop_reason);
            let golem_finish_reason = bedrock_stop_reason_to_golem_finish_reason(event.stop_reason().as_ref());
            let provider_metadata_json = event.additional_model_response_fields().and_then(|doc| serde_json::to_string(doc).ok());

            Ok(Some(StreamEvent::Finish(ResponseMetadata {
                finish_reason: golem_finish_reason,
                usage: None, // Usage will be populated by BedrockChatStream from stored MetadataEvent
                provider_id: None, 
                timestamp: Some(chrono::Utc::now().to_rfc3339()),
                provider_metadata_json,
            })))
        }
        ConverseStreamOutput::Metadata(event) => {
            log::trace!("Bedrock Stream: MetadataEvent - Usage: {:?}", event.usage());
            // Usage from this event is handled statefully in BedrockChatStream
            // This function itself does not produce a direct Golem event from Metadata.
            Ok(None) 
        }
        ConverseStreamOutput::InternalServerError(e) => Err(LlmError { 
            code: LlmErrorCode::InternalError, 
            message: format!("Bedrock stream internal server error: {}", e.message().unwrap_or_default()),
            provider_error_json: None 
        }),
        ConverseStreamOutput::ModelStreamErrorException(e) => Err(LlmError { 
            code: LlmErrorCode::InternalError, 
            message: format!("Bedrock model stream error: {}", e.message().unwrap_or_default()),
            provider_error_json: None 
        }),
        ConverseStreamOutput::ThrottlingException(e) => Err(LlmError { 
            code: LlmErrorCode::RateLimitExceeded, 
            message: format!("Bedrock stream throttling: {}", e.message().unwrap_or_default()),
            provider_error_json: None 
        }),
        ConverseStreamOutput::ValidationException(e) => Err(LlmError { 
            code: LlmErrorCode::InvalidRequest, 
            message: format!("Bedrock stream validation error: {}", e.message().unwrap_or_default()),
            provider_error_json: None 
        }),
        _ => {
            log::warn!("Bedrock Stream: Unhandled event type: {:?}", stream_output);
            Ok(None)
        }
    }
}

pub(crate) fn sdk_error_to_llm_error<E: std::error::Error + Send + Sync + 'static>(
    sdk_error: aws_smithy_runtime_api::client::result::SdkError<E, aws_smithy_runtime_api::client::orchestrator::HttpResponse>,
    operation_name: &str,
) -> LlmError {
    let service_error_str = format!("{:?}", sdk_error.as_service_error());
    let error_message = format!("Bedrock operation '{}' failed: {}", operation_name, sdk_error.to_string());
    log::error!("{}", error_message);
    
    let http_response_status = sdk_error.raw().http().status();
    let error_code = match http_response_status.as_u16() {
        400 => LlmErrorCode::InvalidRequest,
        401 | 403 => LlmErrorCode::AuthenticationFailed,
        429 => LlmErrorCode::RateLimitExceeded,
        500..=599 => LlmErrorCode::InternalError,
        _ => LlmErrorCode::Unknown,
    };

    LlmError {
        code: error_code,
        message: error_message,
        provider_error_json: Some(service_error_str),
    }
} 