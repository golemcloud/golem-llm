use aws_sdk_bedrockruntime::types::{ContentBlock, ConversationRole, Message as BedrockMessage, ImageBlock, ImageSource, ImageFormat};
use aws_sdk_bedrockruntime::primitives::Blob;
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, ContentPart, Error, ErrorCode, FinishReason, Message,
    ResponseMetadata, Role, ToolCall, ToolResult, Usage,
};
use log::{trace, warn};
use base64::{Engine as _, engine::general_purpose};

/// Convert golem-llm messages to Bedrock Converse API format
pub fn messages_to_bedrock_converse(
    messages: &[Message],
) -> Result<Vec<BedrockMessage>, Error> {
    let mut bedrock_messages = Vec::new();

    for message in messages {
        let role = match message.role {
            Role::User => ConversationRole::User,
            Role::Assistant => ConversationRole::Assistant,
            _ => {
                return Err(Error {
                    code: ErrorCode::InvalidRequest,
                    message: format!("Unsupported message role: {:?}", message.role),
                    provider_error_json: None,
                });
            }
        };

        let mut content_blocks = Vec::new();

        for content in &message.content {
            match content {
                ContentPart::Text(text) => {
                    content_blocks.push(ContentBlock::Text(text.clone()));
                }
                ContentPart::Image(image) => {
                    // Convert image to Bedrock ImageBlock
                    match convert_image_to_bedrock_block(&image.url) {
                        Ok(image_block) => {
                            content_blocks.push(ContentBlock::Image(image_block));
                        }
                        Err(err) => {
                            warn!("Failed to convert image {}: {:?}. Using fallback.", image.url, err);
                            // Fallback: Add a text description instead of failing
                            content_blocks.push(ContentBlock::Text(format!(
                                "[Image: {}]",
                                image.url
                            )));
                        }
                    }
                }
            }
        }

        if content_blocks.is_empty() {
            content_blocks.push(ContentBlock::Text("".to_string()));
        }

        let bedrock_message = BedrockMessage::builder()
            .role(role)
            .set_content(Some(content_blocks))
            .build()
            .map_err(|e| Error {
                code: ErrorCode::InternalError,
                message: format!("Failed to build Bedrock message: {e}"),
                provider_error_json: None,
            })?;

        bedrock_messages.push(bedrock_message);
    }

    Ok(bedrock_messages)
}

/// Convert an image URL to a Bedrock ImageBlock
/// Currently supports data URLs, with HTTP URLs planned for future implementation
fn convert_image_to_bedrock_block(url: &str) -> Result<ImageBlock, Error> {
    // Handle base64 data URLs (supported)
    if url.starts_with("data:") {
        return convert_data_url_to_image_block(url);
    }
    
    // For HTTP URLs, return an informative error for now
    // TODO: Implement async HTTP fetching in a future version
    Err(Error {
        code: ErrorCode::Unsupported,
        message: format!(
            "HTTP image URLs not yet supported. Use data URLs like 'data:image/png;base64,...' or configure images via S3. URL: {}", 
            url
        ),
        provider_error_json: None,
    })
}

/// Convert a data URL to ImageBlock
fn convert_data_url_to_image_block(data_url: &str) -> Result<ImageBlock, Error> {
    // Parse data URL format: data:image/png;base64,<data>
    if !data_url.starts_with("data:image/") {
        return Err(Error {
            code: ErrorCode::InvalidRequest,
            message: "Only image data URLs are supported".to_string(),
            provider_error_json: None,
        });
    }

    let parts: Vec<&str> = data_url.splitn(2, ',').collect();
    if parts.len() != 2 {
        return Err(Error {
            code: ErrorCode::InvalidRequest,
            message: "Invalid data URL format".to_string(),
            provider_error_json: None,
        });
    }

    let header = parts[0];
    let data = parts[1];

    // Extract format from header: data:image/png;base64
    let format = if header.contains("image/png") {
        "png"
    } else if header.contains("image/jpeg") || header.contains("image/jpg") {
        "jpeg"
    } else if header.contains("image/gif") {
        "gif"
    } else if header.contains("image/webp") {
        "webp"
    } else {
        return Err(Error {
            code: ErrorCode::InvalidRequest,
            message: "Unsupported image format in data URL".to_string(),
            provider_error_json: None,
        });
    };

    // Decode base64 data
    let image_bytes = general_purpose::STANDARD.decode(data).map_err(|e| Error {
        code: ErrorCode::InvalidRequest,
        message: format!("Failed to decode base64 image data: {}", e),
        provider_error_json: None,
    })?;

    // Create AWS Blob from bytes
    let blob = Blob::new(image_bytes);
    let image_source = ImageSource::Bytes(blob);

    // Build ImageBlock
    let image_block = ImageBlock::builder()
        .format(ImageFormat::from(format))
        .source(image_source)
        .build()
        .map_err(|e| Error {
            code: ErrorCode::InternalError,
            message: format!("Failed to build ImageBlock: {}", e),
            provider_error_json: None,
        })?;

    Ok(image_block)
}

/// Convert Bedrock Converse response to ChatEvent
pub fn bedrock_converse_to_chat_event(
    output: aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
    _model_id: &str,
) -> Result<ChatEvent, Error> {
    let message = output
        .output()
        .ok_or_else(|| Error {
            code: ErrorCode::InternalError,
            message: "No output in Bedrock response".to_string(),
            provider_error_json: None,
        })?
        .as_message()
        .map_err(|_| Error {
            code: ErrorCode::InternalError,
            message: "Bedrock output is not a message".to_string(),
            provider_error_json: None,
        })?;

    // Extract text and tool calls from content blocks
    let mut response_text = String::new();
    let mut tool_calls = Vec::new();

    for content_block in message.content() {
        match content_block {
            aws_sdk_bedrockruntime::types::ContentBlock::Text(text) => {
                response_text.push_str(&text);
            }
            aws_sdk_bedrockruntime::types::ContentBlock::ToolUse(tool_use) => {
                // Convert Bedrock tool use to golem-llm ToolCall
                let arguments_json = format!("{:?}", tool_use.input());

                let tool_call = ToolCall {
                    id: tool_use.tool_use_id().to_string(),
                    name: tool_use.name().to_string(),
                    arguments_json,
                };
                tool_calls.push(tool_call);
            }
            _ => {
                // Handle other content block types if needed
                trace!("Unhandled content block type in Bedrock response");
            }
        }
    }

    // Extract usage information
    let usage = output.usage().map(|bedrock_usage| Usage {
        input_tokens: Some(bedrock_usage.input_tokens() as u32),
        output_tokens: Some(bedrock_usage.output_tokens() as u32),
        total_tokens: Some((bedrock_usage.input_tokens() + bedrock_usage.output_tokens()) as u32),
    });

    // Determine finish reason
    let finish_reason = match output.stop_reason() {
        aws_sdk_bedrockruntime::types::StopReason::EndTurn => Some(FinishReason::Stop),
        aws_sdk_bedrockruntime::types::StopReason::ToolUse => Some(FinishReason::ToolCalls),
        aws_sdk_bedrockruntime::types::StopReason::MaxTokens => Some(FinishReason::Length),
        aws_sdk_bedrockruntime::types::StopReason::StopSequence => Some(FinishReason::Stop),
        aws_sdk_bedrockruntime::types::StopReason::ContentFiltered => {
            Some(FinishReason::ContentFilter)
        }
        _ => Some(FinishReason::Stop), // Default to Stop for unknown reasons
    };

    let metadata = ResponseMetadata {
        provider_id: None,
        usage,
        finish_reason,
        timestamp: None,
        provider_metadata_json: None,
    };

    Ok(ChatEvent::Message(CompleteResponse {
        id: String::new(), // Bedrock doesn't provide message IDs in basic converse
        content: vec![ContentPart::Text(response_text)],
        tool_calls,
        metadata,
    }))
}

/// Extract model configuration from Config
pub fn extract_model_config(
    config: &golem_llm::golem::llm::llm::Config,
) -> (Option<i32>, Option<f32>, Option<String>) {
    let max_tokens = config
        .provider_options
        .iter()
        .find(|kv| {
            kv.key.eq_ignore_ascii_case("max_tokens") || kv.key.eq_ignore_ascii_case("maxTokens")
        })
        .and_then(|kv| kv.value.parse().ok());

    let temperature = config
        .provider_options
        .iter()
        .find(|kv| kv.key.eq_ignore_ascii_case("temperature"))
        .and_then(|kv| kv.value.parse().ok());

    let system_prompt = config
        .provider_options
        .iter()
        .find(|kv| {
            kv.key.eq_ignore_ascii_case("system") || kv.key.eq_ignore_ascii_case("system_prompt")
        })
        .map(|kv| kv.value.clone());

    (max_tokens, temperature, system_prompt)
}

/// Convert tool results to Bedrock messages for the continue_ flow
pub fn tool_results_to_bedrock_messages(
    tool_results: Vec<(ToolCall, ToolResult)>,
) -> Vec<BedrockMessage> {
    let mut messages = Vec::new();

    for (tool_call, tool_result) in tool_results {
        // Convert tool calls and results to text messages

        // Add assistant message describing the tool call
        let tool_call_text = format!("Tool call: {} with id {}", tool_call.name, tool_call.id);
        let assistant_message = BedrockMessage::builder()
            .role(ConversationRole::Assistant)
            .content(ContentBlock::Text(tool_call_text))
            .build()
            .expect("Failed to build assistant message with tool call");

        messages.push(assistant_message);

        // Add user message with the tool result
        let result_text = match tool_result {
            ToolResult::Success(success) => format!("Tool result: {}", success.result_json),
            ToolResult::Error(error) => format!("Tool error: {}", error.error_message),
        };

        let user_message = BedrockMessage::builder()
            .role(ConversationRole::User)
            .content(ContentBlock::Text(result_text))
            .build()
            .expect("Failed to build user message with tool result");

        messages.push(user_message);
    }

    messages
}
