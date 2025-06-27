use crate::client::{
    Content, ImageSource as ClientImageSource, MediaType, MessagesRequest, MessagesRequestMetadata,
    MessagesResponse, StopReason, Tool, ToolChoice,
};
use base64::{engine::general_purpose, Engine as _};
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, FinishReason,
    ImageReference, ImageSource, ImageUrl, Message, ResponseMetadata, Role, ToolCall,
    ToolDefinition, ToolResult, Usage,
};
use std::collections::HashMap;

pub fn messages_to_request(
    messages: Vec<Message>,
    config: Config,
) -> Result<MessagesRequest, Error> {
    let options = config
        .provider_options
        .into_iter()
        .map(|kv| (kv.key, kv.value))
        .collect::<HashMap<_, _>>();

    let mut anthropic_messages = Vec::new();
    for message in &messages {
        if message.role != Role::System {
            anthropic_messages.push(crate::client::Message {
                role: match &message.role {
                    Role::User => crate::client::Role::User,
                    Role::Assistant => crate::client::Role::Assistant,
                    Role::Tool => crate::client::Role::User,
                    Role::System => unreachable!(),
                },
                content: message_to_content(message),
            })
        }
    }

    let mut system_messages = Vec::new();
    for message in &messages {
        if message.role == Role::System {
            system_messages.extend(message_to_content(message))
        }
    }

    let tool_choice = config.tool_choice.map(convert_tool_choice);
    let tools = if config.tools.is_empty() {
        None
    } else {
        let mut tools = Vec::new();
        for tool in &config.tools {
            tools.push(tool_definition_to_tool(tool)?)
        }
        Some(tools)
    };

    Ok(MessagesRequest {
        max_tokens: config.max_tokens.unwrap_or(4096),
        messages: anthropic_messages,
        model: config.model,
        metadata: options
            .get("user_id")
            .map(|user_id| MessagesRequestMetadata {
                user_id: Some(user_id.to_string()),
            }),
        stop_sequences: config.stop_sequences,
        stream: false,
        system: system_messages,
        temperature: config.temperature,
        tool_choice,
        tools,
        top_k: options
            .get("top_k")
            .and_then(|top_k_s| top_k_s.parse::<u32>().ok()),
        top_p: options
            .get("top_p")
            .and_then(|top_p_s| top_p_s.parse::<f32>().ok()),
    })
}

fn convert_tool_choice(tool_name: String) -> ToolChoice {
    if &tool_name == "auto" {
        ToolChoice::Auto {
            disable_parallel_tool_use: None,
        }
    } else if &tool_name == "none" {
        ToolChoice::None {}
    } else if &tool_name == "any" {
        ToolChoice::Any {
            disable_parallel_tool_use: None,
        }
    } else {
        ToolChoice::Tool {
            name: tool_name,
            disable_parallel_tool_use: None,
        }
    }
}

pub fn process_response(response: MessagesResponse) -> ChatEvent {
    let mut contents = Vec::new();
    let mut tool_calls = Vec::new();

    for content in response.content {
        match content {
            Content::Text { text, .. } => contents.push(ContentPart::Text(text)),
            Content::Image { source, .. } => match source {
                ClientImageSource::Url { url } => {
                    contents.push(ContentPart::Image(ImageReference::Url(ImageUrl {
                        url,
                        detail: None,
                    })))
                }
                ClientImageSource::Base64 { data, media_type } => {
                    match general_purpose::STANDARD.decode(data) {
                        Ok(decoded_data) => {
                            let mime_type_str = match media_type {
                                crate::client::MediaType::Jpeg => "image/jpeg".to_string(),
                                crate::client::MediaType::Png => "image/png".to_string(),
                                crate::client::MediaType::Gif => "image/gif".to_string(),
                                crate::client::MediaType::Webp => "image/webp".to_string(),
                            };
                            contents.push(ContentPart::Image(ImageReference::Inline(
                                ImageSource {
                                    data: decoded_data,
                                    mime_type: mime_type_str,
                                    detail: None,
                                },
                            )));
                        }
                        Err(e) => {
                            return ChatEvent::Error(Error {
                                code: ErrorCode::InvalidRequest,
                                message: format!("Failed to decode base64 image data: {e}"),
                                provider_error_json: None,
                            });
                        }
                    }
                }
            },
            Content::ToolUse {
                id, input, name, ..
            } => tool_calls.push(ToolCall {
                id,
                name,
                arguments_json: serde_json::to_string(&input).unwrap(),
            }),
            Content::ToolResult { .. } => {}
        }
    }

    if contents.is_empty() {
        ChatEvent::ToolRequest(tool_calls)
    } else {
        let metadata = ResponseMetadata {
            finish_reason: response.stop_reason.map(stop_reason_to_finish_reason),
            usage: Some(convert_usage(response.usage)),
            provider_id: None,
            timestamp: None,
            provider_metadata_json: None,
        };

        ChatEvent::Message(CompleteResponse {
            id: response.id,
            content: contents,
            tool_calls,
            metadata,
        })
    }
}

pub fn tool_results_to_messages(
    tool_results: Vec<(ToolCall, ToolResult)>,
) -> Vec<crate::client::Message> {
    let mut messages = Vec::new();

    for (tool_call, tool_result) in tool_results {
        messages.push(crate::client::Message {
            content: vec![Content::ToolUse {
                id: tool_call.id.clone(),
                input: serde_json::from_str(&tool_call.arguments_json).unwrap(),
                name: tool_call.name,
                cache_control: None,
            }],
            role: crate::client::Role::Assistant,
        });
        let content = match tool_result {
            ToolResult::Success(success) => Content::ToolResult {
                tool_use_id: tool_call.id,
                cache_control: None,
                content: vec![Content::Text {
                    text: success.result_json,
                    cache_control: None,
                }],
                is_error: false,
            },
            ToolResult::Error(error) => Content::ToolResult {
                tool_use_id: tool_call.id,
                cache_control: None,
                content: vec![Content::Text {
                    text: error.error_message,
                    cache_control: None,
                }],
                is_error: true,
            },
        };
        messages.push(crate::client::Message {
            content: vec![content],
            role: crate::client::Role::User,
        });
    }

    messages
}

pub fn stop_reason_to_finish_reason(stop_reason: StopReason) -> FinishReason {
    match stop_reason {
        StopReason::EndTurn => FinishReason::Other,
        StopReason::MaxTokens => FinishReason::Length,
        StopReason::StopSequence => FinishReason::Stop,
        StopReason::ToolUse => FinishReason::ToolCalls,
    }
}

pub fn convert_usage(usage: crate::client::Usage) -> Usage {
    Usage {
        input_tokens: Some(usage.input_tokens),
        output_tokens: Some(usage.output_tokens),
        total_tokens: None,
    }
}

fn message_to_content(message: &Message) -> Vec<Content> {
    let mut result = Vec::new();

    for content_part in &message.content {
        match content_part {
            ContentPart::Text(text) => result.push(Content::Text {
                text: text.clone(),
                cache_control: None,
            }),
            ContentPart::Image(image_reference) => match image_reference {
                ImageReference::Url(image_url) => result.push(Content::Image {
                    source: ClientImageSource::Url {
                        url: image_url.url.clone(),
                    },
                    cache_control: None,
                }),
                ImageReference::Inline(image_source) => {
                    let base64_data = general_purpose::STANDARD.encode(&image_source.data);
                    let media_type = match image_source.mime_type.as_str() {
                        "image/jpeg" => MediaType::Jpeg,
                        "image/png" => MediaType::Png,
                        "image/gif" => MediaType::Gif,
                        "image/webp" => MediaType::Webp,
                        _ => MediaType::Jpeg,
                    };

                    result.push(Content::Image {
                        source: ClientImageSource::Base64 {
                            data: base64_data,
                            media_type,
                        },
                        cache_control: None,
                    });
                }
            },
        }
    }

    result
}

fn tool_definition_to_tool(tool: &ToolDefinition) -> Result<Tool, Error> {
    match serde_json::from_str(&tool.parameters_schema) {
        Ok(value) => Ok(Tool::CustomTool {
            input_schema: value,
            name: tool.name.clone(),
            cache_control: None,
            description: tool.description.clone(),
        }),
        Err(error) => Err(Error {
            code: ErrorCode::InternalError,
            message: format!("Failed to parse tool parameters for {}: {error}", tool.name),
            provider_error_json: None,
        }),
    }
}
