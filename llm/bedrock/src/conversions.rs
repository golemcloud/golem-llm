use crate::client::{
    ContentBlock, ConverseRequest, ConverseResponse, ImageFormat, ImageSource as ClientImageSource,
    InferenceConfig, Message as ClientMessage, Role as ClientRole, StopReason, SystemContentBlock,
    Tool, ToolChoice, ToolConfig, ToolSpec, ToolResultContentBlock, ToolResultStatus,
    ImageBlock, ToolUseBlock, ToolResultBlock, ToolInputSchema, ToolChoiceTool,
};
use base64::{engine::general_purpose, Engine as _};
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, FinishReason,
    ImageReference, ImageSource, Message, ResponseMetadata, Role, ToolCall,
    ToolDefinition, ToolResult, Usage,
};
use reqwest::{Client, Url};
use std::{collections::HashMap, fs, path::Path};

pub fn messages_to_request(
    messages: Vec<Message>,
    config: Config,
) -> Result<ConverseRequest, Error> {
    let options = config
        .provider_options
        .into_iter()
        .map(|kv| (kv.key, kv.value))
        .collect::<HashMap<_, _>>();

    let mut bedrock_messages = Vec::new();
    let mut system_messages = Vec::new();

    for message in &messages {
        match message.role {
            Role::System => {
                system_messages.extend(message_to_system_content(message));
            }
            Role::User | Role::Assistant | Role::Tool => {
                bedrock_messages.push(ClientMessage {
                    role: match &message.role {
                        Role::User | Role::Tool => ClientRole::User,
                        Role::Assistant => ClientRole::Assistant,
                        Role::System => unreachable!(),
                    },
                    content: message_to_content(message)?,
                });
            }
        }
    }

    let inference_config = if config.max_tokens.is_some()
        || config.temperature.is_some()
        || config.stop_sequences.is_some()
        || options.contains_key("top_p")
    {
        Some(InferenceConfig {
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: options
                .get("top_p")
                .and_then(|top_p_s| top_p_s.parse::<f32>().ok()),
            stop_sequences: config.stop_sequences,
        })
    } else {
        None
    };

    let tool_config = if config.tools.is_empty() {
        None
    } else {
        let mut tools = Vec::new();
        for tool in &config.tools {
            tools.push(tool_definition_to_tool(tool)?);
        }
        
        let tool_choice = config.tool_choice.map(convert_tool_choice);
        
        if tools.is_empty() {
            None
        } else {
            Some(ToolConfig {
                tools,
                tool_choice,
            })
        }
    };

    Ok(ConverseRequest {
        model_id: config.model.clone(),
        messages: bedrock_messages,
        system: if system_messages.is_empty() {
            None
        } else {
            Some(system_messages)
        },
        inference_config,
        tool_config,
        guardrail_config: None,
        additional_model_request_fields: None,
    })
}

fn convert_tool_choice(tool_name: String) -> ToolChoice {
    use serde_json::Value;
    
    match tool_name.as_str() {
        "auto" => ToolChoice::Auto {
            auto: Value::Object(serde_json::Map::new()),
        },
        "any" => ToolChoice::Any {
            any: Value::Object(serde_json::Map::new()),
        },
        name => ToolChoice::Tool {
            tool: ToolChoiceTool {
                name: name.to_string(),
            },
        },
    }
}

pub fn process_response(response: ConverseResponse) -> ChatEvent {
    let mut contents = Vec::new();
    let mut tool_calls = Vec::new();

    for content in response.output.message.content {
        match content {
            ContentBlock::Text { text } => contents.push(ContentPart::Text(text)),
            ContentBlock::Image { image } => {
                match general_purpose::STANDARD.decode(&image.source.bytes) {
                    Ok(decoded_data) => {
                        let mime_type = match image.format {
                            ImageFormat::Jpeg => "image/jpeg",
                            ImageFormat::Png => "image/png",
                            ImageFormat::Gif => "image/gif",
                            ImageFormat::Webp => "image/webp",
                        };
                        contents.push(ContentPart::Image(ImageReference::Inline(
                            ImageSource {
                                data: decoded_data,
                                mime_type: mime_type.to_string(),
                                detail: None,
                            },
                        )));
                    }
                    Err(e) => {
                        return ChatEvent::Error(Error {
                            code: ErrorCode::InvalidRequest,
                            message: format!("Failed to decode base64 image data: {}", e),
                            provider_error_json: None,
                        });
                    }
                }
            }
            ContentBlock::ToolUse { tool_use } => tool_calls.push(ToolCall {
                id: tool_use.tool_use_id,
                name: tool_use.name,
                arguments_json: serde_json::to_string(&tool_use.input).unwrap(),
            }),
            ContentBlock::ToolResult { .. } => {}
        }
    }

    if contents.is_empty() && !tool_calls.is_empty() {
        ChatEvent::ToolRequest(tool_calls)
    } else {
        let request_id = "bedrock-response".to_string();
        
        let metadata = ResponseMetadata {
            finish_reason: Some(stop_reason_to_finish_reason(response.stop_reason)),
            usage: Some(convert_usage(response.usage)),
            provider_id: Some(request_id.clone()),
            timestamp: None,
            provider_metadata_json: None,
        };

        ChatEvent::Message(CompleteResponse {
            id: request_id,
            content: contents,
            tool_calls,
            metadata,
        })
    }
}

pub fn tool_results_to_messages(
    tool_results: Vec<(ToolCall, ToolResult)>,
) -> Vec<ClientMessage> {
    let mut messages = Vec::new();

    for (tool_call, tool_result) in tool_results {
        messages.push(ClientMessage {
            content: vec![ContentBlock::ToolUse {
                tool_use: ToolUseBlock {
                    tool_use_id: tool_call.id.clone(),
                    name: tool_call.name,
                    input: serde_json::from_str(&tool_call.arguments_json).unwrap(),
                },
            }],
            role: ClientRole::Assistant,
        });

        let (content, status) = match tool_result {
            ToolResult::Success(success) => (
                vec![ToolResultContentBlock::Text {
                    text: success.result_json,
                }],
                Some(ToolResultStatus::Success),
            ),
            ToolResult::Error(error) => (
                vec![ToolResultContentBlock::Text {
                    text: error.error_message,
                }],
                Some(ToolResultStatus::Error),
            ),
        };

        messages.push(ClientMessage {
            content: vec![ContentBlock::ToolResult {
                tool_result: ToolResultBlock {
                    tool_use_id: tool_call.id,
                    content,
                    status,
                },
            }],
            role: ClientRole::User,
        });
    }

    messages
}

pub fn stop_reason_to_finish_reason(stop_reason: StopReason) -> FinishReason {
    match stop_reason {
        StopReason::EndTurn => FinishReason::Stop,
        StopReason::ToolUse => FinishReason::ToolCalls,
        StopReason::MaxTokens => FinishReason::Length,
        StopReason::StopSequence => FinishReason::Stop,
        StopReason::GuardrailIntervened => FinishReason::ContentFilter,
        StopReason::ContentFiltered => FinishReason::ContentFilter,
    }
}

pub fn convert_usage(usage: crate::client::Usage) -> Usage {
    Usage {
        input_tokens: Some(usage.input_tokens),
        output_tokens: Some(usage.output_tokens),
        total_tokens: Some(usage.total_tokens),
    }
}

fn message_to_content(message: &Message) -> Result<Vec<ContentBlock>, Error> {
    let mut result = Vec::new();

    for content_part in &message.content {
        match content_part {
            ContentPart::Text(text) => result.push(ContentBlock::Text {
                text: text.clone(),
            }),
            ContentPart::Image(image_reference) => match image_reference {
                ImageReference::Url(image_url) => {
                    let url = &image_url.url;
                    let mut format = ImageFormat::Png;
                    let bytes = if Url::parse(url).is_ok() {
                        let client = Client::new();
                        let response = client.get(url).send().map_err(|e| Error {
                            code: ErrorCode::InvalidRequest,
                            message: format!("Failed to fetch image from URL: {}", e),
                            provider_error_json: None,
                        });
                        response.map(|r| {
                            format = match r.headers().get("Content-Type").unwrap().to_str().unwrap() {
                                "image/jpeg" => ImageFormat::Jpeg,
                                "image/png" => ImageFormat::Png,
                                "image/gif" => ImageFormat::Gif,
                                "image/webp" => ImageFormat::Webp,
                                _ => ImageFormat::Jpeg,
                            };
                            r.bytes().unwrap().to_vec()
                        })
                    } else {
                        let path = Path::new(url);
                        fs::read(path).map_err(|e| Error {
                            code: ErrorCode::InvalidRequest,
                            message: format!("Failed to read image from path: {}", e),
                            provider_error_json: None,
                        })
                    };
                
                    let base64_data = general_purpose::STANDARD.encode(&bytes.unwrap());
                    result.push(ContentBlock::Image {
                        image: ImageBlock {
                            format: ImageFormat::Png,
                            source: ClientImageSource {
                                bytes: base64_data,
                            },
                        },
                    });
                },
                ImageReference::Inline(image_source) => {
                    let base64_data = general_purpose::STANDARD.encode(&image_source.data);
                    let format = match image_source.mime_type.as_str() {
                        "image/jpeg" => ImageFormat::Jpeg,
                        "image/png" => ImageFormat::Png,
                        "image/gif" => ImageFormat::Gif,
                        "image/webp" => ImageFormat::Webp,
                        _ => ImageFormat::Jpeg,
                    };

                    result.push(ContentBlock::Image {
                        image: ImageBlock {
                            format,
                            source: ClientImageSource {
                                bytes: base64_data,
                            },
                        },
                    });
                }
            },
        }
    }

    Ok(result)
}

fn message_to_system_content(message: &Message) -> Vec<SystemContentBlock> {
    let mut result = Vec::new();

    for content_part in &message.content {
        match content_part {
            ContentPart::Text(text) => result.push(SystemContentBlock::Text {
                text: text.clone(),
            }),
            ContentPart::Image(_) => {}
        }
    }

    result
}

fn tool_definition_to_tool(tool: &ToolDefinition) -> Result<Tool, Error> {
    use serde_json::Value;
    
    let schema_value = if tool.parameters_schema.trim().is_empty() {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        })
    } else {
        match serde_json::from_str::<Value>(&tool.parameters_schema) {
            Ok(value) => value,
            Err(error) => {
                return Err(Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to parse tool parameters for {}: {error}", tool.name),
                    provider_error_json: None,
                });
            }
        }
    };
    
    Ok(Tool {
        tool_spec: ToolSpec {
            name: tool.name.clone(),
            description: tool.description.clone().unwrap_or_default(),
            input_schema: ToolInputSchema {
                json: schema_value,
            },
        },
    })
}