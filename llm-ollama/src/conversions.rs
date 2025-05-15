use crate::client::{
    ContentPart, MessageContentPayload, OllamaApi, OllamaChatRequest, OllamaChatResponse,
    OllamaFunction, OllamaMessage, OllamaMessageContent, OllamaTool, OllamaToolCall,
    OllamaToolCallFunction, ToolChoice,
};
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart as GolemContentPart, Error, ErrorCode,
    FinishReason, Message, ResponseMetadata, Role, ToolCall, ToolDefinition, ToolResult, Usage,
};
use std::collections::HashMap;

pub fn messages_to_request(
    messages: Vec<Message>,
    config: Config,
    api: &OllamaApi,
) -> Result<OllamaChatRequest, Error> {
    let options = config
        .provider_options
        .iter()
        .map(|kv| (kv.key.clone(), kv.value.clone()))
        .collect::<HashMap<_, _>>();

    let mut ollama_messages = Vec::new();
    for message in messages {
        ollama_messages.push(message_to_ollama_message(message, api)?);
    }

    let tools = if config.tools.is_empty() {
        None
    } else {
        let mut tools = Vec::new();
        for tool in &config.tools {
            tools.push(tool_definition_to_tool(tool)?);
        }
        Some(tools)
    };

    let tool_choice = if let Some(tc) = config.tool_choice.as_ref() {
        if tc == "none" || tc == "auto" {
            Some(ToolChoice::String(tc.clone()))
        } else {
            Some(ToolChoice::Object {
                typ: "function".to_string(),
                function: crate::client::OllamaFunctionChoice { name: tc.clone() },
            })
        }
    } else {
        None
    };

    Ok(OllamaChatRequest {
        model: config.model,
        messages: ollama_messages,
        tools,
        tool_choice,
        response_format: options.get("response_format").cloned(),
        temperature: config.temperature,
        top_p: options.get("top_p").and_then(|v| v.parse().ok()),
        stop: config.stop_sequences,
        frequency_penalty: options
            .get("frequency_penalty")
            .and_then(|v| v.parse().ok()),
        presence_penalty: options.get("presence_penalty").and_then(|v| v.parse().ok()),
        seed: options.get("seed").and_then(|v| v.parse().ok()),
        max_tokens: config.max_tokens,
        keep_alive: options.get("keep_alive").cloned(),
        stream: false,
    })
}

fn message_to_ollama_message(message: Message, api: &OllamaApi) -> Result<OllamaMessage, Error> {
    let role = match message.role {
        Role::User => "user".to_string(),
        Role::Assistant => "assistant".to_string(),
        Role::System => "system".to_string(),
        Role::Tool => "tool".to_string(),
    };

    let mut images = false;
    let mut content = String::new();
    let mut parts = Vec::new();

    for part in message.content {
        match part {
            GolemContentPart::Text(text) => {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&text);
            }
            GolemContentPart::Image(image) => {
                images = true;

                if !content.is_empty() {
                    parts.push(ContentPart::Text {
                        text: std::mem::take(&mut content),
                    });
                }

                let base64 = api.image_url_to_base64(&image.url)?;
                parts.push(ContentPart::ImageUrl {
                    image_url: crate::client::ImageUrl {
                        url: base64,
                        detail: None,
                    },
                });
            }
        }
    }
    if images && !content.is_empty() {
        parts.push(ContentPart::Text {
            text: content.clone(),
        });
    }

    let final_content = if images {
        MessageContentPayload::Array { content: parts }
    } else {
        MessageContentPayload::Text { content }
    };

    Ok(OllamaMessage {
        role: role.clone(),
        content: OllamaMessageContent {
            role,
            content: Some(final_content),
            tool_calls: None,
        },
        tool_calls: None,
    })
}

fn tool_definition_to_tool(tool: &ToolDefinition) -> Result<OllamaTool, Error> {
    let parameters = match serde_json::from_str(&tool.parameters_schema) {
        Ok(params) => params,
        Err(error) => {
            return Err(Error {
                code: ErrorCode::InternalError,
                message: format!("Failed to parse tool parameters for {}: {error}", tool.name),
                provider_error_json: None,
            });
        }
    };

    Ok(OllamaTool {
        typ: "function".to_string(),
        function: OllamaFunction {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters,
        },
    })
}

pub fn process_response(response: OllamaChatResponse) -> ChatEvent {
    let choice = match response.choices.first() {
        Some(c) => c,
        None => {
            return ChatEvent::Error(Error {
                code: ErrorCode::InternalError,
                message: "No choices in Ollama response".to_string(),
                provider_error_json: Some(serde_json::to_string(&response).unwrap_or_default()),
            });
        }
    };

    if let Some(tool_calls) = &choice.message.tool_calls {
        if !tool_calls.is_empty() {
            let tool_calls = tool_calls
                .iter()
                .map(|tc| ToolCall {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments_json: tc.function.arguments.to_string(),
                })
                .collect();

            return ChatEvent::ToolRequest(tool_calls);
        }
    }

    let content = match &choice.message.content {
        Some(MessageContentPayload::Text { content }) => {
            vec![GolemContentPart::Text(content.clone())]
        }
        Some(MessageContentPayload::Array { content }) => {
            let mut parts = Vec::new();
            for part in content {
                match part {
                    ContentPart::Text { text } => {
                        parts.push(GolemContentPart::Text(text.clone()));
                    }
                    ContentPart::ImageUrl { image_url } => {
                        parts.push(GolemContentPart::Text(format!(
                            "[Image: {}]",
                            image_url.url
                        )));
                    }
                }
            }
            parts
        }
        None => vec![],
    };

    let finish_reason = choice.finish_reason.as_deref().map(|reason| match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    });

    let usage = response.usage.as_ref().map(|u| Usage {
        input_tokens: Some(u.prompt_tokens),
        output_tokens: Some(u.completion_tokens),
        total_tokens: Some(u.total_tokens),
    });

    let metadata = ResponseMetadata {
        finish_reason,
        usage,
        provider_id: Some(response.id.clone()),
        timestamp: Some(response.created.to_string()),
        provider_metadata_json: Some(serde_json::to_string(&response.usage).unwrap_or_default()),
    };

    ChatEvent::Message(CompleteResponse {
        id: response.id.clone(),
        content,
        tool_calls: Vec::new(),
        metadata,
    })
}

pub fn tool_results_to_messages(tool_results: Vec<(ToolCall, ToolResult)>) -> Vec<OllamaMessage> {
    let mut messages = Vec::new();

    for (tool_call, tool_result) in tool_results {
        let tool_call_obj = OllamaToolCall {
            id: tool_call.id.clone(),
            function: OllamaToolCallFunction {
                name: tool_call.name.clone(),
                arguments: serde_json::from_str(&tool_call.arguments_json)
                    .unwrap_or(serde_json::Value::Null),
            },
        };

        messages.push(OllamaMessage {
            role: "assistant".to_string(),
            content: OllamaMessageContent {
                role: "assistant".to_string(),

                content: Some(MessageContentPayload::Text {
                    content: "".to_string(),
                }),
                tool_calls: Some(vec![tool_call_obj]),
            },
            tool_calls: None,
        });

        let result_content = match tool_result {
            ToolResult::Success(success) => success.result_json,
            ToolResult::Error(error) => format!("Error: {}", error.error_message),
        };

        messages.push(OllamaMessage {
            role: "tool".to_string(),
            content: OllamaMessageContent {
                role: "tool".to_string(),
                content: Some(MessageContentPayload::Text {
                    content: result_content,
                }),
                tool_calls: None,
            },
            tool_calls: None,
        });
    }

    messages
}
