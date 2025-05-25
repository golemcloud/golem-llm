use crate::client::{
    OllamaApi, OllamaChatRequest, OllamaChatResponse, OllamaFunction, OllamaMessage,
    OllamaMessageContent, OllamaTool, OllamaToolCall, OllamaToolCallFunction, ToolChoice,
};
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart as GolemContentPart, Error, ErrorCode,
    FinishReason, Message, ResponseMetadata, Role, ToolCall, ToolDefinition, ToolResult,
};

pub fn messages_to_request(
    messages: Vec<Message>,
    config: Config,
    api: &OllamaApi,
) -> Result<OllamaChatRequest, Error> {
    let options = config
        .provider_options
        .iter()
        .map(|kv| (kv.key.clone(), serde_json::Value::String(kv.value.clone())))
        .collect::<serde_json::Map<_, _>>();

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
        options: Some(options),
        keep_alive: None,
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

    let mut text = String::new();
    let mut images = Vec::new();

    for part in message.content {
        match part {
            GolemContentPart::Text(t) => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&t);
            }
            GolemContentPart::Image(image) => {
                let base64 = api.image_url_to_base64(&image.url)?;
                images.push(base64);
            }
        }
    }

    let image_field = if images.is_empty() {
        None
    } else {
        Some(images)
    };

    Ok(OllamaMessage {
        role: role.clone(),
        content: OllamaMessageContent {
            role,
            content: text,
            images: image_field,
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
    let message = &response.message;

    if let Some(tool_calls) = &message.tool_calls {
        if !tool_calls.is_empty() {
            let calls = tool_calls
                .iter()
                .map(|tc| ToolCall {
                    id: tc
                        .id
                        .clone()
                        .unwrap_or_else(|| "unknown_tool_call".to_string()),
                    name: tc.function.name.clone(),
                    arguments_json: tc.function.arguments.to_string(),
                })
                .collect();

            return ChatEvent::ToolRequest(calls);
        }
    }

    let mut content_parts = Vec::new();

    if let Some(images) = &message.images {
        for image_base64 in images {
            content_parts.push(GolemContentPart::Text(format!(
                "[Image(base64)]: {}",
                image_base64
            )));
        }
    }

    if !message.content.trim().is_empty() {
        content_parts.push(GolemContentPart::Text(message.content.clone()));
    }

    let finish_reason = if response.done {
        Some(FinishReason::Stop)
    } else {
        None
    };

    let metadata = ResponseMetadata {
        finish_reason,
        usage: None,
        provider_id: None,
        timestamp: Some(response.created_at.to_string()),
        provider_metadata_json: Some(serde_json::to_string(&response).unwrap_or_default()),
    };

    ChatEvent::Message(CompleteResponse {
        id: "ollama".to_string(),
        content: content_parts,
        tool_calls: Vec::new(),
        metadata,
    })
}

pub fn tool_results_to_messages(tool_results: Vec<(ToolCall, ToolResult)>) -> Vec<OllamaMessage> {
    let mut messages = Vec::new();

    for (tool_call, tool_result) in tool_results {
        let tool_call_obj = OllamaToolCall {
            id: Some(tool_call.id.clone()),
            function: OllamaToolCallFunction {
                name: tool_call.name.clone(),
                id: None,
                arguments: serde_json::from_str(&tool_call.arguments_json)
                    .unwrap_or(serde_json::Value::Null),
            },
        };

        messages.push(OllamaMessage {
            role: "assistant".to_string(),
            content: OllamaMessageContent {
                role: "assistant".to_string(),
                content: "".to_string(),
                images: None,
                tool_calls: Some(vec![tool_call_obj]),
            },
            tool_calls: None,
        });

        let result_text = match tool_result {
            ToolResult::Success(success) => success.result_json,
            ToolResult::Error(error) => format!("Error: {}", error.error_message),
        };

        messages.push(OllamaMessage {
            role: "tool".to_string(),
            content: OllamaMessageContent {
                role: "tool".to_string(),
                content: result_text,
                images: None,
                tool_calls: None,
            },
            tool_calls: None,
        });
    }

    messages
}
