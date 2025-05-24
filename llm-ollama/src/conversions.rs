use crate::client::{ChatApi, ChatRequest, ChatResponse, Function, ModelOptions, Tool, ToolCall};
use base64::{engine::general_purpose, Engine as _};
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, FinishReason,
    ImageReference, Message, ResponseMetadata, Role, ToolCall as LlmToolCall, ToolDefinition,
    ToolResult, Usage,
};
use std::collections::HashMap;

pub fn messages_to_request(
    messages: Vec<Message>,
    config: Config,
    client: &ChatApi,
) -> Result<ChatRequest, Error> {
    let options = config
        .provider_options
        .into_iter()
        .map(|kv| (kv.key, kv.value))
        .collect::<HashMap<_, _>>();

    let mut ollama_messages = Vec::new();
    for message in &messages {
        let role = match message.role {
            Role::User => "user".to_string(),
            Role::Assistant => "assistant".to_string(),
            Role::System => "system".to_string(),
            Role::Tool => "user".to_string(), // Ollama doesn't have a separate tool role
        };

        let (content, images) = message_to_content_and_images(message, client)?;

        ollama_messages.push(crate::client::Message {
            role,
            content,
            images,
            tool_calls: None, // Will be set separately if needed
        });
    }

    let tools = if config.tools.is_empty() {
        None
    } else {
        let mut ollama_tools = Vec::new();
        for tool in &config.tools {
            ollama_tools.push(tool_definition_to_tool(tool)?);
        }
        Some(ollama_tools)
    };

    let model_options = ModelOptions {
        temperature: config.temperature,
        top_p: options
            .get("top_p")
            .and_then(|top_p_s| top_p_s.parse::<f32>().ok()),
        top_k: options
            .get("top_k")
            .and_then(|top_k_s| top_k_s.parse::<u32>().ok()),
        num_predict: config.max_tokens,
        stop: config.stop_sequences,
    };

    Ok(ChatRequest {
        model: config.model,
        messages: ollama_messages,
        tools,
        stream: Some(false),
        format: None,
        options: Some(model_options),
        keep_alive: options.get("keep_alive").cloned(),
    })
}

pub fn process_response(response: ChatResponse) -> ChatEvent {
    let mut contents = Vec::new();
    let mut tool_calls = Vec::new();

    // Parse content
    if !response.message.content.is_empty() {
        contents.push(ContentPart::Text(response.message.content));
    }

    // Parse tool calls
    if let Some(response_tool_calls) = response.message.tool_calls {
        for tool_call in response_tool_calls {
            tool_calls.push(LlmToolCall {
                id: generate_tool_call_id(), // Ollama might not provide IDs, so we generate them
                name: tool_call.function.name,
                arguments_json: serde_json::to_string(&tool_call.function.arguments).unwrap(),
            });
        }
    }

    let finish_reason = response
        .done_reason
        .as_ref()
        .map(|reason| match reason.as_str() {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            _ => FinishReason::Other,
        });

    // Calculate usage from response metadata
    let usage = Usage {
        input_tokens: response.prompt_eval_count,
        output_tokens: response.eval_count,
        total_tokens: response
            .prompt_eval_count
            .zip(response.eval_count)
            .map(|(input, output)| input + output),
    };

    let metadata = ResponseMetadata {
        finish_reason,
        usage: Some(usage),
        provider_id: Some(response.model.clone()),
        timestamp: Some(response.created_at),
        provider_metadata_json: None,
    };

    if contents.is_empty() && !tool_calls.is_empty() {
        ChatEvent::ToolRequest(tool_calls)
    } else {
        ChatEvent::Message(CompleteResponse {
            id: generate_response_id(), // Ollama doesn't provide response IDs
            content: contents,
            tool_calls,
            metadata,
        })
    }
}

pub fn tool_results_to_messages(
    tool_results: Vec<(LlmToolCall, ToolResult)>,
) -> Vec<crate::client::Message> {
    let mut messages = Vec::new();

    for (tool_call, tool_result) in tool_results {
        // Add assistant message with tool call
        messages.push(crate::client::Message {
            role: "assistant".to_string(),
            content: String::new(),
            images: None,
            tool_calls: Some(vec![ToolCall {
                function: crate::client::FunctionCall {
                    name: tool_call.name.clone(),
                    arguments: serde_json::from_str(&tool_call.arguments_json).unwrap_or_default(),
                },
            }]),
        });

        // Add user message with tool result
        let content = match tool_result {
            ToolResult::Success(success) => success.result_json,
            ToolResult::Error(error) => {
                format!("Error: {}", error.error_message)
            }
        };

        messages.push(crate::client::Message {
            role: "user".to_string(),
            content: format!("Tool {} returned: {}", tool_call.name, content),
            images: None,
            tool_calls: None,
        });
    }

    messages
}

fn message_to_content_and_images(
    message: &Message,
    client: &ChatApi,
) -> Result<(String, Option<Vec<String>>), Error> {
    let mut text_parts = Vec::new();
    let mut images = Vec::new();

    for content_part in &message.content {
        match content_part {
            ContentPart::Text(text) => text_parts.push(text.clone()),
            ContentPart::Image(image_reference) => {
                let base64_image = match image_reference {
                    ImageReference::Url(image_url) => {
                        // Download the image and convert to base64
                        client.download_image_as_base64(&image_url.url)?
                    }
                    ImageReference::Inline(image_source) => {
                        // Return just the base64 string, not the data URL format
                        // Ollama expects raw base64, not "data:image/type;base64,base64string"
                        general_purpose::STANDARD.encode(&image_source.data)
                    }
                };
                images.push(base64_image);
            }
        }
    }

    let content = text_parts.join("\n");
    let images = if images.is_empty() {
        None
    } else {
        Some(images)
    };

    Ok((content, images))
}

fn tool_definition_to_tool(tool: &ToolDefinition) -> Result<Tool, Error> {
    match serde_json::from_str(&tool.parameters_schema) {
        Ok(parameters) => Ok(Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters,
            },
        }),
        Err(error) => Err(Error {
            code: ErrorCode::InternalError,
            message: format!("Failed to parse tool parameters for {}: {error}", tool.name),
            provider_error_json: None,
        }),
    }
}

// Ollama doesn't provide tool call IDs, so we generate them
fn generate_tool_call_id() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut hasher = DefaultHasher::new();
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .hash(&mut hasher);
    format!("call_{:x}", hasher.finish())
}

// Ollama doesn't provide response IDs, so we generate them
// Probably there is a better way to do this
fn generate_response_id() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut hasher = DefaultHasher::new();
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .hash(&mut hasher);
    format!("resp_{:x}", hasher.finish())
}
