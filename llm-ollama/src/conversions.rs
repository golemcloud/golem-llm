// This module contains functions for converting between the type system used
// by Golem LLM (defined in the WIT interface) and the types needed by Ollama's
// OpenAI compatibility API.

use crate::client::{
    OllamaApi, OllamaChatCompletionRequest, OllamaChatCompletionResponse, OllamaFunction,
    OllamaMessage, OllamaTool, OllamaToolCall, OllamaToolCallFunction, ToolChoice,
};
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, FinishReason, Message,
    ResponseMetadata, Role as GolemRole, ToolCall, ToolDefinition, ToolResult, Usage,
};
use std::collections::HashMap;

// Limitations:
//  - Image URL are not supported by Ollama,
//  - Fix: URL Image are downloaded converted to base64 and added to the content as image_url
//  - Which is supported by Ollama

//  - Tool calls are only supported in Ollama Unary calls, and not in Streaming calls

pub fn messages_to_request(
    messages: Vec<Message>,
    config: Config,
    client: &OllamaApi,
) -> Result<OllamaChatCompletionRequest, Error> {
    // Extract provider-specific options
    let options = config
        .provider_options
        .iter()
        .map(|kv| (kv.key.clone(), kv.value.clone()))
        .collect::<HashMap<_, _>>();

    // Convert each message to Ollama format
    let mut ollama_messages = Vec::new();
    for message in messages {
        ollama_messages.push(message_to_ollama_message(message, client)?);
    }

    // Convert tool definitions if present
    let tools = if config.tools.is_empty() {
        None
    } else {
        let mut tools = Vec::new();
        for tool in &config.tools {
            tools.push(tool_definition_to_tool(tool)?);
        }
        Some(tools)
    };

    // Handle tool choice configuration
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

    // Construct the final request
    Ok(OllamaChatCompletionRequest {
        model: config.model,
        messages: ollama_messages,
        frequency_penalty: options
            .get("frequency_penalty")
            .and_then(|v| v.parse::<f32>().ok()),
        max_tokens: config.max_tokens,
        presence_penalty: options
            .get("presence_penalty")
            .and_then(|v| v.parse::<f32>().ok()),
        temperature: config.temperature,
        top_p: options.get("top_p").and_then(|v| v.parse::<f32>().ok()),
        tools,
        tool_choice,
        stream: false,
        seed: options.get("seed").and_then(|v| v.parse::<i32>().ok()),
        stop: config.stop_sequences,
    })
}

fn message_to_ollama_message(message: Message, client: &OllamaApi) -> Result<OllamaMessage, Error> {
    // Convert role to string format
    let role = match message.role {
        GolemRole::User => "user".to_string(),
        GolemRole::Assistant => "assistant".to_string(),
        GolemRole::System => "system".to_string(),
        GolemRole::Tool => "tool".to_string(),
    };

    // For image support, we'll collect parts into a multimodal format if needed
    let mut has_images = false;
    let mut content_parts = Vec::new();
    let mut text_content = String::new();

    for part in &message.content {
        match part {
            ContentPart::Text(text) => {
                if !has_images {
                    // If we haven't seen any images yet, just append text
                    if !text_content.is_empty() {
                        text_content.push('\n');
                    }
                    text_content.push_str(text);
                } else {
                    // If there are images, add this as a text part
                    content_parts.push(serde_json::json!({
                        "type": "text",
                        "text": text
                    }));
                }
            }
            ContentPart::Image(image_url) => {
                has_images = true;
                // Convert previous text content to a part if there's any
                if !text_content.is_empty() {
                    content_parts.push(serde_json::json!({
                        "type": "text",
                        "text": text_content
                    }));
                    text_content.clear();
                }

                // Download and convert the image to base64
                match client.download_image_as_base64(&image_url.url) {
                    Ok(base64_url) => {
                        content_parts.push(serde_json::json!({
                            "type": "image_url",
                            "image_url": { "url": base64_url }
                        }));
                    }
                    Err(err) => {
                        log::warn!(
                            "Failed to download image {}: {}",
                            image_url.url,
                            err.message
                        );
                        // Add a placeholder text instead of failing the entire request
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": format!("[Could not load image from {}]", image_url.url)
                        }));
                    }
                }
            }
        }
    }

    // Create the final content based on whether we have images or not
    let content_container = if has_images {
        // If there's any remaining text, add it as a part
        if !text_content.is_empty() {
            content_parts.push(serde_json::json!({
                "type": "text",
                "text": text_content
            }));
        }

        // Use the multimodal content format
        crate::client::OllamaMessageContent::Multimodal {
            content: content_parts,
        }
    } else {
        // If no images, just use the text format
        crate::client::OllamaMessageContent::Text {
            content: text_content,
        }
    };

    Ok(OllamaMessage {
        role,
        content_container,
        tool_calls: None,
    })
}

fn tool_definition_to_tool(tool: &ToolDefinition) -> Result<OllamaTool, Error> {
    // Parse the parameters schema from string to JSON
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

fn map_finish_reason_to_enum(finish_reason: Option<&String>) -> Option<FinishReason> {
    finish_reason.map(|reason| match reason.as_str() {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    })
}

pub fn process_response(response: OllamaChatCompletionResponse) -> ChatEvent {
    let first_choice = match response.choices.first() {
        Some(choice) => choice,
        None => {
            return ChatEvent::Error(Error {
                code: ErrorCode::InternalError,
                message: "Response does not contain any choices".to_string(),
                provider_error_json: Some(serde_json::to_string(&response).unwrap_or_default()),
            })
        }
    };

    // Check if the response contains tool calls
    if let Some(tool_calls) = &first_choice.message.tool_calls {
        if !tool_calls.is_empty() {
            let golem_tool_calls = tool_calls
                .iter()
                .map(|tc| ToolCall {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments_json: tc.function.arguments.clone(),
                })
                .collect();

            return ChatEvent::ToolRequest(golem_tool_calls);
        }
    }

    // Process regular content response
    let content = match &first_choice.message.content_container {
        Some(crate::client::OllamaMessageContent::Text { content }) => {
            vec![ContentPart::Text(content.clone())]
        }
        Some(crate::client::OllamaMessageContent::Multimodal { content }) => {
            // Try to extract text components from multimodal content
            let mut text_parts = Vec::new();
            for part in content {
                if let Some(text_type) = part.get("type").and_then(|t| t.as_str()) {
                    if text_type == "text" {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            text_parts.push(ContentPart::Text(text.to_string()));
                        }
                    }
                }
            }
            if text_parts.is_empty() {
                vec![ContentPart::Text("".to_string())]
            } else {
                text_parts
            }
        }
        None => vec![ContentPart::Text("".to_string())],
    };

    let finish_reason = map_finish_reason_to_enum(first_choice.finish_reason.as_ref());

    let usage = response.usage.as_ref().map(|u| Usage {
        input_tokens: Some(u.prompt_tokens),
        output_tokens: Some(u.completion_tokens),
        total_tokens: Some(u.total_tokens),
    });

    // Generate provider metadata for durability
    let provider_metadata_json = response.usage.as_ref().map(|usage| {
        format!(
            r#"{{"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        )
    });

    let metadata = ResponseMetadata {
        finish_reason,
        usage,
        provider_id: Some(response.id.clone()),
        timestamp: Some(response.created.to_string()),
        provider_metadata_json,
    };

    ChatEvent::Message(CompleteResponse {
        id: response.id.clone(),
        content,
        tool_calls: Vec::new(),
        metadata,
    })
}

// Tool calls are not supported in Ollama Streaming calls
pub fn tool_results_to_messages(tool_results: Vec<(ToolCall, ToolResult)>) -> Vec<OllamaMessage> {
    let mut messages = Vec::new();

    for (tool_call, tool_result) in tool_results {
        // Create assistant message with tool calls
        let tool_call_obj = OllamaToolCall {
            id: tool_call.id.clone(),
            typ: "function".to_string(),
            function: OllamaToolCallFunction {
                name: tool_call.name.clone(),
                arguments: tool_call.arguments_json.clone(),
            },
        };

        // Add assistant message with tool call
        messages.push(OllamaMessage {
            role: "assistant".to_string(),
            content_container: crate::client::OllamaMessageContent::Text {
                content: "".to_string(),
            },
            tool_calls: Some(vec![tool_call_obj]),
        });

        // Add tool message with result
        let result_content = match tool_result {
            ToolResult::Success(success) => success.result_json,
            ToolResult::Error(error) => {
                format!("{{\"error\": \"{}\" }}", error.error_message)
            }
        };

        messages.push(OllamaMessage {
            role: "tool".to_string(),
            content_container: crate::client::OllamaMessageContent::Text {
                content: result_content,
            },
            tool_calls: None,
        });
    }

    messages
}
