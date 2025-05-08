use crate::client::{
    OllamaChatRequest, OllamaChatResponse, OllamaFunction, OllamaMessage, OllamaTool,
    OllamaToolCall, OllamaToolCallFunction, RequestOptions, Role,
};
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, FinishReason, Message,
    ResponseMetadata, Role as GolemRole, ToolCall, ToolDefinition, ToolResult, Usage,
};
use std::collections::HashMap;

pub fn messages_to_request(
    messages: Vec<Message>,
    config: Config,
) -> Result<OllamaChatRequest, Error> {
    let options = config
        .provider_options
        .iter()
        .map(|kv| (kv.key.clone(), kv.value.clone()))
        .collect::<HashMap<_, _>>();
    let mut ollama_messages = Vec::new();
    for message in messages {
        ollama_messages.push(message_to_ollama_message(message)?);
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

    let request_options = create_request_options(&config, &options);

    Ok(OllamaChatRequest {
        model: config.model,
        messages: ollama_messages,
        tools,
        format: options.get("format").map(|f| f.to_string()),
        options: request_options,
        template: options.get("template").map(|t| t.to_string()),
        keep_alive: options.get("keep_alive").map(|k| k.to_string()),
        stream: false,
    })
}

fn create_request_options(
    config: &Config,
    options: &HashMap<String, String>,
) -> Option<RequestOptions> {
    let request_options = RequestOptions {
        temperature: config.temperature,
        top_p: options.get("top_p").and_then(|v| v.parse::<f32>().ok()),
        top_k: options.get("top_k").and_then(|v| v.parse::<i32>().ok()),
        num_predict: options
            .get("num_predict")
            .and_then(|v| v.parse::<i32>().ok()),
        stop: config.stop_sequences.clone(),
        repeat_penalty: options
            .get("repeat_penalty")
            .and_then(|v| v.parse::<f32>().ok()),
        num_ctx: options.get("num_ctx").and_then(|v| v.parse::<i32>().ok()),
        seed: options.get("seed").and_then(|v| v.parse::<i32>().ok()),
        mirostat: options.get("mirostat").and_then(|v| v.parse::<i32>().ok()),
        mirostat_eta: options
            .get("mirostat_eta")
            .and_then(|v| v.parse::<f32>().ok()),
        mirostat_tau: options
            .get("mirostat_tau")
            .and_then(|v| v.parse::<f32>().ok()),
        num_gpu: options.get("num_gpu").and_then(|v| v.parse::<i32>().ok()),
        num_thread: options
            .get("num_thread")
            .and_then(|v| v.parse::<i32>().ok()),
        tfs_z: options.get("tfs_z").and_then(|v| v.parse::<f32>().ok()),
        penalize_newline: options
            .get("penalize_newline")
            .and_then(|v| v.parse::<bool>().ok()),
    };

    let is_stop_empty = request_options
        .stop
        .as_ref()
        .map(|s| s.is_empty())
        .unwrap_or(true);

    // Check if any fields are set
    if request_options.temperature.is_none()
        && request_options.top_p.is_none()
        && request_options.top_k.is_none()
        && request_options.num_predict.is_none()
        && is_stop_empty
        && request_options.repeat_penalty.is_none()
        && request_options.num_ctx.is_none()
        && request_options.seed.is_none()
        && request_options.mirostat.is_none()
        && request_options.mirostat_eta.is_none()
        && request_options.mirostat_tau.is_none()
        && request_options.num_gpu.is_none()
        && request_options.num_thread.is_none()
        && request_options.tfs_z.is_none()
        && request_options.penalize_newline.is_none()
    {
        None
    } else {
        Some(request_options)
    }
}

fn message_to_ollama_message(message: Message) -> Result<OllamaMessage, Error> {
    let role = match message.role {
        GolemRole::User => Role::User,
        GolemRole::Assistant => Role::Assistant,
        GolemRole::System => Role::System,
        GolemRole::Tool => Role::Tool,
    };

    let mut content = String::new();
    let mut images = None;

    for part in message.content {
        match part {
            ContentPart::Text(text) => {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&text);
            }
            ContentPart::Image(image_url) => {
                // Ollama expects base64-encoded images
                // For this implementation, we're just collecting the URLs
                if images.is_none() {
                    images = Some(Vec::new());
                }
                if let Some(imgs) = &mut images {
                    imgs.push(image_url.url);
                }
            }
        }
    }

    Ok(OllamaMessage {
        role,
        content,
        images,
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
    if let Some(tool_calls) = &response.message.tool_calls {
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

    let content = vec![ContentPart::Text(response.message.content)];

    let finish_reason = if response.done {
        Some(FinishReason::Stop)
    } else {
        None
    };

    let usage = Usage {
        input_tokens: response.prompt_eval_count.map(|c| c as u32),
        output_tokens: response.eval_count.map(|c| c as u32),
        total_tokens: None,
    };

    let timestamp = response.created_at.clone();

    let metadata = ResponseMetadata {
        finish_reason,
        usage: Some(usage),
        provider_id: None,
        timestamp: Some(timestamp.clone()),
        provider_metadata_json: None,
    };

    ChatEvent::Message(CompleteResponse {
        id: format!("ollama-{}", timestamp),
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

        let tool_calls = vec![tool_call_obj];
        messages.push(OllamaMessage {
            role: Role::Assistant,
            content: String::new(),
            images: None,
            tool_calls: Some(tool_calls),
        });

        let result_content = match tool_result {
            ToolResult::Success(success) => success.result_json,
            ToolResult::Error(error) => format!("Error: {}", error.error_message),
        };

        messages.push(OllamaMessage {
            role: Role::Tool,
            content: result_content,
            images: None,
            tool_calls: None,
        });
    }

    messages
}
