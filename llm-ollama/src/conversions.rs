use std::collections::HashMap;

use crate::client::{
    CompletionsRequest, CompletionsResponse, FunctionTool, MessageRequest, MessageRole,
    OllamaModelOptions, Tool,
};
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, FinishReason, ImageDetail,
    Message, ResponseMetadata, Role, ToolCall as golem_llm_ToolCall, ToolDefinition, ToolResult,
    Usage,
};

pub fn messages_to_request(
    messages: Vec<Message>,
    config: Config,
) -> Result<CompletionsRequest, Error> {
    let options = config
        .provider_options
        .into_iter()
        .map(|Kv| (Kv.key, Kv.value))
        .collect::<HashMap<_, _>>();

    let mut request_message = Vec::new();

    for message in messages {
        let message_role = match message.role {
            Role::Assistant => MessageRole::Assistant,
            Role::System => MessageRole::System,
            Role::User => MessageRole::User,
            Role::Tool => MessageRole::Tool,
        };

        let mut message_content = String::new();
        let mut attached_image = None;

        for content_part in message.content {
            match content_part {
                ContentPart::Text(text) => {
                    if !message_content.is_empty() {
                        message_content.push_str("\n");
                    }
                    message_content.push_str(&text);
                }
                ContentPart::Image(_image_url) => {
                    todo!("Ollama accept base64 image")
                }
            }
        }

        request_message.push(MessageRequest {
            content: message_content,
            role: message_role,
            images: attached_image,
            tools_calls: None,
        });
    }

    let mut tools = Vec::new();
    for tool in config.tools {
        let param = serde_json::from_str(&tool.parameters_schema).map_err(|err| Error {
            code: ErrorCode::InternalError,
            message: format!("Failed to parse tool parameters for {}: {err}", tool.name),
            provider_error_json: None,
        })?;
        tools.push(Tool {
            tool_type: String::from("function"),
            function: FunctionTool {
                description: tool.description.unwrap_or(String::new()),
                name: tool.name,
                parameters: param,
            },
        });
    }

    let ollama_options = OllamaModelOptions {
        min_p: parse_option(&options, "min_p"),
        temperature: config.temperature,
        top_p: parse_option(&options, "top_p"),
        top_k: parse_option(&options, "top_k"),
        num_predict: parse_option(&options, "num_predict"),
        stop: config.stop_sequences.clone(),
        repeat_penalty: parse_option(&options, "repeat_penalty"),
        num_ctx: parse_option(&options, "num_ctx"),
        seed: parse_option(&options, "seed"),
        mirostat: parse_option(&options, "mirostat"),
        mirostat_eta: parse_option(&options, "mirostat_eta"),
        mirostat_tau: parse_option(&options, "mirostat_tau"),
        num_gpu: parse_option(&options, "num_gpu"),
        num_thread: parse_option(&options, "num_thread"),
        penalize_newline: parse_option(&options, "penalize_newline"),
        num_keep: parse_option(&options, "num_keep"),
        typical_p: parse_option(&options, "typical_p"),
        repeat_last_n: parse_option(&options, "repeat_last_n"),
        presence_penalty: parse_option(&options, "presence_penalty"),
        frequency_penalty: parse_option(&options, "frequency_penalty"),
        numa: parse_option(&options, "numa"),
        num_batch: parse_option(&options, "num_batch"),
        main_gpu: parse_option(&options, "main_gpu"),
        use_mmap: parse_option(&options, "use_mmap"),
    };

    Ok(CompletionsRequest {
        model: Some(config.model),
        messages: Some(request_message),
        tools: Some(tools),
        format: options.get("format").map(|f| f.to_string()),
        options: Some(ollama_options),
        keep_alive: options.get("keep_alive").map(|k| k.to_string()),
        stream: Some(false),
    })
}

fn parse_option<T: std::str::FromStr>(options: &HashMap<String, String>, key: &str) -> Option<T> {
    options.get(key).and_then(|v| v.parse::<T>().ok())
}

pub fn process_response(response: CompletionsResponse) -> ChatEvent {
    if response.messages.is_some() {
        let mut chat_events = Vec::<golem_llm_ToolCall>::new();

        for message in response.messages.clone().unwrap() {
            if message.tool_calls.is_some() {
                for tool_call in message.tool_calls.clone().unwrap() {
                    chat_events.push(golem_llm_ToolCall {
                        id: tool_call.id,
                        name: tool_call.name,
                        arguments_json: tool_call.function.unwrap().arguments.to_string(),
                    });
                }
            }
        }

        if chat_events.len() > 0 {
            return ChatEvent::ToolRequest(chat_events);
        }

        let mut content = Vec::new();
        for message in response.messages.unwrap() {
            if message.content.is_some() {
                content.push(ContentPart::Text(message.content.unwrap()));
            }
        }

        let finish_reason = if response.done.unwrap_or(false) {
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
    } else {
        ChatEvent::Error(Error {
            code: ErrorCode::InternalError,
            message: String::from("No messages in response"),
            provider_error_json: None,
        })
    }
}
