use std::collections::HashMap;

use crate::client::{
    image_to_base64, CompletionsRequest, CompletionsResponse, FunctionTool, MessageRequest,
    MessageRole, OllamaModelOptions, Tool,
};
use base64::{engine::general_purpose, Engine};
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, FinishReason,
    ImageReference, Message, ResponseMetadata, Role, ToolCall as golem_llm_ToolCall, ToolResult,
    Usage,
};
use log::trace;

pub fn messages_to_request(
    messages: Vec<Message>,
    config: Config,
    tool_results: Option<Vec<(golem_llm_ToolCall, ToolResult)>>,
) -> Result<CompletionsRequest, Error> {
    let options = config
        .provider_options
        .into_iter()
        .map(|kv| (kv.key, kv.value))
        .collect::<HashMap<_, _>>();

    let mut request_message = Vec::new();

    for message in messages {
        let message_role = match message.role {
            Role::Assistant => MessageRole::Assistant,
            Role::System => MessageRole::System,
            Role::User => MessageRole::User,
            Role::Tool => MessageRole::User, // Ollama treats tool results as user input
        };

        let mut message_content = String::new();
        let mut attached_image = Vec::new();

        for content_part in message.content {
            match content_part {
                ContentPart::Text(text) => {
                    if !message_content.is_empty() {
                        message_content.push('\n');
                    }
                    message_content.push_str(&text);
                }
                ContentPart::Image(reference) => match reference {
                    ImageReference::Url(image_url) => {
                        let url = &image_url.url;
                        match image_to_base64(url) {
                            Ok(image) => attached_image.push(image),
                            Err(err) => {
                                trace!("Failed to encode image: {url}\nError: {err}\n");
                            }
                        }
                    }
                    ImageReference::Inline(image_source) => {
                        let base64_data = general_purpose::STANDARD.encode(&image_source.data);
                        attached_image.push(base64_data);
                    }
                },
            }
        }

        request_message.push(MessageRequest {
            content: message_content,
            role: message_role,
            images: if attached_image.is_empty() {
                None
            } else {
                Some(attached_image)
            },
            tools_calls: None,
        });
    }

    if let Some(tool_results) = tool_results {
        request_message.extend(tool_results_to_messages(tool_results));
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
                description: tool.description.unwrap_or_default(),
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
        format: options.get("format").cloned(),
        options: Some(ollama_options),
        keep_alive: options.get("keep_alive").cloned(),
        stream: Some(false),
    })
}

fn tool_results_to_messages(
    tool_results: Vec<(golem_llm_ToolCall, ToolResult)>,
) -> Vec<MessageRequest> {
    let mut messages = Vec::new();

    for (tool_call, result) in tool_results {
        let content = match result {
            ToolResult::Success(success) => {
                format!("[ToolCall Result]: Successed , [ToolCall ID]: {}, [ToolCall Name]: {}, [Result]: {}] ",success.id,success.name,success.result_json )
            },
            ToolResult::Error(error) => format!("[ToolCall Result]: Failed, [ToolCall ID]: {}, [ErrorName]: {}, [ErrorCode]: {}, [Error]: {}",error.id, error.name, error.error_code.unwrap_or_default(), error.error_message),
        };
        messages.push(MessageRequest {
            role: MessageRole::Assistant,
            // For better durability, we will add the tool call result in a structured format.
            // This will help in retying and contnuing the interrupted conversation.
            // This will help preventing branching conversations and repeating the tool call.
            content,
            images: None,
            // This is the tool called by llm
            tools_calls: Some(vec![Tool {
                tool_type: String::from("function"),
                function: FunctionTool {
                    name: tool_call.name,
                    description: String::new(),
                    parameters: serde_json::json!({}),
                },
            }]),
        });
    }
    messages
}

fn parse_option<T: std::str::FromStr>(options: &HashMap<String, String>, key: &str) -> Option<T> {
    options.get(key).and_then(|v| v.parse::<T>().ok())
}

pub fn process_response(response: CompletionsResponse) -> ChatEvent {
    if let Some(ref message) = response.message {
        let mut content = Vec::<ContentPart>::new();
        let mut tool_calls = Vec::<golem_llm_ToolCall>::new();

        if let Some(ref message_content) = message.content {
            content.push(ContentPart::Text(message_content.clone()));
        }

        if let Some(ref message_tool_calls) = message.tool_calls {
            for tool_call in message_tool_calls {
                tool_calls.push(golem_llm_ToolCall {
                    id: format!("ollama-{}", response.created_at.clone()),
                    name: tool_call.name.clone().unwrap_or_default(),
                    arguments_json: tool_call.function.as_ref().unwrap().arguments.to_string(),
                });
            }
        }

        let finish_reason = if response.done.unwrap_or(false) {
            Some(FinishReason::Stop)
        } else {
            None
        };
        let input_tokens = response.prompt_eval_count.map(|c| c as u32);
        let output_tokens = response.eval_count.map(|c| c as u32);

        let usage = Usage {
            input_tokens,
            output_tokens,
            total_tokens: Some(input_tokens.unwrap_or(0) + output_tokens.unwrap_or(0)),
        };

        let timestamp = response.created_at.clone();

        let metadata = ResponseMetadata {
            finish_reason,
            usage: Some(usage),
            provider_id: Some("ollama".to_string()),
            timestamp: Some(timestamp.clone()),
            provider_metadata_json: Some(get_provider_metadata(&response)),
        };

        ChatEvent::Message(CompleteResponse {
            id: format!("ollama-{timestamp}"),
            content,
            tool_calls,
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

pub fn get_provider_metadata(response: &CompletionsResponse) -> String {
    format!(
        r#"{{
    "total_duration":"{}",
    "load_duration":"{}",
    "prompt_eval_duration":{},
    "eval_duration":{},
    "context":{},
    }}"#,
        response.total_duration.unwrap_or(0),
        response.load_duration.unwrap_or(0),
        response.prompt_eval_duration.unwrap_or(0),
        response.eval_duration.unwrap_or(0),
        response.eval_count.unwrap_or(0)
    )
}
