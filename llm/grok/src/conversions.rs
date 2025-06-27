use crate::client::{CompletionsRequest, CompletionsResponse, Detail, Effort};
use base64::{engine::general_purpose, Engine as _};
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, FinishReason, ImageDetail,
    ImageReference, Message, ResponseMetadata, Role, ToolCall, ToolDefinition, ToolResult, Usage,
};
use std::collections::HashMap;

pub fn messages_to_request(
    messages: Vec<Message>,
    config: Config,
) -> Result<CompletionsRequest, Error> {
    let options = config
        .provider_options
        .into_iter()
        .map(|kv| (kv.key, kv.value))
        .collect::<HashMap<_, _>>();

    let mut completion_messages = Vec::new();
    for message in messages {
        match message.role {
            Role::User => completion_messages.push(crate::client::Message::User {
                name: message.name,
                content: convert_content_parts(message.content),
            }),
            Role::Assistant => completion_messages.push(crate::client::Message::Assistant {
                name: message.name,
                content: Some(convert_content_parts(message.content)),
                tool_calls: None,
            }),
            Role::System => completion_messages.push(crate::client::Message::System {
                name: message.name,
                content: convert_content_parts(message.content),
            }),
            Role::Tool => completion_messages.push(crate::client::Message::Tool {
                name: message.name,
                content: convert_content_parts(message.content),
                tool_call_id: None,
            }),
        }
    }

    let mut tools = Vec::new();
    for tool in config.tools {
        tools.push(tool_definition_to_tool(tool)?)
    }

    Ok(CompletionsRequest {
        messages: completion_messages,
        model: config.model,
        frequency_penalty: options
            .get("frequency_penalty")
            .and_then(|fp_s| fp_s.parse::<f32>().ok()),
        max_completion_tokens: config.max_tokens,
        n: options.get("n").and_then(|n_s| n_s.parse::<u32>().ok()),
        presence_penalty: options
            .get("presence_penalty")
            .and_then(|pp_s| pp_s.parse::<f32>().ok()),
        reasoning_effort: options
            .get("reasoning_effort")
            .and_then(|effort_s| effort_s.parse::<Effort>().ok()),
        seed: options
            .get("seed")
            .and_then(|seed_s| seed_s.parse::<u32>().ok()),
        stop: config.stop_sequences,
        stream: Some(false),
        stream_options: None,
        temperature: config.temperature,
        tool_choice: config.tool_choice,
        tools,
        top_logprobs: options
            .get("top_logprobs")
            .and_then(|top_logprobs_s| top_logprobs_s.parse::<u8>().ok()),
        top_p: options
            .get("top_p")
            .and_then(|top_p_s| top_p_s.parse::<f32>().ok()),
        user: options.get("user_id").cloned(),
    })
}

pub fn process_response(response: CompletionsResponse) -> ChatEvent {
    let choice = response.choices.first();
    if let Some(choice) = choice {
        let mut contents = Vec::new();
        let mut tool_calls = Vec::new();

        if let Some(content) = &choice.message.content {
            contents.push(ContentPart::Text(content.clone()));
        }

        let empty = Vec::new();
        for tool_call in choice.message.tool_calls.as_ref().unwrap_or(&empty) {
            tool_calls.push(convert_tool_call(tool_call));
        }

        if contents.is_empty() {
            ChatEvent::ToolRequest(tool_calls)
        } else {
            let metadata = ResponseMetadata {
                finish_reason: choice.finish_reason.as_ref().map(convert_finish_reason),
                usage: response.usage.as_ref().map(convert_usage),
                provider_id: None,
                timestamp: Some(response.created.to_string()),
                provider_metadata_json: None,
            };

            ChatEvent::Message(CompleteResponse {
                id: response.id,
                content: contents,
                tool_calls,
                metadata,
            })
        }
    } else {
        ChatEvent::Error(Error {
            code: ErrorCode::InternalError,
            message: "No choices in response".to_string(),
            provider_error_json: None,
        })
    }
}

pub fn tool_results_to_messages(
    tool_results: Vec<(ToolCall, ToolResult)>,
) -> Vec<crate::client::Message> {
    let mut messages = Vec::new();
    for (tool_call, tool_result) in tool_results {
        messages.push(crate::client::Message::Assistant {
            content: None,
            name: None,
            tool_calls: Some(vec![crate::client::ToolCall::Function {
                function: crate::client::FunctionCall {
                    arguments: tool_call.arguments_json,
                    name: tool_call.name,
                },
                id: tool_call.id.clone(),
                index: None,
            }]),
        });
        let content = match tool_result {
            ToolResult::Success(success) => crate::client::ContentPart::TextInput {
                text: success.result_json,
            },
            ToolResult::Error(failure) => crate::client::ContentPart::TextInput {
                text: failure.error_message,
            },
        };
        messages.push(crate::client::Message::Tool {
            name: None,
            content: crate::client::Content::List(vec![content]),
            tool_call_id: Some(tool_call.id),
        });
    }
    messages
}

pub fn convert_tool_call(tool_call: &crate::client::ToolCall) -> ToolCall {
    match tool_call {
        crate::client::ToolCall::Function { function, id, .. } => ToolCall {
            id: id.clone(),
            name: function.name.clone(),
            arguments_json: function.arguments.clone(),
        },
    }
}

fn convert_content_parts(contents: Vec<ContentPart>) -> crate::client::Content {
    let mut result = Vec::new();
    for content in contents {
        match content {
            ContentPart::Text(text) => result.push(crate::client::ContentPart::TextInput { text }),
            ContentPart::Image(image_reference) => match image_reference {
                ImageReference::Url(image_url) => {
                    result.push(crate::client::ContentPart::ImageInput {
                        image_url: crate::client::ImageUrl {
                            url: image_url.url,
                            detail: image_url.detail.map(|d| d.into()),
                        },
                    })
                }
                ImageReference::Inline(image_source) => {
                    let base64_data = general_purpose::STANDARD.encode(&image_source.data);
                    let media_type = &image_source.mime_type; // This is already a string
                    result.push(crate::client::ContentPart::ImageInput {
                        image_url: crate::client::ImageUrl {
                            url: format!("data:{media_type};base64,{base64_data}"),
                            detail: image_source.detail.map(|d| d.into()),
                        },
                    });
                }
            },
        }
    }
    crate::client::Content::List(result)
}

impl From<ImageDetail> for Detail {
    fn from(value: ImageDetail) -> Self {
        match value {
            ImageDetail::Auto => Self::Auto,
            ImageDetail::Low => Self::Low,
            ImageDetail::High => Self::High,
        }
    }
}

pub fn convert_finish_reason(value: &crate::client::FinishReason) -> FinishReason {
    match value {
        crate::client::FinishReason::Stop => FinishReason::Stop,
        crate::client::FinishReason::Length => FinishReason::Length,
        crate::client::FinishReason::EndTurn => FinishReason::Other,
        crate::client::FinishReason::ToolCalls => FinishReason::ToolCalls,
    }
}

pub fn convert_usage(value: &crate::client::Usage) -> Usage {
    Usage {
        input_tokens: Some(value.prompt_tokens),
        output_tokens: Some(value.completion_tokens),
        total_tokens: Some(value.total_tokens),
    }
}

fn tool_definition_to_tool(tool: ToolDefinition) -> Result<crate::client::Tool, Error> {
    match serde_json::from_str(&tool.parameters_schema) {
        Ok(value) => Ok(crate::client::Tool::Function {
            function: crate::client::Function {
                name: tool.name,
                description: tool.description,
                parameters: Some(value),
            },
        }),
        Err(error) => Err(Error {
            code: ErrorCode::InternalError,
            message: format!("Failed to parse tool parameters for {}: {error}", tool.name),
            provider_error_json: None,
        }),
    }
}
