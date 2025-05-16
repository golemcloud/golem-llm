use crate::client::{
    CreateModelResponseRequest, CreateModelResponseResponse, Detail, InnerInput, InnerInputItem,
    Input, InputItem, OutputItem, OutputMessageContent, Tool,
};
use base64::{engine::general_purpose, Engine as _};
use golem_llm::error::error_code_from_status;
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, ImageDetail, ImageSource,
    Message, ResponseMetadata, Role, ToolCall, ToolDefinition, ToolResult, Usage,
};
use reqwest::StatusCode;
use std::collections::HashMap;
use std::str::FromStr;

pub fn create_request(
    items: Vec<InputItem>,
    config: Config,
    tools: Vec<Tool>,
) -> CreateModelResponseRequest {
    let options = config
        .provider_options
        .into_iter()
        .map(|kv| (kv.key, kv.value))
        .collect::<HashMap<_, _>>();

    CreateModelResponseRequest {
        input: Input::List(items),
        model: config.model,
        temperature: config.temperature,
        max_output_tokens: config.max_tokens,
        tools,
        tool_choice: config.tool_choice,
        stream: false,
        top_p: options
            .get("top_p")
            .and_then(|top_p_s| top_p_s.parse::<f32>().ok()),
        user: options
            .get("user")
            .and_then(|user_s| user_s.parse::<String>().ok()),
    }
}

pub fn messages_to_input_items(messages: Vec<Message>) -> Vec<InputItem> {
    let mut items = Vec::new();
    for message in messages {
        let role = to_openai_role_name(message.role).to_string();
        let mut input_items = Vec::new();
        for content_part in message.content {
            input_items.push(content_part_to_inner_input_item(content_part));
        }

        items.push(InputItem::InputMessage {
            role,
            content: InnerInput::List(input_items),
        });
    }
    items
}

pub fn tool_results_to_input_items(tool_results: Vec<(ToolCall, ToolResult)>) -> Vec<InputItem> {
    let mut items = Vec::new();
    for (tool_call, tool_result) in tool_results {
        let tool_call = InputItem::ToolCall {
            arguments: tool_call.arguments_json,
            call_id: tool_call.id,
            name: tool_call.name,
        };
        let tool_result = match tool_result {
            ToolResult::Success(success) => InputItem::ToolResult {
                call_id: success.id,
                output: format!(r#"{{ "success": {} }}"#, success.result_json),
            },
            ToolResult::Error(error) => InputItem::ToolResult {
                call_id: error.id,
                output: format!(
                    r#"{{ "error": {{ "code": {}, "message": {} }} }}"#,
                    error.error_code.unwrap_or_default(),
                    error.error_message
                ),
            },
        };
        items.push(tool_call);
        items.push(tool_result);
    }
    items
}

pub fn tool_defs_to_tools(tool_definitions: &[ToolDefinition]) -> Result<Vec<Tool>, Error> {
    let mut tools = Vec::new();
    for tool_def in tool_definitions {
        match serde_json::from_str(&tool_def.parameters_schema) {
            Ok(value) => {
                let tool = Tool::Function {
                    name: tool_def.name.clone(),
                    description: tool_def.description.clone(),
                    parameters: Some(value),
                    strict: true,
                };
                tools.push(tool);
            }
            Err(error) => {
                Err(Error {
                    code: ErrorCode::InternalError,
                    message: format!(
                        "Failed to parse tool parameters for {}: {error}",
                        tool_def.name
                    ),
                    provider_error_json: None,
                })?;
            }
        }
    }
    Ok(tools)
}

pub fn to_openai_role_name(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Tool => "tool",
    }
}

pub fn content_part_to_inner_input_item(content_part: ContentPart) -> InnerInputItem {
    match content_part {
        ContentPart::Text(msg) => InnerInputItem::TextInput { text: msg },
        ContentPart::Image(image_source) => match image_source {
            ImageSource::Url(image_url) => InnerInputItem::ImageInput {
                image_url: image_url.url,
                detail: match image_url.detail {
                    Some(ImageDetail::Auto) => Detail::Auto,
                    Some(ImageDetail::Low) => Detail::Low,
                    Some(ImageDetail::High) => Detail::High,
                    None => Detail::default(),
                },
            },
            ImageSource::Data(image_data) => {
                // Convert binary data to base64 and format as data URL
                let base64_data = general_purpose::STANDARD.encode(&image_data.data);
                let data_url = format!("data:{};base64,{}", image_data.mime_type, base64_data);

                InnerInputItem::ImageInput {
                    image_url: data_url,
                    detail: match image_data.detail {
                        Some(ImageDetail::Auto) => Detail::Auto,
                        Some(ImageDetail::Low) => Detail::Low,
                        Some(ImageDetail::High) => Detail::High,
                        None => Detail::default(),
                    },
                }
            }
        },
    }
}

pub fn parse_error_code(code: String) -> ErrorCode {
    if let Some(code) = <u16 as FromStr>::from_str(&code)
        .ok()
        .and_then(|code| StatusCode::from_u16(code).ok())
    {
        error_code_from_status(code)
    } else {
        ErrorCode::InternalError
    }
}

pub fn process_model_response(response: CreateModelResponseResponse) -> ChatEvent {
    if let Some(error) = response.error {
        ChatEvent::Error(Error {
            code: parse_error_code(error.code),
            message: error.message,
            provider_error_json: None,
        })
    } else {
        let mut contents = Vec::new();
        let mut tool_calls = Vec::new();

        let metadata = create_response_metadata(&response);

        for output_item in response.output {
            match output_item {
                OutputItem::Message { content, .. } => {
                    for content in content {
                        match content {
                            OutputMessageContent::Text { text, .. } => {
                                contents.push(ContentPart::Text(text));
                            }
                            OutputMessageContent::Refusal { refusal, .. } => {
                                contents.push(ContentPart::Text(format!("Refusal: {refusal}")));
                            }
                        }
                    }
                }
                OutputItem::ToolCall {
                    arguments,
                    call_id,
                    name,
                    ..
                } => {
                    let tool_call = ToolCall {
                        id: call_id,
                        name,
                        arguments_json: arguments,
                    };
                    tool_calls.push(tool_call);
                }
            }
        }

        if contents.is_empty() {
            ChatEvent::ToolRequest(tool_calls)
        } else {
            ChatEvent::Message(CompleteResponse {
                id: response.id,
                content: contents,
                tool_calls,
                metadata,
            })
        }
    }
}

pub fn create_response_metadata(response: &CreateModelResponseResponse) -> ResponseMetadata {
    ResponseMetadata {
        finish_reason: None,
        usage: response.usage.as_ref().map(|usage| Usage {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            total_tokens: Some(usage.total_tokens),
        }),
        provider_id: Some(response.id.clone()),
        timestamp: Some(response.created_at.to_string()),
        provider_metadata_json: response.metadata.as_ref().map(|m| m.to_string()),
    }
}
