use aws_smithy_types::{Document, Number};
use std::collections::HashMap;

use aws_sdk_bedrockruntime::{
    self as bedrock,
    error::SdkError,
    operation::{converse, converse_stream},
    types::{
        ContentBlockDeltaEvent, ContentBlockStartEvent, ConversationRole,
        ConverseStreamMetadataEvent, ConverseStreamOutput, ImageBlock, ImageFormat,
        InferenceConfiguration, MessageStopEvent, SystemContentBlock, Tool, ToolConfiguration,
        ToolInputSchema, ToolSpecification, ToolUseBlock,
    },
};
use golem_llm::golem::llm::llm;

#[derive(Debug)]
pub struct BedrockInput {
    pub model_id: String,
    pub system_instructions: Vec<SystemContentBlock>,
    pub messages: Vec<bedrock::types::Message>,
    pub inference_configuration: InferenceConfiguration,
    pub tools: Option<ToolConfiguration>,
    pub additional_fields: aws_smithy_types::Document,
}

impl BedrockInput {
    pub fn from(
        messages: Vec<llm::Message>,
        config: llm::Config,
        tool_results: Option<Vec<(llm::ToolCall, llm::ToolResult)>>,
    ) -> Result<Self, llm::Error> {
        let (mut user_messages, system_instructions) =
            messages_to_bedrock_message_groups(messages)?;

        if let Some(tool_results) = tool_results {
            user_messages.extend(tool_call_results_to_bedrock_tools(tool_results)?);
        }

        let options = config
            .provider_options
            .into_iter()
            .map(|kv| (kv.key, Document::String(kv.value)))
            .collect::<HashMap<_, _>>();

        Ok(BedrockInput {
            model_id: config.model,
            inference_configuration: InferenceConfiguration::builder()
                .set_max_tokens(config.max_tokens.map(|x| x as i32))
                .set_temperature(config.temperature)
                .set_stop_sequences(config.stop_sequences)
                .set_top_p(options.get("top_p").and_then(|v| match v {
                    Document::String(v) => v.parse::<f32>().ok(),
                    _ => None,
                }))
                .build(),
            messages: user_messages,
            system_instructions,
            tools: tool_defs_to_bedrock_tool_config(config.tools)?,
            additional_fields: Document::Object(options),
        })
    }
}

fn tool_call_results_to_bedrock_tools(
    results: Vec<(llm::ToolCall, llm::ToolResult)>,
) -> Result<Vec<bedrock::types::Message>, llm::Error> {
    let mut tool_calls: Vec<bedrock::types::ContentBlock> = vec![];
    let mut tool_results: Vec<bedrock::types::ContentBlock> = vec![];

    for (tool_call, tool_result) in results {
        tool_calls.push(bedrock::types::ContentBlock::ToolUse(
            bedrock::types::ToolUseBlock::builder()
                .tool_use_id(tool_call.id.clone())
                .name(tool_call.name)
                .input(json_str_to_smithy_document(&tool_call.arguments_json)?)
                .build()
                .unwrap(),
        ));

        tool_results.push(bedrock::types::ContentBlock::ToolResult(
            bedrock::types::ToolResultBlock::builder()
                .tool_use_id(tool_call.id)
                .content(bedrock::types::ToolResultContentBlock::Text(
                    match tool_result {
                        llm::ToolResult::Success(success) => success.result_json,
                        llm::ToolResult::Error(failure) => failure.error_message,
                    },
                ))
                .build()
                .unwrap(),
        ));
    }

    Ok(vec![
        bedrock::types::Message::builder()
            .role(ConversationRole::Assistant)
            .set_content(Some(tool_calls))
            .build()
            .unwrap(),
        bedrock::types::Message::builder()
            .role(ConversationRole::User)
            .set_content(Some(tool_results))
            .build()
            .unwrap(),
    ])
}

fn tool_defs_to_bedrock_tool_config(
    tools: Vec<llm::ToolDefinition>,
) -> Result<Option<ToolConfiguration>, llm::Error> {
    if tools.is_empty() {
        return Ok(None);
    }

    let mut specs: Vec<Tool> = vec![];

    for def in tools {
        let schema = json_str_to_smithy_document(&def.parameters_schema)?;

        specs.push(Tool::ToolSpec(
            ToolSpecification::builder()
                .name(def.name)
                .set_description(def.description)
                .input_schema(ToolInputSchema::Json(schema))
                .build()
                .unwrap(),
        ));
    }

    Ok(Some(
        ToolConfiguration::builder()
            .set_tools(Some(specs))
            .build()
            .unwrap(),
    ))
}

fn messages_to_bedrock_message_groups(
    messages: Vec<llm::Message>,
) -> Result<(Vec<bedrock::types::Message>, Vec<SystemContentBlock>), llm::Error> {
    let mut user_messages: Vec<bedrock::types::Message> = vec![];
    let mut system_instructions: Vec<SystemContentBlock> = vec![];

    for message in messages {
        if message.role == llm::Role::System {
            for content in message.content {
                if let llm::ContentPart::Text(text) = content {
                    system_instructions.push(SystemContentBlock::Text(text));
                }
            }
        } else {
            let bedrock_content = content_part_to_bedrock_content_blocks(message.content)?;
            user_messages.push(
                bedrock::types::Message::builder()
                    .role(if message.role == llm::Role::User {
                        ConversationRole::User
                    } else {
                        ConversationRole::Assistant
                    })
                    .set_content(Some(bedrock_content))
                    .build()
                    .unwrap(),
            );
        }
    }
    Ok((user_messages, system_instructions))
}

fn content_part_to_bedrock_content_blocks(
    content_parts: Vec<llm::ContentPart>,
) -> Result<Vec<bedrock::types::ContentBlock>, llm::Error> {
    let mut bedrock_content_blocks: Vec<bedrock::types::ContentBlock> = vec![];
    for part in content_parts {
        match part {
            llm::ContentPart::Text(text) => {
                bedrock_content_blocks.push(bedrock::types::ContentBlock::Text(text.to_owned()));
            }
            llm::ContentPart::Image(image) => {
                bedrock_content_blocks.push(image_ref_to_bedrock_image_content_block(image)?);
            }
        }
    }

    Ok(bedrock_content_blocks)
}

fn image_ref_to_bedrock_image_content_block(
    image_reference: llm::ImageReference,
) -> Result<bedrock::types::ContentBlock, llm::Error> {
    Ok(match image_reference {
        llm::ImageReference::Inline(image) => bedrock::types::ContentBlock::Image(
            ImageBlock::builder()
                .format(str_to_bedrock_mime_type(image.mime_type.as_ref())?)
                .source(bedrock::types::ImageSource::Bytes(image.data.into()))
                .build()
                .unwrap(),
        ),
        llm::ImageReference::Url(url) => get_image_content_block_from_url(url.url.as_ref())?,
    })
}

fn get_image_content_block_from_url(url: &str) -> Result<bedrock::types::ContentBlock, llm::Error> {
    let bytes = get_bytes_from_url(url)?;

    let kind = infer::get(&bytes);

    let mime = match kind {
        Some(kind) => str_to_bedrock_mime_type(kind.mime_type())?,
        None => {
            return Err(custom_error(
                llm::ErrorCode::InvalidRequest,
                format!(
                    "Could not infer the mime type of the image downloaded from url: {}",
                    url
                ),
            ));
        }
    };

    Ok(bedrock::types::ContentBlock::Image(
        ImageBlock::builder()
            .format(mime)
            .source(bedrock::types::ImageSource::Bytes(bytes.into()))
            .build()
            .unwrap(),
    ))
}

fn get_bytes_from_url(url: &str) -> Result<Vec<u8>, llm::Error> {
    let client = reqwest::Client::builder()
        .build()
        .expect("Failed to initialize HTTP client");

    let response = client.get(url).send().map_err(|err| {
        custom_error(
            llm::ErrorCode::InvalidRequest,
            format!("Could not read image bytes from url: {url}, cause: {err}"),
        )
    })?;
    if !response.status().is_success() {
        return Err(custom_error(
            llm::ErrorCode::InvalidRequest,
            format!(
                "Could not read image bytes from url: {url}, cause: request failed with status: {}",
                response.status()
            ),
        ));
    }

    let bytes = response.bytes().map_err(|err| {
        custom_error(
            llm::ErrorCode::InvalidRequest,
            format!("Could not read image bytes from url: {url}, cause: {err}"),
        )
    })?;

    Ok(bytes.to_vec())
}

fn str_to_bedrock_mime_type(mime_type: &str) -> Result<ImageFormat, llm::Error> {
    match mime_type {
        "image/png" => Ok(ImageFormat::Png),
        "image/jpeg" => Ok(ImageFormat::Jpeg),
        "image/webp" => Ok(ImageFormat::Webp),
        "image/gif" => Ok(ImageFormat::Gif),
        other => Err(llm::Error {
            code: llm::ErrorCode::Unsupported,
            message: format!("Unsupported image type: {}", other),
            provider_error_json: None,
        }),
    }
}

pub fn converse_output_to_tool_calls(
    response: converse::ConverseOutput,
) -> Result<Vec<llm::ToolCall>, llm::Error> {
    let output = response.output().ok_or(custom_error(
        llm::ErrorCode::InternalError,
        "An error occurred while converting to tool calls: expected output to not be None"
            .to_owned(),
    ))?;

    match output.as_message() {
        Err(_) => Err(custom_error(
            llm::ErrorCode::InternalError,
            "An error occurred while converting to tool calls: expected output to be a Message"
                .to_owned(),
        )),
        Ok(message) => {
            let mut tool_calls: Vec<llm::ToolCall> = vec![];
            for block in message.content.clone() {
                if let bedrock::types::ContentBlock::ToolUse(tool) = block {
                    tool_calls.push(bedrock_tool_use_to_llm_tool_call(tool)?);
                }
            }
            Ok(tool_calls)
        }
    }
}

pub fn converse_output_to_complete_response(
    response: converse::ConverseOutput,
) -> Result<llm::CompleteResponse, llm::Error> {
    let output = response.output().ok_or(custom_error(
        llm::ErrorCode::InternalError,
        "An error occurred while converting to complete response: expected output to be not be None"
            .to_owned(),
    ))?;

    match output.as_message() {
        Err(_) => Err(custom_error(
            llm::ErrorCode::InternalError,
            "An error occurred while converting to complete response: expected output to be a Message"
               .to_owned(),
        )),
        Ok(message) => {
            let mut content_parts: Vec<llm::ContentPart> = vec![];
            let mut tool_calls: Vec<llm::ToolCall> = vec![];
            for block in message.content.clone() {
                match block {
                    bedrock::types::ContentBlock::Text(text) => {
                        content_parts.push(llm::ContentPart::Text(text.to_owned()));
                    }
                    bedrock::types::ContentBlock::Image(image) => {
                        content_parts.push(bedrock_image_to_llm_content_part(image));
                    }
                    bedrock::types::ContentBlock::ToolUse(tool) => {
                        tool_calls.push(bedrock_tool_use_to_llm_tool_call(tool)?);
                    }
                    _ => {}
                }
            }
            let metadata = converse_output_to_response_metadata(&response);
            Ok(llm::CompleteResponse {
                // bedrock does not return an id as part of the response struct.
                // there may be one present in `additional_model_response_fields`
                // but the schema varies depending on the model being invoked. Leaving it empty for now
                // until we have a better solution for this.
                id: "".to_owned(),
                content: content_parts,
                tool_calls,
                metadata,
            })
        }
    }
}

fn bedrock_tool_use_to_llm_tool_call(tool: ToolUseBlock) -> Result<llm::ToolCall, llm::Error> {
    Ok(llm::ToolCall {
        id: tool.tool_use_id,
        name: tool.name,
        arguments_json: serde_json::to_string(&smithy_document_to_json_value(tool.input)).map_err(
            |err| {
                custom_error(
                    llm::ErrorCode::InternalError,
                    format!("An error occurred while deserializing tool use arguments: {err}"),
                )
            },
        )?,
    })
}

fn converse_output_to_response_metadata(
    response: &converse::ConverseOutput,
) -> llm::ResponseMetadata {
    llm::ResponseMetadata {
        finish_reason: Some(bedrock_stop_reason_to_finish_reason(response.stop_reason())),
        usage: response.usage().map(bedrock_usage_to_llm_usage),
        provider_id: Some("bedrock".to_owned()),
        provider_metadata_json: response
            .additional_model_response_fields
            .clone()
            .and_then(smithy_document_to_metadata_json),
        timestamp: None,
    }
}

fn smithy_document_to_metadata_json(doc: Document) -> Option<String> {
    serde_json::to_string(&smithy_document_to_json_value(doc)).ok()
}

fn bedrock_usage_to_llm_usage(usage: &bedrock::types::TokenUsage) -> llm::Usage {
    llm::Usage {
        input_tokens: Some(usage.input_tokens() as u32),
        output_tokens: Some(usage.output_tokens() as u32),
        total_tokens: Some(usage.total_tokens() as u32),
    }
}

fn bedrock_stop_reason_to_finish_reason(reason: &bedrock::types::StopReason) -> llm::FinishReason {
    match reason {
        bedrock::types::StopReason::StopSequence | bedrock::types::StopReason::EndTurn => {
            llm::FinishReason::Stop
        }
        bedrock::types::StopReason::ToolUse => llm::FinishReason::ToolCalls,
        bedrock::types::StopReason::MaxTokens => llm::FinishReason::Length,
        bedrock::types::StopReason::ContentFiltered
        | bedrock::types::StopReason::GuardrailIntervened => llm::FinishReason::ContentFilter,
        _ => llm::FinishReason::Other,
    }
}

fn bedrock_image_to_llm_content_part(block: bedrock::types::ImageBlock) -> llm::ContentPart {
    let mime_type = format!("image/{}", block.format.as_str());

    let reference = match block.source {
        Some(bedrock::types::ImageSource::Bytes(bytes)) => {
            llm::ImageReference::Inline(llm::ImageSource {
                mime_type,
                data: bytes.into(),
                detail: None,
            })
        }
        Some(bedrock::types::ImageSource::S3Location(location)) => {
            llm::ImageReference::Url(llm::ImageUrl {
                url: location.uri,
                detail: None,
            })
        }
        _ => llm::ImageReference::Inline(llm::ImageSource {
            mime_type,
            data: vec![],
            detail: None,
        }),
    };

    llm::ContentPart::Image(reference)
}

pub fn converse_stream_output_to_stream_event(
    event: ConverseStreamOutput,
) -> Option<llm::StreamEvent> {
    match event {
        ConverseStreamOutput::ContentBlockStart(block) => process_content_block_start_event(block),
        ConverseStreamOutput::ContentBlockDelta(block) => process_content_block_delta_event(block),
        ConverseStreamOutput::Metadata(metadata) => process_metadata_event(metadata),
        ConverseStreamOutput::MessageStop(event) => process_message_stop_event(event),
        _ => None,
    }
}

fn process_content_block_start_event(block: ContentBlockStartEvent) -> Option<llm::StreamEvent> {
    if let Some(start_info) = block.start {
        if let Ok(tool_use) = start_info.as_tool_use() {
            return Some(llm::StreamEvent::Delta(llm::StreamDelta {
                content: None,
                tool_calls: Some(vec![llm::ToolCall {
                    id: tool_use.tool_use_id.clone(),
                    name: tool_use.name.clone(),
                    arguments_json: "".to_owned(),
                }]),
            }));
        }
    }
    None
}

fn process_content_block_delta_event(block: ContentBlockDeltaEvent) -> Option<llm::StreamEvent> {
    if let Some(block_info) = block.delta {
        if let Ok(tool_use) = block_info.as_tool_use() {
            return Some(llm::StreamEvent::Delta(llm::StreamDelta {
                content: None,
                tool_calls: Some(vec![llm::ToolCall {
                    id: "".to_owned(),
                    name: "".to_owned(),
                    arguments_json: tool_use.input.clone(),
                }]),
            }));
        } else if let Ok(text) = block_info.as_text() {
            return Some(llm::StreamEvent::Delta(llm::StreamDelta {
                content: Some(vec![llm::ContentPart::Text(text.clone())]),
                tool_calls: None,
            }));
        }
    }
    None
}

fn process_metadata_event(metadata: ConverseStreamMetadataEvent) -> Option<llm::StreamEvent> {
    Some(llm::StreamEvent::Finish(llm::ResponseMetadata {
        finish_reason: None,
        timestamp: None,
        usage: metadata.usage().map(bedrock_usage_to_llm_usage),
        provider_id: Some("bedrock".to_owned()),
        provider_metadata_json: None,
    }))
}

fn process_message_stop_event(event: MessageStopEvent) -> Option<llm::StreamEvent> {
    Some(llm::StreamEvent::Finish(llm::ResponseMetadata {
        finish_reason: Some(bedrock_stop_reason_to_finish_reason(event.stop_reason())),
        timestamp: None,
        usage: None,
        provider_id: None,
        provider_metadata_json: event
            .additional_model_response_fields
            .clone()
            .and_then(smithy_document_to_metadata_json),
    }))
}

fn json_str_to_smithy_document(value: &str) -> Result<Document, llm::Error> {
    let json_value: serde_json::Value = serde_json::from_str(value).map_err(|err| llm::Error {
        code: llm::ErrorCode::InvalidRequest,
        message: format!("Invalid tool schema: {}", err),
        provider_error_json: None,
    })?;
    Ok(serde_json_to_smithy_document(json_value))
}

fn smithy_document_to_json_value(document: Document) -> serde_json::Value {
    match document {
        Document::Null => serde_json::Value::Null,
        Document::Bool(b) => serde_json::Value::Bool(b),
        Document::Number(num) => match num {
            Number::NegInt(i) => serde_json::Value::Number(serde_json::Number::from(i)),
            Number::PosInt(i) => serde_json::Value::Number(serde_json::Number::from(i)),
            Number::Float(f) => serde_json::Value::Number(serde_json::Number::from_f64(f).unwrap()),
        },
        Document::String(s) => serde_json::Value::String(s),
        Document::Array(arr) => {
            let mut items = vec![];
            for item in arr {
                items.push(smithy_document_to_json_value(item));
            }
            serde_json::Value::Array(items)
        }
        Document::Object(map) => {
            let mut object = serde_json::Map::new();
            for (key, value) in map {
                object.insert(key, smithy_document_to_json_value(value));
            }
            serde_json::Value::Object(object)
        }
    }
}

fn serde_json_to_smithy_document(value: serde_json::Value) -> Document {
    match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(b) => Document::Bool(b),
        serde_json::Value::Number(num) => {
            if num.is_i64() {
                Document::Number(Number::NegInt(num.as_i64().unwrap()))
            } else if num.is_u64() {
                Document::Number(Number::PosInt(num.as_u64().unwrap()))
            } else if num.is_f64() {
                Document::Number(Number::Float(num.as_f64().unwrap()))
            } else {
                // fallback to string if not any of above number types
                Document::String(num.to_string())
            }
        }
        serde_json::Value::String(s) => Document::String(s),
        serde_json::Value::Array(arr) => {
            let mut items = vec![];
            for item in arr {
                items.push(serde_json_to_smithy_document(item));
            }
            Document::Array(items)
        }
        serde_json::Value::Object(map) => {
            let mut object = HashMap::new();
            for (key, value) in map {
                object.insert(key, serde_json_to_smithy_document(value));
            }
            Document::Object(object)
        }
    }
}

pub fn from_converse_sdk_error(
    model_id: String,
    sdk_error: SdkError<converse::ConverseError>,
) -> llm::Error {
    llm::Error {
        code: llm::ErrorCode::InternalError,
        message: format!("Error calling Bedrock model {model_id}: {sdk_error:?}",),
        provider_error_json: None,
    }
}

pub fn from_converse_stream_sdk_error(
    model_id: String,
    sdk_error: SdkError<converse_stream::ConverseStreamError>,
) -> llm::Error {
    llm::Error {
        code: llm::ErrorCode::InternalError,
        message: format!("Error calling Bedrock model {model_id}: {sdk_error:?}",),
        provider_error_json: None,
    }
}

pub fn custom_error(code: llm::ErrorCode, message: String) -> llm::Error {
    llm::Error {
        code,
        message,
        provider_error_json: None,
    }
}

pub fn merge_metadata(
    mut metadata1: llm::ResponseMetadata,
    metadata2: llm::ResponseMetadata,
) -> llm::ResponseMetadata {
    metadata1.usage = metadata1.usage.or(metadata2.usage);
    metadata1.timestamp = metadata1.timestamp.or(metadata2.timestamp);
    metadata1.provider_id = metadata1.provider_id.or(metadata2.provider_id);
    metadata1.finish_reason = metadata1.finish_reason.or(metadata2.finish_reason);
    metadata1.provider_metadata_json = metadata1
        .provider_metadata_json
        .or(metadata2.provider_metadata_json);

    metadata1
}
