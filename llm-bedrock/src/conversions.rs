use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, ContentPart, Error, ErrorCode, FinishReason, Message,
    ResponseMetadata, Role, StreamDelta, StreamEvent, ToolCall, ToolDefinition, ToolResult, Usage, Config
};
use log::{warn, trace};
use serde_json::json;
use std::collections::HashMap;

// --- Request Conversion ---

// Helper to determine Bedrock model family from model_id string
fn get_bedrock_model_family(model_id: &str) -> &str {
    if model_id.starts_with("anthropic.claude") {
        "claude"
    } else if model_id.starts_with("ai21.j2") {
        "jurassic2"
    } else if model_id.starts_with("amazon.titan-text") {
        "titan-text"
    } else if model_id.starts_with("cohere.command") {
        "cohere-command"
    } else if model_id.starts_with("meta.llama2") || model_id.starts_with("meta.llama3") {
        "llama"
    }
    // Add more model families as needed
    else {
        warn!("Unknown Bedrock model family for model_id: {}. Defaulting to generic or expecting provider_options.", model_id);
        "unknown" // Or handle as an error / require explicit family in provider_options
    }
}

pub fn messages_to_bedrock_body(
    messages: &[Message],
    tool_results_opt: Option<&[(ToolCall, ToolResult)]>, // For continue_
    config: &Config,
) -> Result<serde_json::Value, Error> {
    let model_id = &config.model;
    let model_family = get_bedrock_model_family(model_id);

    // Provider options from config
    let provider_options: HashMap<String, String> = config.provider_options.iter().map(|kv| (kv.key.clone(), kv.value.clone())).collect();

    match model_family {
        "claude" => {
            // Based on Anthropic Claude on Bedrock format
            // https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
            let mut anthropic_messages = Vec::new();
            let mut system_prompts = Vec::new();

            for msg in messages {
                if msg.role == Role::System {
                    for part in &msg.content {
                        if let ContentPart::Text(text) = part {
                            system_prompts.push(text.clone());
                        } else {
                            return Err(Error{ code: ErrorCode::InvalidRequest, message: "Claude system prompts on Bedrock only support text".to_string(), provider_error_json: None });
                        }
                    }
                    continue;
                }

                let role_str = match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    // Tool role messages for Claude are handled by appending tool_results content
                    Role::Tool => "user", // This needs careful handling with tool_results
                    Role::System => unreachable!(), // Handled above
                };

                let mut content_parts = Vec::new();
                for part in &msg.content {
                    match part {
                        ContentPart::Text(text) => content_parts.push(json!({ "type": "text", "text": text })),
                        ContentPart::Image(image_url) => {
                            // Assuming image_url.url is base64 encoded data for Claude on Bedrock
                            // e.g. "data:image/jpeg;base64,..."
                            // Claude on Bedrock expects: { "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": "..."}}
                            let (media_type, b64_data) = parse_data_url(&image_url.url)?;
                            content_parts.push(json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_data
                                }
                            }));
                        }
                    }
                }
                anthropic_messages.push(json!({ "role": role_str, "content": content_parts }));
            }

            // Handle tool_results for continue_ case for Claude
            if let Some(tool_results) = tool_results_opt {
                 for (tool_call, tool_result) in tool_results {
                    // Assistant's turn that requested the tool
                    anthropic_messages.push(json!({
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": tool_call.name,
                                "input": serde_json::from_str::<serde_json::Value>(&tool_call.arguments_json).unwrap_or(json!({}))
                            }
                        ]
                    }));
                    // User's turn providing the tool result
                    let (result_content, is_error) = match tool_result {
                        ToolResult::Success(s) => (json!([{"type": "text", "text": s.result_json}]), false),
                        ToolResult::Error(e) => (json!([{"type": "text", "text": e.error_message}]), true),
                    };
                     anthropic_messages.push(json!({
                        "role": "user", // Or "tool" if Claude API differentiates. Docs say user.
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": result_content, // Claude on Bedrock v2.1 expects content to be a string for tool_result, or list of blocks
                                // For simplicity, passing JSON string as text.
                                // If it should be structured:
                                // "content": [{"type": "text", "text": if is_error { tool_result_error.error_message } else { tool_result_success.result_json }}],
                                "is_error": if is_error {Some(true)} else {None}
                            }
                        ]
                    }));
                }
            }


            let mut body = json!({
                "anthropic_version": provider_options.get("anthropic_version").map_or("bedrock-2023-05-31", |s| s.as_str()),
                "messages": anthropic_messages,
                "max_tokens": config.max_tokens.unwrap_or(1024), // Claude calls it max_tokens
            });

            let body_obj = body.as_object_mut().unwrap();
            if !system_prompts.is_empty() {
                body_obj.insert("system".to_string(), json!(system_prompts.join("\n")));
            }
            if let Some(temp) = config.temperature {
                body_obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(stop_sequences) = &config.stop_sequences {
                body_obj.insert("stop_sequences".to_string(), json!(stop_sequences));
            }
            if !config.tools.is_empty() {
                let bedrock_tools = tool_definitions_to_bedrock_tools(&config.tools, model_family)?;
                body_obj.insert("tools".to_string(), json!(bedrock_tools));
                if let Some(tool_choice_str) = &config.tool_choice {
                     body_obj.insert("tool_choice".to_string(), convert_tool_choice_to_bedrock(tool_choice_str, model_family));
                }
            }
             // top_p, top_k for Claude
            if let Some(top_p_str) = provider_options.get("top_p") {
                if let Ok(top_p) = top_p_str.parse::<f32>() { body_obj.insert("top_p".to_string(), json!(top_p));}
            }
            if let Some(top_k_str) = provider_options.get("top_k") {
                if let Ok(top_k) = top_k_str.parse::<u32>() { body_obj.insert("top_k".to_string(), json!(top_k));}
            }

            Ok(json!(body_obj))
        }
        "llama" => {
            // Llama on Bedrock: expects a single string prompt.
            // We need to construct this prompt from messages.
            // Example: "<s>[INST] User: ... [/INST] Assistant: ... </s><s>[INST] User: ... [/INST]"
            let mut prompt_string = String::new();
            if messages.len() == 1 && messages[0].role == Role::User && messages[0].content.len() == 1 {
                 if let ContentPart::Text(text) = &messages[0].content[0] {
                    prompt_string = text.clone(); // Simple case for playground-like single text prompt
                 } else {
                    return Err(Error { code: ErrorCode::InvalidRequest, message: "Llama on Bedrock with image requires specific prompt formatting not yet implemented simply.".to_string(), provider_error_json: None });
                 }
            } else {
                // More complex conversation history for Llama chat models
                for msg in messages {
                    let role_prefix = match msg.role {
                        Role::User => "User: ",
                        Role::Assistant => "Assistant: ",
                        Role::System => "", // System prompts are often prepended or handled differently
                        Role::Tool => "Tool output: ", // Llama might not directly support tool roles this way
                    };
                    prompt_string.push_str(role_prefix);
                    for part in &msg.content {
                        if let ContentPart::Text(text) = part {
                            prompt_string.push_str(text);
                            prompt_string.push('\n');
                        } else {
                             return Err(Error { code: ErrorCode::InvalidRequest, message: "Llama on Bedrock currently only supports text content in this simple conversion.".to_string(), provider_error_json: None });
                        }
                    }
                }
                 // Llama 3 specific format. Adjust if using Llama 2.
                 // This is a simplified formatter; a robust one is more complex.
                let mut formatted_prompt = "<|begin_of_text|>".to_string();
                for msg in messages {
                    let role_str = match msg.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::System => "system",
                        Role::Tool => "user", // Represent tool output as a user message or integrate differently
                    };
                    formatted_prompt.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n", role_str));
                    for part in &msg.content {
                        if let ContentPart::Text(text) = part {
                            formatted_prompt.push_str(text);
                        }
                        // Image handling for Llama needs specific model support & format
                    }
                    formatted_prompt.push_str("<|eot_id|>");
                }
                if messages.last().map_or(false, |m| m.role != Role::Assistant) {
                     formatted_prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                }
                prompt_string = formatted_prompt;
            }


            let mut body = json!({
                "prompt": prompt_string,
                "max_gen_len": config.max_tokens.unwrap_or(512),
            });
            let body_obj = body.as_object_mut().unwrap();
            if let Some(temp) = config.temperature {
                body_obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(top_p_str) = provider_options.get("top_p") { // Llama uses top_p
                if let Ok(top_p) = top_p_str.parse::<f32>() { body_obj.insert("top_p".to_string(), json!(top_p));}
            }
            // Llama does not support tools in the same way as Claude/OpenAI via Bedrock's direct invoke_model.
            // Tool use with Llama on Bedrock typically involves function calling implemented on the client-side or via agents.
            if !config.tools.is_empty() {
                warn!("Tools are configured but Llama on Bedrock (via simple invoke_model) may not support them directly in the API call body. Tool processing might need to be client-side.");
            }

            Ok(body)
        }
        // ... other model families ...
        _ => Err(Error {
            code: ErrorCode::InvalidRequest,
            message: format!("Unsupported or unknown Bedrock model family for request conversion: {}", model_family),
            provider_error_json: None,
        }),
    }
}

fn parse_data_url(url: &str) -> Result<(String, String), Error> {
    if !url.starts_with("data:") {
        return Err(Error { code: ErrorCode::InvalidRequest, message: "Image URL is not a data URL".to_string(), provider_error_json: None });
    }
    let parts: Vec<&str> = url.splitn(2, ',').collect();
    if parts.len() != 2 {
        return Err(Error { code: ErrorCode::InvalidRequest, message: "Invalid data URL format".to_string(), provider_error_json: None });
    }
    let header = parts[0]; // e.g., "data:image/jpeg;base64"
    let b64_data = parts[1].to_string();

    let mime_parts: Vec<&str> = header.trim_start_matches("data:").split(';').collect();
    let media_type = mime_parts.first().ok_or_else(|| Error { code: ErrorCode::InvalidRequest, message: "Missing media type in data URL".to_string(), provider_error_json: None })?.to_string();

    if !mime_parts.contains(&"base64") {
         return Err(Error { code: ErrorCode::InvalidRequest, message: "Data URL is not base64 encoded".to_string(), provider_error_json: None });
    }

    Ok((media_type, b64_data))
}


// --- Response Conversion ---
pub fn bedrock_response_to_chat_event(
    response_json: serde_json::Value,
    model_id: &str,
    base_metadata: ResponseMetadata, // provider_id, timestamp can be filled from SDK if available
) -> ChatEvent {
    trace!("Converting Bedrock response to ChatEvent. Model ID: {}, Response: {}", model_id, response_json);
    let model_family = get_bedrock_model_family(model_id);

    match model_family {
        "claude" => {
            // Example for Anthropic Claude on Bedrock
            // https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html#model-parameters-anthropic-claude-messages-response
            let id = response_json.get("id").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            // let role = response_json.get("role").and_then(|v| v.as_str()).unwrap_or("assistant"); // Should be assistant
            let stop_reason_str = response_json.get("stop_reason").and_then(|v| v.as_str());
            let stop_sequence = response_json.get("stop_sequence").and_then(|v| v.as_str()); // Can also be a stop reason

            let mut content_parts = Vec::new();
            let mut tool_call_parts: Vec<ToolCall> = Vec::new();

            if let Some(content_array) = response_json.get("content").and_then(|v| v.as_array()) {
                for item_val in content_array {
                    let item_type = item_val.get("type").and_then(|v| v.as_str());
                    match item_type {
                        Some("text") => {
                            if let Some(text) = item_val.get("text").and_then(|v| v.as_str()) {
                                content_parts.push(ContentPart::Text(text.to_string()));
                            }
                        }
                        Some("tool_use") => {
                            let tool_id = item_val.get("id").and_then(|v|v.as_str()).unwrap_or_default().to_string();
                            let tool_name = item_val.get("name").and_then(|v|v.as_str()).unwrap_or_default().to_string();
                            let tool_input = item_val.get("input").cloned().unwrap_or(json!({}));
                            tool_call_parts.push(ToolCall {
                                id: tool_id,
                                name: tool_name,
                                arguments_json: tool_input.to_string(),
                            });
                        }
                        _ => warn!("Unknown content type in Claude Bedrock response: {:?}", item_val),
                    }
                }
            }

            let usage_obj = response_json.get("usage");
            let usage = usage_obj.map(|u| Usage {
                input_tokens: u.get("input_tokens").and_then(|v| v.as_u64()).map(|v| v as u32),
                output_tokens: u.get("output_tokens").and_then(|v| v.as_u64()).map(|v| v as u32),
                total_tokens: None, // Bedrock Claude response doesn't provide total_tokens directly
            });

            let finish_reason = match stop_reason_str {
                Some("end_turn") => Some(FinishReason::Stop), // Or Other if more appropriate
                Some("tool_use") => Some(FinishReason::ToolCalls),
                Some("max_tokens") => Some(FinishReason::Length),
                Some("stop_sequence") => Some(FinishReason::Stop),
                _ => stop_sequence.map(|_| FinishReason::Stop), // If stopped by stop_sequence
            };
            
            let metadata = ResponseMetadata {
                finish_reason,
                usage,
                provider_id: Some(id.clone()), // Use response id as provider_id
                ..base_metadata
            };

            if !tool_call_parts.is_empty() {
                 // If there are tool_use blocks, it's a tool request.
                 // Claude might also return text alongside tool_use, the interface needs to decide.
                 // Assuming if tool_calls are present, it's primarily a ToolRequest.
                 // Any text can be ignored or handled based on spec for golem:llm interface.
                 // For now, prioritizing tool_calls if present.
                ChatEvent::ToolRequest(tool_call_parts)
            } else if !content_parts.is_empty() {
                ChatEvent::Message(CompleteResponse {
                    id,
                    content: content_parts,
                    tool_calls: Vec::new(), // No separate tool calls if primary content is text
                    metadata,
                })
            } else {
                 warn!("Bedrock Claude response has no text content or tool_calls: {}", response_json);
                 ChatEvent::Error(Error{ code: ErrorCode::InternalError, message: "Empty response content from Bedrock Claude".to_string(), provider_error_json: Some(response_json.to_string())})
            }
        }
        "llama" => {
            // Llama on Bedrock response: {"generation": "...", "prompt_token_count": ..., "generation_token_count": ..., "stop_reason": ...}
            let generation = response_json.get("generation").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let input_tokens = response_json.get("prompt_token_count").and_then(|v| v.as_u64()).map(|v| v as u32);
            let output_tokens = response_json.get("generation_token_count").and_then(|v| v.as_u64()).map(|v| v as u32);
            let stop_reason_str = response_json.get("stop_reason").and_then(|v| v.as_str());

            let usage = Some(Usage {
                input_tokens,
                output_tokens,
                total_tokens: None,
            });

            let finish_reason = match stop_reason_str {
                Some("stop") => Some(FinishReason::Stop),
                Some("length") => Some(FinishReason::Length),
                // Other Llama stop reasons: "content_filter"
                _ => None,
            };

            let metadata = ResponseMetadata {
                finish_reason,
                usage,
                ..base_metadata
            };
            ChatEvent::Message(CompleteResponse {
                id: String::new(), // Llama response doesn't have an 'id' field
                content: vec![ContentPart::Text(generation)],
                tool_calls: Vec::new(), // Llama invoke_model doesn't directly support tool calls in response
                metadata: ResponseMetadata {
                    finish_reason,
                    usage,
                    provider_id: None,
                    timestamp: None,
                    provider_metadata_json: None,
                },
            })
        }
        _ => ChatEvent::Error(Error {
            code: ErrorCode::InternalError,
            message: format!("Response conversion not implemented for Bedrock model family: {}", model_family),
            provider_error_json: Some(response_json.to_string()),
        }),
    }
}


// --- Streaming Conversion ---
pub fn bedrock_stream_chunk_to_stream_event(
    chunk_json_str: &str, // This is the raw string data from the mimicked SSE
    model_id: &str,
) -> Result<Option<StreamEvent>, String> {
    trace!("Converting Bedrock stream chunk. Model ID: {}, Chunk: {}", model_id, chunk_json_str);
    let model_family = get_bedrock_model_family(model_id);
    let chunk_json: serde_json::Value = serde_json::from_str(chunk_json_str)
        .map_err(|e| format!("Failed to parse stream chunk JSON: {}. Chunk: {}", e, chunk_json_str))?;


    match model_family {
        "claude" => {
            // Example for Anthropic Claude on Bedrock streaming
            // https://docs.aws.amazon.com/bedrock/latest/userguide/streaming-invoke-model.html (general)
            // https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html#model-parameters-anthropic-claude-messages-streaming
            let event_type = chunk_json.get("type").and_then(|t| t.as_str());
            match event_type {
                Some("message_start") => {
                    // Contains overall message metadata like role, id.
                    // Can be used to initialize response metadata if needed.
                    // let message = chunk_json.get("message").unwrap();
                    // let id = message.get("id").and_then(|v|v.as_str()).unwrap_or_default();
                    // store this id for the final StreamEvent::Finish
                    Ok(None) // No immediate delta from message_start itself typically
                }
                Some("content_block_start") => {
                    if let Some(content_block) = chunk_json.get("content_block") {
                        if content_block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                             let tool_id = content_block.get("id").and_then(|v|v.as_str()).unwrap_or_default().to_string();
                             let tool_name = content_block.get("name").and_then(|v|v.as_str()).unwrap_or_default().to_string();
                             // Placeholder for partial tool call, arguments will come in input_json_delta
                             return Ok(Some(StreamEvent::Delta(StreamDelta {
                                content: None,
                                tool_calls: Some(vec![ToolCall { id: tool_id, name: tool_name, arguments_json: "".to_string() }]) // Mark start of tool call
                             })));
                        }
                    }
                    Ok(None)
                }
                Some("content_block_delta") => {
                    if let Some(delta) = chunk_json.get("delta") {
                        let delta_type = delta.get("type").and_then(|t| t.as_str());
                        match delta_type {
                            Some("text_delta") => {
                                let text = delta.get("text").and_then(|t| t.as_str()).unwrap_or_default();
                                Ok(Some(StreamEvent::Delta(StreamDelta {
                                    content: Some(vec![ContentPart::Text(text.to_string())]),
                                    tool_calls: None,
                                })))
                            }
                            Some("input_json_delta") => {
                                // This contains partial JSON for a tool_use block's input.
                                // The LlmChatStream state needs to aggregate these.
                                // The current StreamDelta only sends complete tool_calls.
                                // We might need a new StreamEvent variant for partial tool_arg_delta,
                                // or aggregate here and send once content_block_stop for this tool_use is received.
                                // For now, let's assume aggregation happens in the LlmChatStreamState implementation for Bedrock.
                                // This function should ideally just convert the direct delta.
                                // To fit StreamDelta, we'd need to know which tool_call this belongs to.
                                // The chunk includes an "index" field for the content_block.
                                let content_block_index = chunk_json.get("index").and_then(|i|i.as_u64()).unwrap_or(0);
                                let partial_json = delta.get("partial_json").and_then(|t| t.as_str()).unwrap_or_default();
                                // This is a bit of a hack to fit into StreamDelta, signaling a partial argument update.
                                // The actual aggregation logic should be in BedrockChatStreamDirect.
                                Ok(Some(StreamEvent::Delta(StreamDelta {
                                    content: None,
                                    tool_calls: Some(vec![ToolCall {
                                        id: format!("tool_idx_{}", content_block_index), // Placeholder ID based on index
                                        name: "".to_string(), // Name would have come from content_block_start
                                        arguments_json: partial_json.to_string(), // This is the partial delta
                                    }]),
                                })))
                            }
                            _ => Ok(None)
                        }
                    } else { Ok(None) }
                }
                Some("content_block_stop") => {
                    // Signals a content block (e.g., a tool_use block) is complete.
                    // If aggregating input_json_delta, this is where the full ToolCall could be emitted.
                    // For now, no direct event from this if deltas are handled separately.
                    Ok(None)
                }
                Some("message_delta") => {
                    // Contains "usage" and "stop_reason", "stop_sequence"
                    // These contribute to the final ResponseMetadata.
                    // Not a content delta itself.
                    // Can extract usage and stop_reason here to build final metadata.
                    // Example:
                    // if let Some(usage_val) = chunk_json.get("usage") { ... }
                    // if let Some(delta_val) = chunk_json.get("delta") {
                    //    if let Some(stop_reason_str) = delta_val.get("stop_reason").and_then(|s|s.as_str()) { ... }
                    // }
                    Ok(None)
                }
                Some("message_stop") => {
                    // Final event in the stream for Claude. Contains final usage.
                    let usage_val = chunk_json.get("amazon-bedrock-invocationMetrics"); // For Claude 3 on Bedrock
                    let (input_tokens, output_tokens, latency_ms, first_byte_latency_ms) = if let Some(metrics) = usage_val {
                        (metrics.get("inputTokenCount").and_then(|v|v.as_u64()),
                         metrics.get("outputTokenCount").and_then(|v|v.as_u64()),
                         metrics.get("invocationLatency").and_then(|v|v.as_u64()),
                         metrics.get("firstByteLatency").and_then(|v|v.as_u64()))
                    } else { (None, None, None, None) };


                    // The actual stop_reason should have been collected from message_delta.
                    // This event mainly confirms the end and provides final metrics.
                    let final_metadata = ResponseMetadata {
                        finish_reason: None, // Should be populated from earlier message_delta
                        usage: Some(Usage {
                            input_tokens: input_tokens.map(|v| v as u32),
                            output_tokens: output_tokens.map(|v| v as u32),
                            total_tokens: None,
                        }),
                        provider_id: None, // Should be populated from message_start
                        timestamp: None,
                        provider_metadata_json: Some(json!({
                            "invocationLatency": latency_ms,
                            "firstByteLatency": first_byte_latency_ms
                        }).to_string()),
                    };
                    Ok(Some(StreamEvent::Finish(final_metadata)))
                }
                Some("internal_failure") | Some("error") => {
                    let error_message = chunk_json.get("message").and_then(|m| m.as_str()).unwrap_or("Unknown streaming error").to_string();
                    Err(error_message)
                }
                _ => {
                    warn!("Unknown Bedrock Claude stream event type: {:?}, Chunk: {}", event_type, chunk_json_str);
                    Ok(None)
                }
            }
        }
        "llama" => {
             // Llama on Bedrock streaming response: {"generation": "...", "prompt_token_count": ..., "generation_token_count": ..., "stop_reason": ..., "amazon-bedrock-invocationMetrics": {...}}
             // Each chunk is a JSON object. The "generation" field contains the text delta.
             // The last chunk will have the stop_reason and metrics.
            if let Some(generation_delta) = chunk_json.get("generation").and_then(|g| g.as_str()) {
                // This is a content delta
                Ok(Some(StreamEvent::Delta(StreamDelta {
                    content: Some(vec![ContentPart::Text(generation_delta.to_string())]),
                    tool_calls: None,
                })))
            } else if chunk_json.get("stop_reason").is_some() {
                // This is likely the final chunk with metadata
                let input_tokens = chunk_json.get("prompt_token_count").and_then(|v| v.as_u64()).map(|v| v as u32);
                let output_tokens = chunk_json.get("generation_token_count").and_then(|v| v.as_u64()).map(|v| v as u32);
                let stop_reason_str = chunk_json.get("stop_reason").and_then(|v| v.as_str());
                let metrics = chunk_json.get("amazon-bedrock-invocationMetrics");

                let finish_reason = match stop_reason_str {
                    Some("stop") => Some(FinishReason::Stop),
                    Some("length") => Some(FinishReason::Length),
                    _ => None,
                };
                let usage = Some(Usage { input_tokens, output_tokens, total_tokens: None });
                
                let metadata_json = metrics.map(|m| m.to_string());

                Ok(Some(StreamEvent::Finish(ResponseMetadata {
                    finish_reason,
                    usage,
                    provider_id: None, // Llama stream doesn't provide an ID per message
                    timestamp: None,
                    provider_metadata_json: metadata_json,
                })))
            } else if chunk_json.get("InternalServerException").is_some() || chunk_json.get("ModelStreamErrorException").is_some() {
                Err(chunk_json.to_string())
            }
            else {
                warn!("Unknown Bedrock Llama stream chunk: {}", chunk_json_str);
                Ok(None)
            }
        }
        _ => Err(format!("Streaming conversion not implemented for Bedrock model family: {}", model_family)),
    }
}

fn tool_definitions_to_bedrock_tools(tools: &[ToolDefinition], model_family: &str) -> Result<serde_json::Value, Error> {
    match model_family {
        "claude" => {
            let mut claude_tools = Vec::new();
            for tool_def in tools {
                let params_schema: serde_json::Value = serde_json::from_str(&tool_def.parameters_schema)
                    .map_err(|e| Error {
                        code: ErrorCode::InvalidRequest,
                        message: format!("Invalid JSON schema for tool {}: {}", tool_def.name, e),
                        provider_error_json: Some(tool_def.parameters_schema.clone()),
                    })?;
                claude_tools.push(json!({
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "input_schema": params_schema
                }));
            }
            Ok(json!(claude_tools))
        }
        // Other model families might have different tool formats or no direct support
        _ => {
            if !tools.is_empty() {
                warn!("Tool definitions provided for model family '{}' which may not support them in this format.", model_family);
            }
            Ok(json!([])) // Empty array if not supported or unknown
        }
    }
}

fn convert_tool_choice_to_bedrock(tool_choice_str: &str, model_family: &str) -> serde_json::Value {
    match model_family {
        "claude" => {
            // Claude tool choice: {"type": "auto" | "any" | "tool", "name": "tool_name_if_type_is_tool"}
            match tool_choice_str {
                "auto" | "" => json!({"type": "auto"}), // Default to auto
                "any" => json!({"type": "any"}),
                "none" => json!({"type": "auto"}),
                // If it's a specific tool name:
                tool_name => json!({"type": "tool", "name": tool_name}),
            }
        }
        _ => json!(null) // Or adapt for other families
    }
} 