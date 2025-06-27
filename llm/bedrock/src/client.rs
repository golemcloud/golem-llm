use golem_llm::error::{error_code_from_status, from_event_source_error, from_reqwest_error};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::{Error, ErrorCode};
use hmac::{Hmac, Mac};
use log::trace;
use reqwest::{Client, Method, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt::Debug;
use std::time::{SystemTime, UNIX_EPOCH};
use time::OffsetDateTime;

/// AWS Bedrock client for creating model responses
pub struct BedrockClient {
    access_key_id: String,
    secret_access_key: String,
    region: String,
    client: Client,
}

impl BedrockClient {
    pub fn new(access_key_id: String, secret_access_key: String, region: String) -> Self {
        let client = Client::new();
        Self {
            access_key_id,
            secret_access_key,
            region,
            client,
        }
    }

    pub fn converse(
        &self,
        model_id: &str,
        request: ConverseRequest,
    ) -> Result<ConverseResponse, Error> {
        trace!("Sending request to Bedrock API: {request:?}");
        let url = format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/converse",
            self.region, model_id
        );

        let body = serde_json::to_string(&request).map_err(|err| Error {
            code: ErrorCode::InternalError,
            message: "Failed to serialize request".to_string(),
            provider_error_json: Some(err.to_string()),
        })?;

        let host = format!("bedrock-runtime.{}.amazonaws.com", self.region);
        let headers = generate_sigv4_headers(
            &self.access_key_id,
            &self.secret_access_key,
            &self.region,
            "bedrock",
            "POST",
            &format!("/model/{}/converse", model_id),
            &host,
            &body,
        )
        .map_err(|err| Error {
            code: ErrorCode::InternalError,
            message: "Failed to sign headers".to_string(),
            provider_error_json: Some(err.to_string()),
        })?;

        let mut request_builder = self.client.request(Method::POST, &url);
        request_builder = request_builder.header("content-type", "application/json");
        for (key, value) in headers {
            request_builder = request_builder.header(key, value);
        }

        let response: Response = request_builder.body(body).send().map_err(|err| {
            trace!("HTTP request failed with error: {:?}", err);
            from_reqwest_error("Request failed", err)
        })?;

        trace!("Received response from Bedrock API: {:?}", response);

        parse_response(response)
    }

    pub fn converse_stream(
        &self,
        model_id: &str,
        request: ConverseRequest,
    ) -> Result<EventSource, Error> {
        trace!("Sending streaming request to Bedrock API: {request:?}");
        let url = format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/converse-stream",
            self.region, model_id
        );

        let body = serde_json::to_string(&request).map_err(|err| Error {
            code: ErrorCode::InternalError,
            message: "Failed to serialize request".to_string(),
            provider_error_json: Some(err.to_string()),
        })?;

        let host = format!("bedrock-runtime.{}.amazonaws.com", self.region);
        let headers = generate_sigv4_headers(
            &self.access_key_id,
            &self.secret_access_key,
            &self.region,
            "bedrock",
            "POST",
            &format!("/model/{}/converse-stream", model_id),
            &host,
            &body,
        )
        .map_err(|err| Error {
            code: ErrorCode::InternalError,
            message: "Failed to sign headers".to_string(),
            provider_error_json: Some(err.to_string()),
        })?;

        let mut request_builder = self.client.request(Method::POST, &url);
        request_builder = request_builder.header("content-type", "application/json");
        for (key, value) in headers {
            request_builder = request_builder.header(key, value);
        }

        trace!("Sending streaming HTTP request to Bedrock...");
        let response: Response = request_builder.body(body).send().map_err(|err| {
            trace!("HTTP request failed with error: {:?}", err);
            from_reqwest_error("Request failed", err)
        })?;

        trace!("Initializing SSE stream");

        EventSource::new(response)
            .map_err(|err| from_event_source_error("Failed to create SSE stream", err))
    }
}

pub fn generate_sigv4_headers(
    access_key: &str,
    secret_key: &str,
    region: &str,
    service: &str,
    method: &str,
    uri: &str,
    host: &str,
    body: &str,
) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let timestamp = OffsetDateTime::from_unix_timestamp(now.as_secs() as i64).unwrap();

    let date_str = format!(
        "{:04}{:02}{:02}",
        timestamp.year(),
        timestamp.month() as u8,
        timestamp.day()
    );
    let datetime_str = format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        timestamp.year(),
        timestamp.month() as u8,
        timestamp.day(),
        timestamp.hour(),
        timestamp.minute(),
        timestamp.second()
    );

    // Create canonical request
    let path = if uri.starts_with('/') { uri } else { "/" };
    let query = "";

    // Create canonical headers
    let mut headers: Vec<(String, String)> = vec![
        ("host".to_string(), host.to_string()),
        ("x-amz-date".to_string(), datetime_str.clone()),
    ];
    headers.sort_by(|a, b| a.0.cmp(&b.0));

    let canonical_headers = headers
        .iter()
        .map(|(k, v)| format!("{}:{}", k, v))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";

    let signed_headers = headers
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");

    // Hash payload
    let payload_hash = format!("{:x}", Sha256::digest(body.as_bytes()));

    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method, path, query, canonical_headers, signed_headers, payload_hash
    );

    // Create string to sign
    let credential_scope = format!("{}/{}/{}/aws4_request", date_str, region, service);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{:x}",
        datetime_str,
        credential_scope,
        Sha256::digest(canonical_request.as_bytes())
    );

    // Calculate signature
    type HmacSha256 = Hmac<Sha256>;

    let mut mac = HmacSha256::new_from_slice(format!("AWS4{}", secret_key).as_bytes())?;
    mac.update(date_str.as_bytes());
    let date_key = mac.finalize().into_bytes();

    let mut mac = HmacSha256::new_from_slice(&date_key)?;
    mac.update(region.as_bytes());
    let region_key = mac.finalize().into_bytes();

    let mut mac = HmacSha256::new_from_slice(&region_key)?;
    mac.update(service.as_bytes());
    let service_key = mac.finalize().into_bytes();

    let mut mac = HmacSha256::new_from_slice(&service_key)?;
    mac.update(b"aws4_request");
    let signing_key = mac.finalize().into_bytes();

    let mut mac = HmacSha256::new_from_slice(&signing_key)?;
    mac.update(string_to_sign.as_bytes());
    let signature = format!("{:x}", mac.finalize().into_bytes());

    // Create authorization header
    let auth_header = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        access_key, credential_scope, signed_headers, signature
    );

    let mut result_headers = vec![
        ("authorization".to_string(), auth_header),
        ("x-amz-date".to_string(), datetime_str),
    ];

    Ok(result_headers)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConverseRequest {
    #[serde(skip_serializing, rename = "modelId")]
    pub model_id: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemContentBlock>>,
    #[serde(rename = "inferenceConfig", skip_serializing_if = "Option::is_none")]
    pub inference_config: Option<InferenceConfig>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(rename = "guardrailConfig", skip_serializing_if = "Option::is_none")]
    pub guardrail_config: Option<GuardrailConfig>,
    #[serde(
        rename = "additionalModelRequestFields",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_model_request_fields: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        #[serde(rename = "format")]
        format: ImageFormat,
        #[serde(rename = "source")]
        source: ImageSource,
    },
    #[serde(rename = "toolUse")]
    ToolUse {
        #[serde(rename = "toolUseId")]
        tool_use_id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "toolResult")]
    ToolResult {
        #[serde(rename = "toolUseId")]
        tool_use_id: String,
        content: Vec<ToolResultContentBlock>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<ToolResultStatus>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageFormat {
    #[serde(rename = "png")]
    Png,
    #[serde(rename = "jpeg")]
    Jpeg,
    #[serde(rename = "gif")]
    Gif,
    #[serde(rename = "webp")]
    Webp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "bytes")]
pub struct ImageSource {
    pub bytes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolResultContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        #[serde(rename = "format")]
        format: ImageFormat,
        #[serde(rename = "source")]
        source: ImageSource,
    },
    #[serde(rename = "json")]
    Json { json: Value },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolResultStatus {
    #[serde(rename = "success")]
    Success,
    #[serde(rename = "error")]
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SystemContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    #[serde(rename = "maxTokens", skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    pub tools: Vec<Tool>,
    #[serde(rename = "toolChoice", skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Tool {
    #[serde(rename = "toolSpec")]
    ToolSpec {
        name: String,
        description: String,
        #[serde(rename = "inputSchema")]
        input_schema: ToolInputSchema,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInputSchema {
    pub json: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailConfig {
    #[serde(rename = "guardrailIdentifier")]
    pub guardrail_identifier: String,
    #[serde(rename = "guardrailVersion")]
    pub guardrail_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<GuardrailTrace>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuardrailTrace {
    #[serde(rename = "enabled")]
    Enabled,
    #[serde(rename = "disabled")]
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConverseResponse {
    #[serde(rename = "responseMetadata")]
    pub response_metadata: ResponseMetadata,
    pub output: Output,
    #[serde(rename = "stopReason")]
    pub stop_reason: StopReason,
    pub usage: Usage,
    pub metrics: Metrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    #[serde(rename = "requestId")]
    pub request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Output {
    pub message: Message,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StopReason {
    #[serde(rename = "end_turn")]
    EndTurn,
    #[serde(rename = "tool_use")]
    ToolUse,
    #[serde(rename = "max_tokens")]
    MaxTokens,
    #[serde(rename = "stop_sequence")]
    StopSequence,
    #[serde(rename = "guardrail_intervened")]
    GuardrailIntervened,
    #[serde(rename = "content_filtered")]
    ContentFiltered,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    #[serde(rename = "inputTokens")]
    pub input_tokens: u32,
    #[serde(rename = "outputTokens")]
    pub output_tokens: u32,
    #[serde(rename = "totalTokens")]
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    #[serde(rename = "latencyMs")]
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
}

fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();
    if status.is_success() {
        let body = response
            .json::<T>()
            .map_err(|err| from_reqwest_error("Failed to decode response body", err))?;

        trace!("Received response from Bedrock API: {body:?}");

        Ok(body)
    } else {
        let body = response
            .text()
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;
        trace!("Received {status} response from Bedrock API: {body:?}");

        Err(Error {
            code: error_code_from_status(status),
            message: format!("Request failed with {status}: {}", body),
            provider_error_json: Some(body),
        })
    }
}

        
