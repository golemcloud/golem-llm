use golem_llm::error::{error_code_from_status, from_event_source_error, from_reqwest_error};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::{Error, ErrorCode};
use log::trace;
use reqwest::{Client, Method, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;
use std::collections::HashMap;
use chrono::Utc;
use hmac::{Hmac, Mac};
use sha2::{Sha256, Digest};

type HmacSha256 = Hmac<Sha256>;

/// AWS Bedrock client for creating model responses
pub struct BedrockClient {
    access_key_id: String,
    secret_access_key: String,
    region: String,
    client: Client,
}

impl BedrockClient {
    pub fn new(access_key_id: String, secret_access_key: String, region: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self {
            access_key_id,
            secret_access_key,
            region,
            client,
        }
    }

    pub fn converse(&self, model_id: &str, request: ConverseRequest) -> Result<ConverseResponse, Error> {
        trace!("Sending request to Bedrock API: {request:?}");

        let body = serde_json::to_string(&request)
            .map_err(|err| Error {
                code: ErrorCode::InvalidRequest,
                message: format!("Failed to serialize request: {err}"),
                provider_error_json: None,
            })?;

        let headers = self.sign_request(&body, model_id, false)?;
        let url = format!("https://bedrock-runtime.{}.amazonaws.com/model/{}/converse", self.region, model_id);

        let mut request_builder = self.client.request(Method::POST, url);
        for (key, value) in headers {
            request_builder = request_builder.header(key, value);
        }

        let response: Response = request_builder
            .body(body)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn converse_stream(&self, model_id: &str, request: ConverseRequest) -> Result<EventSource, Error> {
        trace!("Sending streaming request to Bedrock API: {request:?}");

        let body = serde_json::to_string(&request)
            .map_err(|err| Error {
                code: ErrorCode::InvalidRequest,
                message: format!("Failed to serialize request: {err}"),
                provider_error_json: None,
            })?;

        let headers = self.sign_request(&body, model_id, true)?;
        let url = format!("https://bedrock-runtime.{}.amazonaws.com/model/{}/converse-stream", self.region, model_id);

        let mut request_builder = self.client.request(Method::POST, url);
        for (key, value) in headers {
            request_builder = request_builder.header(key, value);
        }

        let response: Response = request_builder
            .body(body)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        trace!("Initializing SSE stream");

        EventSource::new(response)
            .map_err(|err| from_event_source_error("Failed to create SSE stream", err))
    }

    fn sign_request(&self, body: &str, model_id: &str, is_stream: bool) -> Result<HashMap<String, String>, Error> {
        let now = Utc::now();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = now.format("%Y%m%d").to_string();
        
        let service = "bedrock";
        let endpoint = if is_stream { "converse-stream" } else { "converse" };
        let canonical_uri = format!("/model/{}/{}", model_id, endpoint);
        
        let canonical_headers = format!(
            "host:bedrock-runtime.{}.amazonaws.com\nx-amz-date:{}\n",
            self.region, amz_date
        );
        let signed_headers = "host;x-amz-date";
        
        let payload_hash = hex::encode(Sha256::digest(body.as_bytes()));
        
        let canonical_request = format!(
            "POST\n{}\n\n{}\n{}\n{}",
            canonical_uri, canonical_headers, signed_headers, payload_hash
        );
        
        let algorithm = "AWS4-HMAC-SHA256";
        let credential_scope = format!("{}/{}/{}/aws4_request", date_stamp, self.region, service);
        let string_to_sign = format!(
            "{}\n{}\n{}\n{}",
            algorithm,
            amz_date,
            credential_scope,
            hex::encode(Sha256::digest(canonical_request.as_bytes()))
        );
        
        let signing_key = self.get_signature_key(&date_stamp, service)?;
        let signature = hex::encode(
            HmacSha256::new_from_slice(&signing_key)
                .map_err(|_| Error {
                    code: ErrorCode::InternalError,
                    message: "Failed to create HMAC".to_string(),
                    provider_error_json: None,
                })?
                .chain_update(string_to_sign.as_bytes())
                .finalize()
                .into_bytes()
        );
        
        let authorization_header = format!(
            "{} Credential={}/{}, SignedHeaders={}, Signature={}",
            algorithm, self.access_key_id, credential_scope, signed_headers, signature
        );
        
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), authorization_header);
        headers.insert("X-Amz-Date".to_string(), amz_date);
        headers.insert("Host".to_string(), format!("bedrock-runtime.{}.amazonaws.com", self.region));
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        
        Ok(headers)
    }
    
    fn get_signature_key(&self, date_stamp: &str, service: &str) -> Result<Vec<u8>, Error> {
        let k_date = HmacSha256::new_from_slice(format!("AWS4{}", self.secret_access_key).as_bytes())
            .map_err(|_| Error {
                code: ErrorCode::InternalError,
                message: "Failed to create HMAC for date".to_string(),
                provider_error_json: None,
            })?
            .chain_update(date_stamp.as_bytes())
            .finalize()
            .into_bytes();
            
        let k_region = HmacSha256::new_from_slice(&k_date)
            .map_err(|_| Error {
                code: ErrorCode::InternalError,
                message: "Failed to create HMAC for region".to_string(),
                provider_error_json: None,
            })?
            .chain_update(self.region.as_bytes())
            .finalize()
            .into_bytes();
            
        let k_service = HmacSha256::new_from_slice(&k_region)
            .map_err(|_| Error {
                code: ErrorCode::InternalError,
                message: "Failed to create HMAC for service".to_string(),
                provider_error_json: None,
            })?
            .chain_update(service.as_bytes())
            .finalize()
            .into_bytes();
            
        let k_signing = HmacSha256::new_from_slice(&k_service)
            .map_err(|_| Error {
                code: ErrorCode::InternalError,
                message: "Failed to create HMAC for signing".to_string(),
                provider_error_json: None,
            })?
            .chain_update(b"aws4_request")
            .finalize()
            .into_bytes();
            
        Ok(k_signing.to_vec())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConverseRequest {
    #[serde(rename = "modelId")]
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
    #[serde(rename = "additionalModelRequestFields", skip_serializing_if = "Option::is_none")]
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
        let error_body = response
            .json::<ErrorResponse>()
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

        trace!("Received {status} response from Bedrock API: {error_body:?}");

        Err(Error {
            code: error_code_from_status(status),
            message: format!("Request failed with {status}: {}", error_body.message),
            provider_error_json: Some(serde_json::to_string(&error_body).unwrap()),
        })
    }
}