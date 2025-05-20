use golem_llm::error::{error_code_from_status, from_event_source_error, from_reqwest_error};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::Error;
use log::trace;
use reqwest::header::HeaderValue;
use reqwest::{Client, Method, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// The Ollama API client for creating model responses.
pub struct OllamaApi {
    base_url: String,
    client: Client,
}

impl OllamaApi {
    pub fn new() -> Self {
        let base_url =
            std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        Self::with_base_url(base_url)
    }

    pub fn with_base_url(base_url: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self { base_url, client }
    }

    pub fn image_url_to_base64(&self, url: &str) -> Result<String, Error> {
        use base64::engine::general_purpose;
        use base64::Engine;

        let response = self
            .client
            .get(url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .map_err(|err| from_reqwest_error("Failed to download image", err))?;

        let status = response.status();
        if !status.is_success() {
            return Err(Error {
                code: error_code_from_status(status),
                message: format!("Failed to fetch image: {}", status),
                provider_error_json: None,
            });
        }

        let mime_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("image/png")
            .to_string();
        let bytes = response
            .bytes()
            .map_err(|err| from_reqwest_error("Failed to read image bytes", err))?;

        let encoded = general_purpose::STANDARD.encode(&bytes);

        Ok(format!("data:{};base64,{}", mime_type, encoded))
    }

    pub fn send_messages(&self, request: OllamaChatRequest) -> Result<OllamaChatResponse, Error> {
        trace!("Sending chat request to Ollama API: {request:?}");

        let mut stream_request = request;
        stream_request.stream = false;

        let response: Response = self
            .client
            .request(
                Method::POST,
                format!("{}/v1/chat/completions", self.base_url),
            )
            .json(&stream_request)
            .send()
            .map_err(|err| {
                log::error!("Failed to send HTTP request to Ollama: {err:?}");
                from_reqwest_error("Request failed", err)
            })?;

        parse_response(response)
    }

    pub fn stream_send_messages(&self, request: OllamaChatRequest) -> Result<EventSource, Error> {
        trace!("Sending streaming chat request to Ollama API: {request:?}");
        let mut stream_request = request;
        stream_request.stream = true;

        let response: Response = self
            .client
            .request(
                Method::POST,
                format!("{}/v1/chat/completions", self.base_url),
            )
            .header(
                reqwest::header::ACCEPT,
                HeaderValue::from_static("text/event-stream"),
            )
            .json(&stream_request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        trace!("Initializing SSE stream");

        EventSource::new(response)
            .map_err(|err| from_event_source_error("Failed to create SSE stream", err))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OllamaTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    #[serde(flatten)]
    pub content: OllamaMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContentPayload {
    Text { content: String },
    Array { content: Vec<ContentPart> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTool {
    #[serde(rename = "type")]
    pub typ: String,
    pub function: OllamaFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCall {
    pub id: String,
    pub function: OllamaToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCallFunction {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String),
    Object {
        #[serde(rename = "type")]
        typ: String,
        function: OllamaFunctionChoice,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunctionChoice {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaErrorResponse {
    pub error: OllamaError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaError {
    pub message: String,
    #[serde(rename = "type")]
    pub typ: Option<String>,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OllamaChoice>,
    pub usage: Option<OllamaUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChoice {
    pub index: u32,
    pub message: OllamaMessageContent,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessageContent {
    pub role: String,
    #[serde(flatten)]
    pub content: Option<MessageContentPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatDeltaResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OllamaDeltaChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaDeltaChoice {
    pub index: u32,
    pub delta: OllamaDeltaMessageContent,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaDeltaMessageContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();
    if status.is_success() {
        let body = response
            .json::<T>()
            .map_err(|err| from_reqwest_error("Failed to decode response body", err))?;

        trace!("Received response from Ollama API: {body:?}");

        Ok(body)
    } else {
        let error_body = response
            .json::<OllamaErrorResponse>()
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

        trace!("Received {status} response from Ollama API: {error_body:?}");

        Err(Error {
            code: error_code_from_status(status),
            message: format!("Request failed with {status}: {}", error_body.error.message),
            provider_error_json: Some(serde_json::to_string(&error_body).unwrap()),
        })
    }
}
