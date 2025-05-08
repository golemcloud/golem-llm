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
///
/// Based on https://github.com/ollama/ollama/blob/main/docs/api.md
pub struct OllamaApi {
    base_url: String,
    client: Client,
}

impl OllamaApi {
    pub fn new() -> Self {
        Self::with_base_url(DEFAULT_BASE_URL.to_string())
    }

    pub fn with_base_url(base_url: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self { base_url, client }
    }

    pub fn send_messages(&self, request: OllamaChatRequest) -> Result<OllamaChatResponse, Error> {
        trace!("Sending chat request to Ollama API: {request:?}");

        let mut stream_request = request;
        stream_request.stream = false;

        let response: Response = self
            .client
            .request(Method::POST, format!("{}/api/chat", self.base_url))
            .json(&stream_request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn stream_send_messages(&self, request: OllamaChatRequest) -> Result<EventSource, Error> {
        trace!("Sending streaming chat request to Ollama API: {request:?}");
        let mut stream_request = request;
        stream_request.stream = true;

        let response: Response = self
            .client
            .request(Method::POST, format!("{}/api/chat", self.base_url))
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
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<RequestOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
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
pub enum Role {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "system")]
    System,
    #[serde(rename = "tool")]
    Tool,
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
pub struct RequestOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_eta: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tfs_z: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub penalize_newline: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaErrorResponse {
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: OllamaMessageContent,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessageContent {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
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
            message: format!("Request failed with {status}: {}", error_body.error),
            provider_error_json: Some(serde_json::to_string(&error_body).unwrap()),
        })
    }
}
