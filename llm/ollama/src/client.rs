use std::{fmt::Debug, fs, path::Path};

use base64::{engine::general_purpose, Engine};
use golem_llm::{
    error::{error_code_from_status, from_event_source_error},
    event_source::EventSource,
    golem::llm::llm::{Error, ErrorCode},
};
use log::trace;
use reqwest::{
    header::{HeaderMap, HeaderValue, CONTENT_TYPE},
    Client, Method, Response, StatusCode,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use url::Url;

pub struct OllamaApi {
    default_model: String,
    base_url: String,
    client: Client,
}

impl OllamaApi {
    pub fn new(default_model: String) -> Self {
        let base_url =
            std::env::var("GOLEM_OLLAMA_BASE_URL").unwrap_or("http://localhost:11434".to_string());
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self {
            default_model,
            base_url,
            client,
        }
    }

    pub fn send_chat(&self, params: CompletionsRequest) -> Result<CompletionsResponse, Error> {
        trace!("Sending request to Ollama API: {params:?}");

        let mut modified_params = params;
        modified_params.stream = Some(false);
        if modified_params.model.is_none() {
            modified_params.model = Some(self.default_model.clone())
        };

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let url = format!("{}/api/chat", self.base_url);
        let response: Response = self
            .client
            .request(Method::POST, url)
            .headers(headers)
            .json(&modified_params)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        handle_response::<CompletionsResponse>(response)
    }

    pub fn send_chat_stream(&self, params: CompletionsRequest) -> Result<EventSource, Error> {
        trace!("Sending request to Ollama API: {params:?}");

        let mut modified_params = params;
        modified_params.stream = Some(true);
        if modified_params.model.is_none() {
            modified_params.model = Some(self.default_model.clone())
        };

        let json_body = serde_json::to_string(&modified_params).map_err(|e| Error {
            code: ErrorCode::InternalError,
            message: format!("Failed to serialize request body: {e}"),
            provider_error_json: None,
        })?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert("Accept", HeaderValue::from_static("application/x-ndjson"));

        let url = format!("{}/api/chat", self.base_url);
        let response = self
            .client
            .request(Method::POST, url)
            .headers(headers)
            .body(json_body)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;
        EventSource::new(response)
            .map_err(|err| from_event_source_error("Failed to create EventSource stream", err))
    }
}

/// GenerateOptions is Options for generating completions
///
/// Refer to https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values for more details
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaModelOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_keep: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_eta: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub penalize_newline: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numa: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_batch: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_gpu: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_mmap: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<i32>,
}

/// ChatRequest is parameters for a request to the chat endpoint
///
/// Refer to https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion for more details
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompletionsRequest {
    /// If NONE then the default model will be used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<MessageRequest>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaModelOptions>,

    /// If false the response will be returned as a single response object, rather than a stream of objects.
    /// For `chat_completion` this will be set to false.
    /// For `chat_completion_stream` this will be set to true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Format {
    #[serde(rename = "type")]
    pub format_type: String,
    pub properties: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MessageRequest {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_calls: Option<Vec<Tool>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionTool,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionTool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompletionsResponse {
    pub model: String,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<MessageResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum MessageRole {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "system")]
    System,
    #[serde(rename = "tool")]
    Tool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MessageResponse {
    pub role: MessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<Function>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OllamaRequestError {
    status_code: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_message: Option<String>,
}

pub fn handle_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();

    match status {
        StatusCode::OK => {
            let raw_body = response
                .text()
                .map_err(|err| from_reqwest_error("Failed to receive response body", err))?;

            match serde_json::from_str::<T>(&raw_body) {
                Ok(body) => Ok(body),
                Err(err) => Err(Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to parse response body: {err}"),
                    provider_error_json: Some(raw_body),
                }),
            }
        }
        _ => {
            let raw_error_body = response
                .text()
                .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;
            trace!("Received {status} response from OpenRouter API: {raw_error_body:?}");

            let error_body: OllamaRequestError =
                serde_json::from_str(&raw_error_body).map_err(|err| Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to parse error response body: {err}"),
                    provider_error_json: Some(raw_error_body),
                })?;

            Err(Error {
                code: error_code_from_status(status),
                message: error_body.status.unwrap_or_default(),
                provider_error_json: error_body.error_message,
            })
        }
    }
}

pub fn image_to_base64(source: &str) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = if Url::parse(source).is_ok() {
        let client = Client::new();
        let response = client.get(source).send()?;

        response.bytes()?.to_vec()
    } else {
        let path = Path::new(source);

        fs::read(path)?
    };

    let base64_data = general_purpose::STANDARD.encode(&bytes);
    Ok(base64_data)
}

pub fn from_reqwest_error(context: &str, err: reqwest::Error) -> Error {
    Error {
        code: ErrorCode::InternalError,
        message: format!("{context}: {err}"),
        provider_error_json: None,
    }
}
