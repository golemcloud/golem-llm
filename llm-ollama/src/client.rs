use std::fmt::Debug;

use golem_llm::{
    error::{error_code_from_status, from_event_source_error, from_reqwest_error},
    event_source::{error, EventSource},
    golem::llm::llm::{Error, ErrorCode},
};
use log::trace;
use reqwest::{
    header::{HeaderMap, HeaderValue, CONTENT_TYPE},
    Client, Method, Response, StatusCode,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

struct OllamaApi {
    default_model: String,
    base_url: String,
    client: Client,
}

impl OllamaApi {
    pub fn new(default_model: String, base_url: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self {
            default_model,
            base_url,
            client,
        }
    }

    pub fn generate_completion(&self, params: &GenerateParams) -> Result<GenerateResponse, Error> {
        trace!("Sending request to Ollama API: {params:?}");

        let mut modified_params = params.clone();
        modified_params.stream = Some(false);
        if modified_params.model.is_none() {
            modified_params.model = Some(self.default_model.clone())
        };

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let url = format!("{}/api/generate", self.base_url);
        let response: Response = self
            .client
            .request(Method::POST, url)
            .headers(headers)
            .json(&modified_params)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        handle_response::<GenerateResponse>(response)
    }

    pub fn generate_completion_stream<F>(
        &self,
        params: GenerateParams,
    ) -> Result<EventSource, Error>
    where
        F: FnMut(GenerateResponse) -> (),
    {
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

        let url = format!("{}/api/generate", self.base_url);
        let response = self
            .client
            .request(Method::POST, url)
            .headers(headers)
            .json(&json_body)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;
        trace!("Initializing SSE stream");

        EventSource::new(response)
            .map_err(|err| from_event_source_error("Failed to create SSE stream", err))
    }

    /// Load a model into memory
    pub fn load_model(&self, model_name: &str) -> Result<GenerateResponse, Error> {
        let params = GenerateParams {
            model: Some(model_name.to_owned()),
            prompt: None,
            ..Default::default()
        };

        self.generate_completion(&params)
    }

    /// Unload a model from memory
    pub fn unload_model(&self, model_name: &str) -> Result<GenerateResponse, Error> {
        let params = GenerateParams {
            model: Some(model_name.to_owned()),
            prompt: None,
            keep_alive: Some("0".to_string()),
            ..Default::default()
        };

        self.generate_completion(&params)
    }
}

/// GenerateParams is Params for generating completions
///
/// Refer to https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion for more details.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerateParams {
    /// If NONE then the default model will be used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaModelOptions>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,

    /// If false the response will be returned as a single response object, rather than a stream of objects.
    /// For `generate_completion`  will be set to false
    /// For `generate_completion_stream`  will be set to true
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

/// ChatRequest is parameters for a request to the chat endpoint
///
/// Refer to https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion for more details
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatRequest {
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

#[derive(Debug, Serialize, Deserialize)]
pub struct MessageRequest {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub description: Option<String>,
    pub tool_object: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<MessageResponse>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<bool>,
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

#[derive(Debug, Serialize, Deserialize)]
pub struct MessageResponse {
    pub role: MessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ToolCall>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<Vec<Function>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaRequestError {
    status_code: i32,
    status: String,
    error_message: String,
}

pub fn handle_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();

    match status {
        StatusCode::OK => {
            let raw_body = response
                .text()
                .map_err(|err| from_reqwest_error("Failed to receive response body", err))?;
            trace!("Received response from OpenRouter API: {raw_body:?}");

            match serde_json::from_str::<T>(&raw_body) {
                Ok(body) => {
                    trace!("Received response from OpenRouter API: {body:?}");
                    Ok(body)
                }
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
                message: error_body.status,
                provider_error_json: Some(error_body.error_message),
            })
        }
    }
}
