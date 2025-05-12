use base64::engine::general_purpose;
use base64::Engine;
use golem_llm::error::{error_code_from_status, from_event_source_error, from_reqwest_error};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::{Error, ErrorCode};
use log::trace;
use reqwest::header::HeaderValue;
use reqwest::{Client, Method, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub struct OllamaApi {
    base_url: String,
    client: Client,
}

impl OllamaApi {
    pub fn with_base_url(base_url: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        trace!("Creating Ollama API client with base URL: {}", base_url);
        Self { base_url, client }
    }

    //Function to download image and return it as base64
    pub fn download_image_as_base64(&self, image_url: &str) -> Result<String, Error> {
        trace!("Downloading image from URL: {}", image_url);

        // Check if URL is valid
        if !image_url.starts_with("http://") && !image_url.starts_with("https://") {
            return Err(Error {
                code: ErrorCode::InvalidRequest,
                message: format!("Invalid image URL: {}", image_url),
                provider_error_json: None,
            });
        }

        // Download the image with a reasonable timeout
        let response = self
            .client
            .get(image_url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .map_err(|err| {
                from_reqwest_error(format!("Failed to download image from {}", image_url), err)
            })?;

        if !response.status().is_success() {
            return Err(Error {
                code: error_code_from_status(response.status()),
                message: format!(
                    "Failed to download image: HTTP status {}",
                    response.status()
                ),
                provider_error_json: None,
            });
        }

        // Check content type to ensure it's an image - extract and clone the headers first
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();

        if !content_type.starts_with("image/") && content_type != "application/octet-stream" {
            return Err(Error {
                code: ErrorCode::InvalidRequest,
                message: format!(
                    "URL does not point to an image. Content-Type: {}",
                    content_type
                ),
                provider_error_json: None,
            });
        }

        // Get the image bytes with a size limit (10MB)
        let image_bytes = response
            .bytes()
            .map_err(|err| from_reqwest_error("Failed to read image data", err))?;

        // Check image size to avoid extremely large images
        const MAX_IMAGE_SIZE: usize = 10 * 1024 * 1024; // 10MB
        if image_bytes.len() > MAX_IMAGE_SIZE {
            return Err(Error {
                code: ErrorCode::InvalidRequest,
                message: format!(
                    "Image too large: {} bytes (max allowed: {} bytes)",
                    image_bytes.len(),
                    MAX_IMAGE_SIZE
                ),
                provider_error_json: None,
            });
        }

        // Convert to base64 using the recommended Engine API
        let base64_image = general_purpose::STANDARD.encode(image_bytes);

        // Determine MIME type - use the content_type we already extracted
        let mime_type = if content_type.starts_with("image/") {
            content_type
        } else {
            // If header wasn't image, try to infer from URL extension
            if image_url.ends_with(".png") {
                "image/png".to_string()
            } else if image_url.ends_with(".jpg") || image_url.ends_with(".jpeg") {
                "image/jpeg".to_string()
            } else if image_url.ends_with(".gif") {
                "image/gif".to_string()
            } else if image_url.ends_with(".webp") {
                "image/webp".to_string()
            } else if image_url.ends_with(".svg") {
                "image/svg+xml".to_string()
            } else {
                // Default to png if we can't determine
                "image/png".to_string()
            }
        };

        // Create the base64 data URL
        let data_url = format!("data:{};base64,{}", mime_type, base64_image);
        trace!("Successfully converted image to base64");

        Ok(data_url)
    }

    pub fn create_chat_completion(
        &self,
        request: OllamaChatCompletionRequest,
    ) -> Result<OllamaChatCompletionResponse, Error> {
        trace!("Sending chat completion request to Ollama OpenAI API: {request:?}");

        let mut stream_request = request.clone();
        stream_request.stream = false;

        let response: Response = self
            .client
            .request(Method::POST, format!("{}/chat/completions", self.base_url))
            .json(&stream_request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    // The OpenAI compatibility API uses standard Server-Sent Events (SSE),
    // which is compatible with golem-llm's event source handling, unlike
    // Ollama's native API which uses ndjson.
    pub fn stream_chat_completion(
        &self,
        request: OllamaChatCompletionRequest,
    ) -> Result<EventSource, Error> {
        trace!("Sending streaming chat completion request to Ollama OpenAI API: {request:?}");

        let mut stream_request = request.clone();
        stream_request.stream = true;

        let response: Response = self
            .client
            .request(Method::POST, format!("{}/chat/completions", self.base_url))
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
pub struct OllamaChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OllamaTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    #[serde(flatten)]
    pub content_container: OllamaMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

//With support for text and multimodal content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OllamaMessageContent {
    Text { content: String },
    Multimodal { content: Vec<serde_json::Value> },
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
pub struct OllamaToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub typ: String,
    pub function: OllamaToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCallFunction {
    pub name: String,
    pub arguments: String,
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
pub struct OllamaChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OllamaChatCompletionChoice>,
    pub usage: Option<OllamaUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatCompletionChoice {
    pub index: u32,
    pub message: OllamaResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaResponseMessage {
    pub role: String,
    #[serde(flatten)]
    pub content_container: Option<OllamaMessageContent>,
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
pub struct OllamaChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OllamaChunkChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChunkChoice {
    pub index: u32,
    pub delta: OllamaDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaDelta {
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
