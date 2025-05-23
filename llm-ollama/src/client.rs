use golem_llm::error::{error_code_from_status, from_reqwest_error};
use golem_llm::golem::llm::llm::{Error, ErrorCode};
use log::trace;
use reqwest::{Client, Method, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use base64::{engine::general_purpose, Engine as _};

/// The Ollama API client for creating model responses.
pub struct ChatApi {
    base_url: String,
    client: Client,
}

impl ChatApi {
    pub fn new(base_url: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self { base_url, client }
    }

    pub fn send_messages(&self, request: ChatRequest) -> Result<ChatResponse, Error> {
        trace!("Sending request to Ollama API: {request:?}");

        let response: Response = self
            .client
            .request(Method::POST, format!("{}/api/chat", self.base_url))
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn stream_send_messages(&self, request: ChatRequest) -> Result<Response, Error> {
        trace!("Sending streaming request to Ollama API: {request:?}");

        let response: Response = self
            .client
            .request(Method::POST, format!("{}/api/chat", self.base_url))
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        if response.status().is_success() {
            Ok(response)
        } else {
            let status = response.status();
            let error_body = response
                .text()
                .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

            trace!("Received {status} response from Ollama API: {error_body:?}");

            Err(Error {
                code: error_code_from_status(status),
                message: format!("Request failed with {status}: {error_body}"),
                provider_error_json: Some(error_body),
            })
        }
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
        // Return just the base64 string, not the data URL format
        // Ollama expects raw base64, not "data:image/type;base64,base64string"
        let base64_image = general_purpose::STANDARD.encode(image_bytes);

        trace!("Successfully converted image to base64");

        Ok(base64_image)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ModelOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: ResponseMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    pub done: bool,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
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
            .text()
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

        trace!("Received {status} response from Ollama API: {error_body:?}");

        // Try to parse as ErrorResponse
        if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&error_body) {
            Err(Error {
                code: error_code_from_status(status),
                message: error_response.error,
                provider_error_json: Some(error_body),
            })
        } else {
            Err(Error {
                code: error_code_from_status(status),
                message: format!("Request failed with {status}: {error_body}"),
                provider_error_json: Some(error_body),
            })
        }
    }
}
