use std::fmt::Debug;

use base64::{engine::general_purpose, Engine};
use golem_embed::{
    error::{error_code_from_status, from_reqwest_error},
    golem::embed::embed::Error,
};
use log::trace;

#[allow(dead_code, unused, unused_imports)]
use reqwest::Client;
use reqwest::{Method, Response};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

const BASE_URL: &str = "https://api.openai.com";

/// The OpenAI API client for creating embeddings.
///
/// Based on https://platform.openai.com/docs/api-reference/embeddings/create
pub struct EmbeddingsApi {
    openai_api_key: String,
    client: reqwest::Client,
}

impl EmbeddingsApi {
    pub fn new(openai_api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self {
            openai_api_key,
            client,
        }
    }

    pub fn generate_embeding(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        trace!("Sending request to OpenAI API: {request:?}");
        let response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v1/embeddings"))
            .bearer_auth(&self.openai_api_key)
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;
        parse_response::<EmbeddingResponse>(response)
    }
}

fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();
    let response_text = response
        .text()
        .map_err(|err| from_reqwest_error("Failed to read response body", err))?;
    match serde_json::from_str::<T>(&response_text) {
        Ok(response_data) => {
            trace!("Response from Hugging Face API: {response_data:?}");
            Ok(response_data)
        }
        Err(error) => {
            trace!("Error parsing response: {error:?}");
            Err(Error {
                code: error_code_from_status(status),
                message: format!("Failed to decode response body: {}", response_text),
                provider_error_json: Some(error.to_string()),
            })
        }
    }
}

/// OpenAI allows only allows float and base64 as output formats.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub enum EncodingFormat {
    #[serde(rename = "float")]
    Float,
    #[serde(rename = "base64")]
    Base64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct EmbeddingRequest {
    pub input: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EncodingFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimension: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: EmbeddingVector,
    pub index: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum EmbeddingVector {
    FloatArray(Vec<f32>),
    Base64(String),
}

impl EmbeddingVector {
    pub fn to_float_vec(&self) -> Result<Vec<f32>, String> {
        match self {
            EmbeddingVector::FloatArray(vec) => Ok(vec.clone()),
            EmbeddingVector::Base64(base64_str) => {
                let bytes = general_purpose::STANDARD
                    .decode(base64_str)
                    .map_err(|e| format!("Failed to decode base64: {}", e))?;

                if bytes.len() % 4 != 0 {
                    return Err("Invalid base64 data: length not divisible by 4".to_string());
                }

                let mut floats: Vec<f32> = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }

                Ok(floats)
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIError {
    pub error: OpenAIErrorDetails,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIErrorDetails {
    pub message: String,
    #[serde(rename = "type")]
    pub _type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}
