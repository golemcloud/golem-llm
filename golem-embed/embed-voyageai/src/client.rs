use std::fmt::Debug;

use golem_embed::{
    error::{error_code_from_status, from_reqwest_error},
    golem::embed::embed::Error,
};
use log::trace;
use reqwest::{Client, Method, Response};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json;

const BASE_URL: &str = "https://api.voyageai.com";

/// The VoyageAI API client for creating embeddings and reranking.
///
/// Based on https://docs.voyageai.com/reference/embeddings-api
/// and https://docs.voyageai.com/reference/reranker-api
pub struct VoyageAIApi {
    api_key: String,
    client: Client,
}

impl VoyageAIApi {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self { api_key, client }
    }

    pub fn generate_embedding(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, Error> {
        trace!("Sending embedding request to VoyageAI API: {request:?}");
        let response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v1/embeddings"))
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Embedding request failed", err))?;
        parse_response::<EmbeddingResponse>(response)
    }

    pub fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, Error> {
        trace!("Sending rerank request to VoyageAI API: {request:?}");
        let response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v1/rerank"))
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Rerank request failed", err))?;
        parse_response::<RerankResponse>(response)
    }
}

fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();
    match status.is_success() {
        true => {
            let response_text = response.text().map_err(|err| from_reqwest_error("Failed to read response body", err))?;
            match serde_json::from_str::<T>(&response_text) {
                Ok(response_data) => {
                    trace!("Response from VoyageAI API: {response_data:?}");
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
        },
        false => {
            let error_text = response.text().ok();
            Err(Error {
                code: error_code_from_status(status),
                message: "Failed to parse response".to_string(),
                provider_error_json: error_text,
            })
        },
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingRequest {
    pub input: Vec<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<InputType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimension: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dtype: Option<OutputDtype>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EncodingFormat>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum EncodingFormat {
    Base64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum InputType {
    Document,
    Query,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum OutputDtype {
    Float,
    Int8,
    Uint8,
    Binary,
    Ubinary,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingUsage {
    pub total_tokens: u32,
}

// Rerank API structures
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankRequest {
    pub query: String,
    pub documents: Vec<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankResponse {
    pub object: String,
    pub data: Vec<RerankResult>,
    pub model: String,
    pub usage: RerankUsage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankResult {
    pub index: u32,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankUsage {
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VoyageAIError {
    pub error: VoyageAIErrorDetails,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VoyageAIErrorDetails {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}
