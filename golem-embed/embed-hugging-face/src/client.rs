use std::fmt::Debug;

use golem_embed::{
    error::{error_code_from_status, from_reqwest_error},
    golem::embed::embed::Error,
};
use log::trace;
use reqwest::{Client, Method, Response};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

const BASE_URL: &str = "https://router.huggingface.co/hf-inference";

/// The Hugging Face API client for creating embeddings.
///
/// Based on https://huggingface.co/docs/api-inference/index
///
pub struct EmbeddingsApi {
    huggingface_api_key: String,
    client: Client,
}

impl EmbeddingsApi {
    pub fn new(huggingface_api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self {
            huggingface_api_key,
            client,
        }
    }

    pub fn generate_embedding(
        &self,
        request: EmbeddingRequest,
        model: &str,
    ) -> Result<EmbeddingResponse, Error> {
        trace!("Sending request to Hugging Face API: {request:?}");
        let response = self
            .client
            .request(Method::POST, format!(
                "{BASE_URL}/models/{model}/pipeline/feature-extraction"
            ))
            .bearer_auth(&self.huggingface_api_key)
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;
        parse_response::<EmbeddingResponse>(response)
    }


    pub fn rerank(&self, request: RerankRequest, model: &str) -> Result<RerankResponse, Error> {
        trace!("Sending rerank request to Hugging Face API: {request:?}");
        let response = self
            .client
            .request(Method::POST, format!(
                "{BASE_URL}/models/{model}/pipeline/text-classification"
            ))
            .bearer_auth(&self.huggingface_api_key)
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;
        parse_response::<RerankResponse>(response)
    }
}

fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();
    if status.is_success() {
        let response_data = response
            .json::<T>()
            .map_err(|error| from_reqwest_error("Failed to decode response body", error))?;
        trace!("Response from Hugging Face API: {response_data:?}");
        Ok(response_data)
    } else {
        let response_data = response
            .text()
            .map_err(|error| from_reqwest_error("Failed to decode response body", error))?;
        trace!("Error response from Hugging Face API: {response_data:?}");
        Err(Error {
            code: error_code_from_status(status),
            message: format!("Request failed with {status}"),
            provider_error_json: Some(response_data),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
    /// The name of the prompt that should be used by for encoding.
    /// If not set, no prompt will be applied. Must be a key in the
    /// `sentence-transformers` configuration `prompts` dictionary.
    /// For example if `prompt_name` is “query” and the `prompts` is {“query”: “query: ”, …},
    /// then the sentence “What is the capital of France?” will be encoded as
    /// “query: What is the capital of France?” because the prompt text will
    /// be prepended before any text to encode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate_direction: Option<TruncateDirection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TruncateDirection {
    #[serde(rename = "left")]
    Left,
    #[serde(rename = "right")]
    Right,
}

pub type EmbeddingResponse = Vec<Vec<f32>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankRequest {
    pub query: String,
    pub documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    pub results: Vec<RerankResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    pub index: u32,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}
