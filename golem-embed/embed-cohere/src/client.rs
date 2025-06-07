use std::fmt::Debug;

use golem_embed::{
    error::{error_code_from_status, from_reqwest_error},
    golem::embed::embed::Error,
};
use log::trace;
use reqwest::{Client, Method, Response};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

const BASE_URL: &str = "https://api.cohere.ai";

/// The Cohere API client for creating embeddings.
///
/// Based on https://docs.cohere.com/reference/embed
pub struct EmbeddingsApi {
    cohere_api_key: String,
    client: Client,
}

impl EmbeddingsApi {
    pub fn new(cohere_api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self {
            cohere_api_key,
            client,
        }
    }

    pub fn generate_embeding(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, Error> {
        trace!("Sending request to Cohere API: {request:?}");
        let response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v2/embed"))
            .bearer_auth(&self.cohere_api_key)
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;
        trace!("Recived response: {:#?}", response);
        parse_response::<EmbeddingResponse>(response)
    }

    pub fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, Error> {
        trace!("Sending request to Cohere API: {request:?}");
        let response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v2/rerank"))
            .bearer_auth(&self.cohere_api_key)
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;
        trace!("Recived response: {:#?}", response);
        parse_response::<RerankResponse>(response)
    }
}

fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();
    if status.is_success() {
        let response_data = response
            .json::<T>()
            .map_err(|error| from_reqwest_error("Failed to decode response body", error))?;
        trace!("Response from Cohere API: {response_data:?}");
        Ok(response_data)
    } else {
        let response_data = response
            .text()
            .map_err(|error| from_reqwest_error("Failed to decode response body", error))?;
        trace!("Response from Cohere API: {response_data:?}");
        Err(Error {
            code: error_code_from_status(status),
            message: format!("Request failed with {status}"),
            provider_error_json: Some(response_data),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputType {
    #[serde(rename = "search_document")]
    SearchDocument,
    #[serde(rename = "search_query")]
    SearchQuery,
    #[serde(rename = "classification")]
    Classification,
    #[serde(rename = "clustering")]
    Clustering,
    #[serde(rename = "image")]
    Image,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingType {
    #[serde(rename = "float")]
    Float,
    #[serde(rename = "int8")]
    Int8,
    #[serde(rename = "uint8")]
    Uint8,
    #[serde(rename = "binary")]
    Binary,
    #[serde(rename = "ubinary")]
    Ubinary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Truncate {
    #[serde(rename = "NONE")]
    None,
    #[serde(rename = "START")]
    Start,
    #[serde(rename = "END")]
    End,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input_type: InputType,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_types: Option<Vec<EmbeddingType>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub texts: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimension: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<Truncate>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingResponse {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ImageRespnse>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub texts: Option<Vec<String>>,

    pub embeddings: EmbeddingData,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<Meta>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageRespnse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bit_depth: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub float: Option<Vec<Vec<f32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub int8: Option<Vec<Vec<i8>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uint8: Option<Vec<Vec<u8>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary: Option<Vec<Vec<i8>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ubinary: Option<Vec<Vec<u8>>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CohereError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens_per_doc: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankResponse {
    pub results: Vec<RerankData>,
    pub scores: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<Meta>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankData {
    pub index: u32,
    pub relevance_score: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Meta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<ApiVersion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub billed_units: Option<BilledUnits>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<MetaTokens>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MetaTokens {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiVersion {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_deprecated: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_experimental: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BilledUnits {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_units: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classifications: Option<u32>,
}

#[cfg(test)]
mod tests {
    use crate::client;

    use super::*;
    use serde_json;

    fn parse_json<T: DeserializeOwned + Debug>(response: &str) -> Result<T, serde_json::Error> {
        serde_json::from_str::<T>(response)
    }

    #[test]
    fn test_parse_embedding_response_data() {
        let json_data = r#"{
                "id": "54910170-852f-4322-9767-63d36e55c3bf",
                "texts": [
                "This is the sentence I want to embed.",
                "Hey !"
                ],
                "embeddings": {
                "binary": [[-54,99,-87,60,9]], 
                "float": [[0.016967773,0.031982422,0.041503906],[0.013977051,0.012084961,0.005554199,-0.053955078]], 
                "int8": [[-15,-65,0,-31,-43,-14],[14,38,-30,-13,-49,4,-33,-49]]
                },
                "meta": {
                "api_version": {
                    "version": "2"
                },
                "billed_units": {
                    "input_tokens": 11,
                    "image_tokens": 0
                }
                },
                "response_type": "embeddings_by_type"
          }"#;

        let result = parse_json::<client::EmbeddingResponse>(json_data);
        print!("{:?}", result);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_rerank_response_data() {
        let json_data = r#"{
            "results": [
                {
                "index": 3,
                "relevance_score": 0.999071
                },
                {
                "index": 4,
                "relevance_score": 0.7867867
                },
                {
                "index": 0,
                "relevance_score": 0.32713068
                }
            ],
            "id": "07734bd2-2473-4f07-94e1-0d9f0e6843cf",
            "meta": {
                "api_version": {
                "version": "2",
                "is_experimental": false
                },
                "billed_units": {
                "search_units": 1
                }
            }
        }"#;
        let result = parse_json::<client::RerankResponse>(json_data);
        print!("{:?}", result);
        assert!(result.is_ok());
    }
}
