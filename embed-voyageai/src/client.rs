use golem_embed::error::EmbedError;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct EmbeddingRequest {
    pub input: Vec<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_prompt: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct RerankRequest {
    pub query: String,
    pub documents: Vec<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct RerankResult {
    pub index: u32,
    pub score: f32,
    pub document: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RerankResponse {
    pub results: Vec<RerankResult>,
    pub model: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct VoyageAIError {
    pub message: String,
    pub error_type: Option<String>,
}

pub async fn create_embeddings(
    api_key: &str,
    request: &EmbeddingRequest,
) -> Result<EmbeddingResponse, EmbedError> {
    let client = reqwest::Client::new();
    
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", api_key))
            .map_err(|e| EmbedError::InvalidRequest(format!("Invalid API key: {}", e)))?,
    );
    
    let response = client
        .post("https://api.voyageai.com/v1/embeddings")
        .headers(headers)
        .json(request)
        .send()
        .await
        .map_err(|e| EmbedError::ProviderError(format!("Failed to send request: {}", e)))?;
    
    let status = response.status();
    
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
            
        // Try to parse as Voyage AI error format
        if let Ok(error_response) = serde_json::from_str::<VoyageAIError>(&error_text) {
            return Err(EmbedError::ProviderError(error_response.message));
        }
        
        // If we couldn't parse the error, return the raw text
        return Err(EmbedError::ProviderError(format!("Error {}: {}", status, error_text)));
    }
    
    let embedding_response = response
        .json::<EmbeddingResponse>()
        .await
        .map_err(|e| EmbedError::ProviderError(format!("Failed to parse response: {}", e)))?;
    
    Ok(embedding_response)
}

pub async fn rerank_documents(
    api_key: &str,
    request: &RerankRequest,
) -> Result<RerankResponse, EmbedError> {
    let client = reqwest::Client::new();
    
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", api_key))
            .map_err(|e| EmbedError::InvalidRequest(format!("Invalid API key: {}", e)))?,
    );
    
    let response = client
        .post("https://api.voyageai.com/v1/rerank")
        .headers(headers)
        .json(request)
        .send()
        .await
        .map_err(|e| EmbedError::ProviderError(format!("Failed to send request: {}", e)))?;
    
    let status = response.status();
    
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
            
        // Try to parse as Voyage AI error format
        if let Ok(error_response) = serde_json::from_str::<VoyageAIError>(&error_text) {
            return Err(EmbedError::ProviderError(error_response.message));
        }
        
        // If we couldn't parse the error, return the raw text
        return Err(EmbedError::ProviderError(format!("Error {}: {}", status, error_text)));
    }
    
    let rerank_response = response
        .json::<RerankResponse>()
        .await
        .map_err(|e| EmbedError::ProviderError(format!("Failed to parse response: {}", e)))?;
    
    Ok(rerank_response)
}