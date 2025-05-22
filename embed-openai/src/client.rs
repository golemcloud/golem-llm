use golem_embed::error::EmbedError;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub embedding: Vec<f32>,
    pub index: u32,
    pub object: String,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub object: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIError {
    pub message: String,
    pub r#type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

pub async fn create_embeddings(
    api_key: &str,
    organization_id: Option<&str>,
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
    
    if let Some(org_id) = organization_id {
        headers.insert(
            "OpenAI-Organization",
            HeaderValue::from_str(org_id)
                .map_err(|e| EmbedError::InvalidRequest(format!("Invalid organization ID: {}", e)))?,
        );
    }
    
    let response = client
        .post("https://api.openai.com/v1/embeddings")
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
            
        // Try to parse as OpenAI error format
        if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(&error_text) {
            return match error_response.error.r#type.as_str() {
                "invalid_request_error" => Err(EmbedError::InvalidRequest(error_response.error.message)),
                "authentication_error" => Err(EmbedError::ProviderError(error_response.error.message)),
                "permission_error" => Err(EmbedError::ProviderError(error_response.error.message)),
                "rate_limit_error" => Err(EmbedError::RateLimitExceeded(error_response.error.message)),
                "server_error" => Err(EmbedError::ProviderError(error_response.error.message)),
                _ => Err(EmbedError::Unknown(error_response.error.message)),
            };
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