use golem_embed::error::EmbedError;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct EmbeddingRequest {
    pub inputs: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<EmbeddingOptions>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
pub struct HuggingFaceError {
    pub error: String,
}

pub async fn create_embeddings(
    api_key: &str,
    model_id: &str,
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
    
    let endpoint = format!("https://api-inference.huggingface.co/models/{}", model_id);
    
    let response = client
        .post(&endpoint)
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
            
        // Try to parse as Hugging Face error format
        if let Ok(error_response) = serde_json::from_str::<HuggingFaceError>(&error_text) {
            return Err(EmbedError::ProviderError(error_response.error));
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