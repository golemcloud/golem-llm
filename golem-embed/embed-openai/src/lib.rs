use golem_embed::{types, EmbedError, DurableEmbed};
use reqwest::{Client, header};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

const OPENAI_API_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "text-embedding-3-large";

#[derive(Clone)]
pub struct OpenAIEmbedding {
    client: Arc<Client>,
    api_key: String,
    organization_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
    index: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
    model: String,
    usage: OpenAIEmbeddingUsage,
    object: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorDetail {
    message: Option<String>,
    #[serde(rename = "type")]
    error_type: Option<String>,
    param: Option<String>,
    code: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorResponse {
    error: OpenAIErrorDetail,
}

impl OpenAIEmbedding {
    pub fn new(api_key: String, organization_id: Option<String>) -> Result<Self, EmbedError> {
        let mut headers = header::HeaderMap::new();
        
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .map_err(EmbedError::ReqwestError)?;
        
        Ok(Self {
            client: Arc::new(client),
            api_key,
            organization_id,
        })
    }

    fn extract_text_inputs(&self, inputs: &[types::ContentPart]) -> Result<Vec<String>, EmbedError> {
        let mut texts = Vec::new();
        
        for input in inputs {
            match input {
                types::ContentPart::Text(text) => texts.push(text.clone()),
                types::ContentPart::Image(_) => {
                    return Err(EmbedError::Unsupported(
                        "OpenAI embedding API does not support image inputs".to_string(),
                    ))
                }
            }
        }
        
        Ok(texts)
    }
    
    async fn make_request<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        payload: &T,
    ) -> Result<R, EmbedError> {
        let mut request_builder = self.client
            .post(&format!("{}{}", OPENAI_API_URL, endpoint))
            .header("Authorization", format!("Bearer {}", self.api_key));
            
        if let Some(org_id) = &self.organization_id {
            request_builder = request_builder.header("OpenAI-Organization", org_id);
        }
        
        let response = request_builder
            .json(payload)
            .send()
            .await
            .map_err(EmbedError::ReqwestError)?;
        
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            
            // 尝试解析OpenAI格式的错误响应
            if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(&error_text) {
                let message = error_response.error.message.unwrap_or_else(|| 
                    format!("Unknown OpenAI error: {}", error_text)
                );
                
                return match status.as_u16() {
                    400 => Err(EmbedError::InvalidRequest(message)),
                    401 => Err(EmbedError::InvalidRequest(format!("invalid request: {}", message))),
                    404 => Err(EmbedError::ModelNotFound(message)),
                    429 => Err(EmbedError::RateLimitExceeded(message)),
                    _ => Err(EmbedError::ProviderError(message)),
                };
            }
            
            return Err(EmbedError::ProviderError(format!(
                "OpenAI API returned error: {} - {}",
                status, error_text
            )));
        }
        
        let response_data = response
            .json::<R>()
            .await
            .map_err(|e| EmbedError::ProviderError(format!("failed to parse response: {}", e)))?;
            
        Ok(response_data)
    }

    async fn create_embeddings(
        &self,
        texts: Vec<String>,
        model: &str,
    ) -> Result<OpenAIEmbeddingResponse, EmbedError> {
        let payload = json!({
            "input": texts,
            "model": model,
        });
        
        self.make_request("/embeddings", &payload).await
    }
}

impl DurableEmbed for OpenAIEmbedding {
    fn generate_durable(
        &self,
        inputs: Vec<types::ContentPart>,
        config: types::Config,
    ) -> Result<types::EmbeddingResponse, EmbedError> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| EmbedError::InternalError(format!("failed to create tokio runtime: {}", e)))?;
        
        let texts = self.extract_text_inputs(&inputs)?;
        if texts.is_empty() {
            return Err(EmbedError::InvalidRequest("no text inputs provided".to_string()));
        }
        
        // 从配置中获取模型或使用默认值
        let model = config.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
        
        // 调用OpenAI API
        let response = rt.block_on(self.create_embeddings(texts, &model))?;
        
        // 转换为通用格式的响应
        let embeddings = response.data.into_iter()
            .map(|data| types::Embedding {
                index: data.index,
                vector: data.embedding,
            })
            .collect();
            
        let usage = Some(types::Usage {
            input_tokens: Some(response.usage.prompt_tokens),
            total_tokens: Some(response.usage.total_tokens),
        });
        
        Ok(types::EmbeddingResponse {
            embeddings,
            usage,
            model: response.model,
            provider_metadata_json: Some(json!({
                "object": response.object
            }).to_string()),
        })
    }
    
    fn rerank_durable(
        &self,
        _query: String,
        _documents: Vec<String>,
        _config: types::Config,
    ) -> Result<types::RerankResponse, EmbedError> {
        Err(EmbedError::Unsupported(
            "OpenAI does not directly support reranking API. Consider using embeddings and custom ranking logic".to_string(),
        ))
    }
} 