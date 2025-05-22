use golem_embed::config::with_config_key;
use golem_embed::durability::{DurableEmbed, ExtendedGuest};
use golem_embed::error::{map_error_code, EmbedError};
use golem_embed::golem::embed::embed::{Config, ContentPart, Error, Guest, EmbeddingResponse, ReRankResponse};
use golem_embed::LOGGING_STATE;
use log::{debug, error, info};
use serde::{Deserialize, Serialize};

#[cfg(feature = "durability")]
use golem_rust::durability::*;

mod client;
mod conversions;

pub struct VoyageAIEmbedding {
    api_key: String,
    default_embedding_model: String,
    default_rerank_model: String,
}

impl VoyageAIEmbedding {
    pub fn new() -> Self {
        LOGGING_STATE.with(|state| state.borrow_mut().init());
        
        let api_key = with_config_key("VOYAGEAI_API_KEY", |key| Some(key.to_string()))
            .expect("VOYAGEAI_API_KEY environment variable must be set");
            
        let default_embedding_model = with_config_key("VOYAGEAI_EMBEDDING_MODEL", |model| Some(model.to_string()))
            .unwrap_or_else(|| "voyage-2".to_string());
            
        let default_rerank_model = with_config_key("VOYAGEAI_RERANK_MODEL", |model| Some(model.to_string()))
            .unwrap_or_else(|| "voyage-rerank-2".to_string());
        
        Self {
            api_key,
            default_embedding_model,
            default_rerank_model,
        }
    }
    
    async fn generate_embeddings(
        &self,
        inputs: Vec<ContentPart>,
        config: Config,
    ) -> Result<EmbeddingResponse, Error> {
        let model = config.model.as_deref().unwrap_or(&self.default_embedding_model);
        
        // Convert inputs to text (handling images if needed)
        let input_texts = conversions::content_parts_to_text(inputs)?;
        
        // Create the request to Voyage AI API
        let request = client::EmbeddingRequest {
            input: input_texts,
            model: model.to_string(),
            truncate: config.truncation,
            normalize: None,
            include_prompt: None,
        };
        
        // Send the request to Voyage AI
        match client::create_embeddings(&self.api_key, &request).await {
            Ok(response) => {
                // Convert the response to the WIT interface format
                Ok(conversions::convert_embedding_response(response))
            }
            Err(err) => {
                error!("Error generating embeddings: {}", err);
                Err(Error {
                    code: map_error_code(&err),
                    message: err.to_string(),
                    provider_error_json: None,
                })
            }
        }
    }
    
    async fn rerank_documents(
        &self,
        query: String,
        documents: Vec<String>,
        config: Config,
    ) -> Result<ReRankResponse, Error> {
        let model = config.model.as_deref().unwrap_or(&self.default_rerank_model);
        
        // Create the request to Voyage AI API
        let request = client::RerankRequest {
            query,
            documents,
            model: model.to_string(),
            return_documents: Some(true),
            top_k: None,
        };
        
        // Send the request to Voyage AI
        match client::rerank_documents(&self.api_key, &request).await {
            Ok(response) => {
                // Convert the response to the WIT interface format
                Ok(conversions::convert_rerank_response(response))
            }
            Err(err) => {
                error!("Error reranking documents: {}", err);
                Err(Error {
                    code: map_error_code(&err),
                    message: err.to_string(),
                    provider_error_json: None,
                })
            }
        }
    }
}

#[cfg(feature = "durability")]
impl DurableEmbed for VoyageAIEmbedding {
    fn save_state(&self) -> Result<(), String> {
        // No state to save for Voyage AI embeddings
        Ok(())
    }

    fn load_state(&self) -> Result<(), String> {
        // No state to load for Voyage AI embeddings
        Ok(())
    }
}

impl Guest for VoyageAIEmbedding {
    fn generate(
        &mut self,
        inputs: Vec<ContentPart>,
        config: Config,
    ) -> Result<EmbeddingResponse, Error> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error {
                code: golem_embed::golem::embed::embed::ErrorCode::InternalError,
                message: format!("Failed to create Tokio runtime: {}", e),
                provider_error_json: None,
            })?;

        runtime.block_on(self.generate_embeddings(inputs, config))
    }

    fn rerank(
        &mut self,
        query: String,
        documents: Vec<String>,
        config: Config,
    ) -> Result<ReRankResponse, Error> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error {
                code: golem_embed::golem::embed::embed::ErrorCode::InternalError,
                message: format!("Failed to create Tokio runtime: {}", e),
                provider_error_json: None,
            })?;

        runtime.block_on(self.rerank_documents(query, documents, config))
    }
}

golem_embed::export_embed!(VoyageAIEmbedding);