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

pub struct CohereEmbedding {
    api_key: String,
}

impl CohereEmbedding {
    pub fn new() -> Self {
        LOGGING_STATE.with(|state| state.borrow_mut().init());
        
        let api_key = with_config_key("COHERE_API_KEY", |key| Some(key.to_string()))
            .expect("COHERE_API_KEY environment variable must be set");
            
        Self {
            api_key,
        }
    }
    
    async fn generate_embeddings(
        &self,
        inputs: Vec<ContentPart>,
        config: Config,
    ) -> Result<EmbeddingResponse, Error> {
        let model = config.model.as_deref().unwrap_or("embed-english-v3.0");
        
        // Convert inputs to text (handling images if needed)
        let input_texts = conversions::content_parts_to_text(inputs)?;
        
        // Create the request to Cohere API
        let request = client::EmbeddingRequest {
            model: model.to_string(),
            texts: input_texts,
            input_type: Some("search_document".to_string()),
            truncate: config.truncation,
        };
        
        // Send the request to Cohere
        match client::create_embeddings(&self.api_key, &request).await {
            Ok(response) => {
                // Convert the response to the WIT interface format
                Ok(conversions::convert_embedding_response(response, model))
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
        let model = config.model.as_deref().unwrap_or("rerank-english-v2.0");
        
        // Create the request to Cohere API
        let request = client::RerankRequest {
            model: model.to_string(),
            query,
            documents,
            top_n: None,
            return_documents: Some(true),
        };
        
        // Send the request to Cohere
        match client::rerank_documents(&self.api_key, &request).await {
            Ok(response) => {
                // Convert the response to the WIT interface format
                Ok(conversions::convert_rerank_response(response, model))
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
impl DurableEmbed for CohereEmbedding {
    fn save_state(&self) -> Result<(), String> {
        // No state to save for Cohere embeddings
        Ok(())
    }

    fn load_state(&self) -> Result<(), String> {
        // No state to load for Cohere embeddings
        Ok(())
    }
}

impl Guest for CohereEmbedding {
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

golem_embed::export_embed!(CohereEmbedding);