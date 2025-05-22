use golem_embed::config::with_config_key;
use golem_embed::durability::{DurableEmbed, ExtendedGuest};
use golem_embed::error::{map_error_code, EmbedError};
use golem_embed::golem::embed::embed::{Config, ContentPart, Error, Guest, EmbeddingResponse, ReRankResponse};
use golem_embed::LOGGING_STATE;
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "durability")]
use golem_rust::durability::*;

mod client;
mod conversions;

#[derive(Debug)]
pub struct OpenAIEmbedding {
    api_key: String,
    organization_id: Option<String>,
}

impl OpenAIEmbedding {
    pub fn new() -> Self {
        LOGGING_STATE.with(|state| state.borrow_mut().init());
        
        let api_key = with_config_key("OPENAI_API_KEY", |key| Some(key.to_string()))
            .expect("OPENAI_API_KEY environment variable must be set");
            
        let organization_id = with_config_key("OPENAI_ORGANIZATION_ID", |org| Some(org.to_string()));
        
        Self {
            api_key,
            organization_id,
        }
    }
    
    async fn generate_embeddings(
        &self,
        inputs: Vec<ContentPart>,
        config: Config,
    ) -> Result<EmbeddingResponse, Error> {
        let model = config.model.as_deref().unwrap_or("text-embedding-3-large");
        
        // Convert inputs to text (handling images if needed)
        let input_texts = conversions::content_parts_to_text(inputs)?;
        
        // Create the request to OpenAI API
        let request = client::EmbeddingRequest {
            model: model.to_string(),
            input: input_texts,
            encoding_format: Some("float".to_string()),
            dimensions: config.dimensions,
            user: config.user,
        };
        
        // Send the request to OpenAI
        match client::create_embeddings(&self.api_key, self.organization_id.as_deref(), &request).await {
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
        // OpenAI doesn't have a native reranking endpoint, so return an error
        Err(Error {
            code: golem_embed::golem::embed::embed::ErrorCode::Unsupported,
            message: "OpenAI does not support native reranking. Consider using a different provider like VoyageAI".to_string(),
            provider_error_json: None,
        })
    }
}

#[cfg(feature = "durability")]
impl DurableEmbed for OpenAIEmbedding {
    fn save_state(&self) -> Result<(), String> {
        // No state to save for OpenAI embeddings
        Ok(())
    }

    fn load_state(&self) -> Result<(), String> {
        // No state to load for OpenAI embeddings
        Ok(())
    }
}

impl Guest for OpenAIEmbedding {
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

golem_embed::export_embed!(OpenAIEmbedding);