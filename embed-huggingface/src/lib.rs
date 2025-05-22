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

pub struct HuggingFaceEmbedding {
    api_key: String,
    default_model: String,
}

impl HuggingFaceEmbedding {
    pub fn new() -> Self {
        LOGGING_STATE.with(|state| state.borrow_mut().init());
        
        let api_key = with_config_key("HUGGINGFACE_API_KEY", |key| Some(key.to_string()))
            .expect("HUGGINGFACE_API_KEY environment variable must be set");
            
        let default_model = with_config_key("HUGGINGFACE_MODEL_ID", |model| Some(model.to_string()))
            .unwrap_or_else(|| "sentence-transformers/all-MiniLM-L6-v2".to_string());
        
        Self {
            api_key,
            default_model,
        }
    }
    
    async fn generate_embeddings(
        &self,
        inputs: Vec<ContentPart>,
        config: Config,
    ) -> Result<EmbeddingResponse, Error> {
        let model = config.model.as_deref().unwrap_or(&self.default_model);
        
        // Convert inputs to text (handling images if needed)
        let input_texts = conversions::content_parts_to_text(inputs)?;
        
        // Create the request to Hugging Face API
        let options = client::EmbeddingOptions {
            truncate: config.truncation,
            normalize: None,
        };
        
        let request = client::EmbeddingRequest {
            inputs: input_texts,
            options: Some(options),
        };
        
        // Send the request to Hugging Face
        match client::create_embeddings(&self.api_key, model, &request).await {
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
        // Hugging Face doesn't have a native reranking endpoint, so return an error
        Err(Error {
            code: golem_embed::golem::embed::embed::ErrorCode::Unsupported,
            message: "Hugging Face does not support native reranking. Consider using a different provider like VoyageAI".to_string(),
            provider_error_json: None,
        })
    }
}

#[cfg(feature = "durability")]
impl DurableEmbed for HuggingFaceEmbedding {
    fn save_state(&self) -> Result<(), String> {
        // No state to save for Hugging Face embeddings
        Ok(())
    }

    fn load_state(&self) -> Result<(), String> {
        // No state to load for Hugging Face embeddings
        Ok(())
    }
}

impl Guest for HuggingFaceEmbedding {
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

golem_embed::export_embed!(HuggingFaceEmbedding);