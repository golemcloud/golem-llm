use golem_embed::error::EmbedError;
use golem_embed::golem::embed::embed::{ContentPart, Embedding, EmbeddingResponse, Usage};
use crate::client;

/// Convert ContentPart vector to text strings for the Hugging Face API
pub fn content_parts_to_text(parts: Vec<ContentPart>) -> Result<Vec<String>, golem_embed::golem::embed::embed::Error> {
    let mut texts = Vec::new();
    
    for part in parts {
        match part {
            ContentPart::Text(text) => texts.push(text),
            ContentPart::Image(image_url) => {
                return Err(golem_embed::golem::embed::embed::Error {
                    code: golem_embed::golem::embed::embed::ErrorCode::Unsupported,
                    message: "Hugging Face embeddings do not support image inputs".to_string(),
                    provider_error_json: None,
                });
            }
        }
    }
    
    Ok(texts)
}

/// Convert Hugging Face embedding response to WIT interface format
pub fn convert_embedding_response(response: client::EmbeddingResponse, model: &str) -> EmbeddingResponse {
    let embeddings = response.embeddings
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| Embedding {
            index: i as u32,
            vector: embedding,
        })
        .collect();
    
    // Hugging Face doesn't provide token usage information
    let usage = None;
    
    EmbeddingResponse {
        embeddings,
        usage,
        model: model.to_string(),
        provider_metadata_json: None,
    }
}