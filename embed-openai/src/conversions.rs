use golem_embed::error::EmbedError;
use golem_embed::golem::embed::embed::{ContentPart, Embedding, EmbeddingResponse, Usage};
use crate::client;

/// Convert ContentPart vector to text strings for the OpenAI API
pub fn content_parts_to_text(parts: Vec<ContentPart>) -> Result<Vec<String>, golem_embed::golem::embed::embed::Error> {
    let mut texts = Vec::new();
    
    for part in parts {
        match part {
            ContentPart::Text(text) => texts.push(text),
            ContentPart::Image(image_url) => {
                return Err(golem_embed::golem::embed::embed::Error {
                    code: golem_embed::golem::embed::embed::ErrorCode::Unsupported,
                    message: "OpenAI embeddings do not support image inputs".to_string(),
                    provider_error_json: None,
                });
            }
        }
    }
    
    Ok(texts)
}

/// Convert OpenAI embedding response to WIT interface format
pub fn convert_embedding_response(response: client::EmbeddingResponse, model: &str) -> EmbeddingResponse {
    let embeddings = response.data
        .into_iter()
        .map(|data| Embedding {
            index: data.index,
            vector: data.embedding,
        })
        .collect();
    
    let usage = Some(Usage {
        input_tokens: Some(response.usage.prompt_tokens),
        total_tokens: Some(response.usage.total_tokens),
    });
    
    EmbeddingResponse {
        embeddings,
        usage,
        model: model.to_string(),
        provider_metadata_json: Some(serde_json::json!({"object": response.object}).to_string()),
    }
}