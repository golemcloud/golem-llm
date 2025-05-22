use golem_embed::error::EmbedError;
use golem_embed::golem::embed::embed::{ContentPart, Embedding, EmbeddingResponse, ReRankResult, ReRankResponse, Usage};
use crate::client;

/// Convert ContentPart vector to text strings for the Cohere API
pub fn content_parts_to_text(parts: Vec<ContentPart>) -> Result<Vec<String>, golem_embed::golem::embed::embed::Error> {
    let mut texts = Vec::new();
    
    for part in parts {
        match part {
            ContentPart::Text(text) => texts.push(text),
            ContentPart::Image(image_url) => {
                return Err(golem_embed::golem::embed::embed::Error {
                    code: golem_embed::golem::embed::embed::ErrorCode::Unsupported,
                    message: "Cohere embeddings do not support image inputs".to_string(),
                    provider_error_json: None,
                });
            }
        }
    }
    
    Ok(texts)
}

/// Convert Cohere embedding response to WIT interface format
pub fn convert_embedding_response(response: client::EmbeddingResponse, model: &str) -> EmbeddingResponse {
    let embeddings = response.embeddings
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| Embedding {
            index: i as u32,
            vector: embedding,
        })
        .collect();
    
    let input_tokens = response.meta
        .as_ref()
        .and_then(|meta| meta.billed_units.as_ref())
        .and_then(|billed| billed.input_tokens);
    
    let usage = if input_tokens.is_some() {
        Some(Usage {
            input_tokens,
            total_tokens: input_tokens,
        })
    } else {
        None
    };
    
    EmbeddingResponse {
        embeddings,
        usage,
        model: model.to_string(),
        provider_metadata_json: Some(serde_json::json!({"id": response.id}).to_string()),
    }
}

/// Convert Cohere rerank response to WIT interface format
pub fn convert_rerank_response(response: client::RerankResponse, model: &str) -> ReRankResponse {
    let results = response.results
        .into_iter()
        .map(|result| ReRankResult {
            index: result.index,
            relevance_score: result.relevance_score,
            document: result.document,
        })
        .collect();
    
    let input_tokens = response.meta
        .as_ref()
        .and_then(|meta| meta.billed_units.as_ref())
        .and_then(|billed| billed.input_tokens);
    
    let usage = if input_tokens.is_some() {
        Some(Usage {
            input_tokens,
            total_tokens: input_tokens,
        })
    } else {
        None
    };
    
    ReRankResponse {
        results,
        usage,
        model: model.to_string(),
        provider_metadata_json: Some(serde_json::json!({"id": response.id}).to_string()),
    }
}