use golem_embed::error::EmbedError;
use golem_embed::golem::embed::embed::{ContentPart, Embedding, EmbeddingResponse, ReRankResult, ReRankResponse, Usage};
use crate::client;

/// Convert ContentPart vector to text strings for the Voyage AI API
pub fn content_parts_to_text(parts: Vec<ContentPart>) -> Result<Vec<String>, golem_embed::golem::embed::embed::Error> {
    let mut texts = Vec::new();
    
    for part in parts {
        match part {
            ContentPart::Text(text) => texts.push(text),
            ContentPart::Image(image_url) => {
                return Err(golem_embed::golem::embed::embed::Error {
                    code: golem_embed::golem::embed::embed::ErrorCode::Unsupported,
                    message: "Voyage AI embeddings do not support image inputs".to_string(),
                    provider_error_json: None,
                });
            }
        }
    }
    
    Ok(texts)
}

/// Convert Voyage AI embedding response to WIT interface format
pub fn convert_embedding_response(response: client::EmbeddingResponse) -> EmbeddingResponse {
    let embeddings = response.embeddings
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| Embedding {
            index: i as u32,
            vector: embedding,
        })
        .collect();
    
    let usage = response.usage.map(|u| Usage {
        input_tokens: u.input_tokens,
        total_tokens: u.total_tokens,
    });
    
    EmbeddingResponse {
        embeddings,
        usage,
        model: response.model,
        provider_metadata_json: None,
    }
}

/// Convert Voyage AI rerank response to WIT interface format
pub fn convert_rerank_response(response: client::RerankResponse) -> ReRankResponse {
    let results = response.results
        .into_iter()
        .map(|result| ReRankResult {
            index: result.index,
            relevance_score: result.score,
            document: result.document,
        })
        .collect();
    
    let usage = response.usage.map(|u| Usage {
        input_tokens: u.input_tokens,
        total_tokens: u.total_tokens,
    });
    
    ReRankResponse {
        results,
        usage,
        model: response.model,
        provider_metadata_json: None,
    }
}