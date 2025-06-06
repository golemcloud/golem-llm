use std::{fs, path::Path};

use base64::{engine::general_purpose, Engine};
use golem_embed::{
    error::unsupported,
    golem::embed::embed::{
        self, Config, ContentPart, Embedding, EmbeddingResponse as GolemEmbeddingResponse, Error,
        OutputDtype as GolemOutputDtype, OutputFormat as GolemOutputFormat,
        RerankResponse as GolemRerankResponse, RerankResult as GolemRerankResult, TaskType, Usage,
    },
};
use log::trace;
use reqwest::{Client, Url};

use crate::client::{
     EmbeddingRequest, EmbeddingResponse, InputType, RerankRequest, RerankResponse,
};

pub fn create_embedding_request(
    inputs: Vec<ContentPart>,
    config: Config,
) -> Result<EmbeddingRequest, Error> {
    let mut text_inputs = Vec::new();

    for input in inputs {
        match input {
            ContentPart::Text(text) => text_inputs.push(text),
            ContentPart::Image(image) => {
                return Err(unsupported(
                    "VoyageAI text embeddings do not support image inputs. Use multimodal embeddings instead.",
                ));
            }
        }
    }

    let model = config.model.unwrap_or_else(|| "voyage-3.5-lite".to_string());

    let mut input_type = None;
    match config.task_type {
        TaskType::RetrievalQuery => input_type = Some(InputType::Query),
        TaskType::RetrievalDocument => input_type = Some(InputType::Document),
        _ => return Err(unsupported("Unsupported task type")),
    }?;

    let output_dtype = match config.output_dtype {
        GolemOutputDtype::FloatArray => OutputDtype::FloatArray,
        GolemOutputDtype::Int8 => OutputDtype::Int8,
        GolemOutputDtype::Uint8 => OutputDtype::Uint8,
        GolemOutputDtype::Binary => OutputDtype::Binary,
        GolemOutputDtype::Ubinary => OutputDtype::Ubinary,
    };

    let encoding_format = match config.output_format {
        GolemOutputFormat::Base64 => OutputFormat::Base64,
        _ => None,
    };

    Ok(EmbeddingRequest {
        input: text_inputs,
        model,
        input_type,
        truncation: config.truncation,
        output_dimension: config.dimensions,
        output_dtype,
        encoding_format,
    })
}

pub fn process_embedding_response(
    response: EmbeddingResponse,
    config: Config,
) -> Result<GolemEmbeddingResponse, Error> {
    let mut embeddings = Vec::new();

    for data in response.data {
        embeddings.push(Embedding {
            index: data.index,
            vector: data.embedding,
        });
    }

    let usage = Usage {
        input_tokens: None,
        total_tokens: Some(response.usage.total_tokens),
    };

    Ok(GolemEmbeddingResponse {
        embeddings,
        usage: Some(usage),
        model: response.model,
        provider_metadata_json: None,
    })
}

pub fn create_rerank_request(
    query: String,
    documents: Vec<String>,
    config: Config,
) -> Result<RerankRequest, Error> {
    let model = config.model.unwrap_or_else(|| "rerank-2-lite".to_string());

    Ok(RerankRequest {
        query,
        documents,
        model,
        top_k : None,
        return_documents: Some(true),
        truncation: config.truncation,
    })
}

pub fn process_rerank_response(
    response: RerankResponse,
    config: Config,
) -> Result<GolemRerankResponse, Error> {
    let mut results = Vec::new();

    for result in response.results {
        results.push(GolemRerankResult {
            index: result.index,
            relevance_score: result.relevance_score,
            document: result.document.map(|doc| doc.text),
        });
    }

    let usage = Usage {
        input_tokens: Some(response.usage.total_tokens),
        total_tokens: Some(response.usage.total_tokens),
    };

    Ok(GolemRerankResponse {
        results,
        usage: Some(usage),
        model: response.model,
        provider_metadata_json: None,
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use golem_embed::golem::embed::embed::{Config, ContentPart, TaskType};

    #[test]
    fn test_create_embedding_request() {
        let inputs = vec![
            ContentPart::Text("Hello world".to_string()),
            ContentPart::Text("How are you?".to_string()),
        ];

        let config = Config {
            model: Some("voyage-3.5-lite".to_string()),
            task_type: Some(TaskType::RetrievalDocument),
            dimensions: Some(1024),
            truncation: Some(true),
            output_format: None,
            output_dtype: None,
            user: None,
            provider_options: vec![],
        };

        let request = create_embedding_request(inputs, config).unwrap();

        assert_eq!(request.model, config.model.unwrap());
        assert_eq!(request.input_type, Some(InputType::Document));
        assert_eq!(request.truncation, config.truncation);
        assert_eq!(request.output_dimension, config.dimensions);
        assert_eq!(request.input.len(), inputs.len());
        
    }

    #[test]
    fn test_create_rerank_request() {
        let query = "What is AI?".to_string();
        let documents = vec![
            "AI is artificial intelligence".to_string(),
            "Machine learning is a subset of AI".to_string(),
        ];

        let config = Config {
            model: Some("rerank-2-lite".to_string()),
            task_type: None,
            dimensions: None,
            truncation: Some(false),
            output_format: None,
            output_dtype: None,
            user: None,
            provider_options: vec![],
        };

        let request = create_rerank_request(query, documents, config).unwrap();

        assert_eq!(request.model, "rerank-2-lite");
        assert_eq!(request.query, "What is AI?");
        assert_eq!(request.documents.len(), 2);
        assert_eq!(request.top_k, Some(5));
        assert_eq!(request.truncation, Some(false));
        assert_eq!(request.return_documents, Some(true));
    }

    #[test]
    fn test_image_input_not_supported() {
        let inputs = vec![
            ContentPart::Text("Hello".to_string()),
            ContentPart::Image(golem_embed::golem::embed::embed::ImageUrl {
                url: "http://example.com/image.jpg".to_string(),
            }),
        ];

        let config = Config {
            model: None,
            task_type: None,
            dimensions: None,
            truncation: None,
            output_format: None,
            output_dtype: None,
            user: None,
            provider_options: vec![],
        };

        let result = create_embedding_request(inputs, config);
        assert!(result.is_err());

        if let Err(error) = result {
            assert_eq!(
                error.code,
                golem_embed::golem::embed::embed::ErrorCode::Unsupported
            );
            assert!(error.message.contains("image inputs"));
        }
    }
}
