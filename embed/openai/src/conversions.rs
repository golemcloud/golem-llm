use golem_embed::error::unsupported;
use golem_embed::golem::embed::embed::{
    Config, ContentPart, EmbeddingResponse as GolemEmbeddingResponse, Error,
};

use crate::client::{EmbeddingRequest, EmbeddingResponse, EncodingFormat};

pub fn create_request(inputs: Vec<ContentPart>, config: Config) -> Result<EmbeddingRequest, Error> {
    let mut input = String::new();
    for content in inputs {
        match content {
            ContentPart::Text(text) => input.push_str(&text),
            ContentPart::Image(_) => {
                return Err(unsupported("Image embeddings is not supported by OpenAI."))
            }
        }
    }

    let model = config
        .model
        .unwrap_or_else(|| "text-embedding-ada-002".to_string());

    let encoding_format = match config.output_format {
        Some(golem_embed::golem::embed::embed::OutputFormat::FloatArray) => {
            Some(EncodingFormat::Float)
        }
        Some(golem_embed::golem::embed::embed::OutputFormat::Base64) => {
            Some(EncodingFormat::Base64)
        }
        _ => {
            return Err(unsupported(
                "OpenAI only supports float and base64 output formats.",
            ))
        }
    };

    Ok(EmbeddingRequest {
        input,
        model,
        encoding_format,
        dimension: config.dimensions,
        user: config.user,
    })
}

pub fn process_embedding_response(
    response: EmbeddingResponse,
) -> Result<GolemEmbeddingResponse, Error> {
    let mut embeddings = Vec::new();
    for embeding_data in &response.data {
        let embed = embeding_data.embedding.to_float_vec().map_err(|e| Error {
            code: golem_embed::golem::embed::embed::ErrorCode::ProviderError,
            message: e,
            provider_error_json: None,
        })?;
        embeddings.push(golem_embed::golem::embed::embed::Embedding {
            index: embeding_data.index as u32,
            vector: embed,
        });
    }

    let usage = golem_embed::golem::embed::embed::Usage {
        input_tokens: Some(response.usage.prompt_tokens),
        total_tokens: Some(response.usage.total_tokens),
    };

    Ok(GolemEmbeddingResponse {
        embeddings,
        usage: Some(usage),
        model: response.model,
        provider_metadata_json: None,
    })
}

#[cfg(test)]
mod tests {
    use golem_embed::golem::embed::embed::{ImageUrl, OutputDtype, OutputFormat, TaskType};

    use crate::client::{EmbeddingData, EmbeddingUsage, EmbeddingVector};

    use super::*;

    #[test]
    fn test_process_embedding_response() {
        let response = EmbeddingResponse {
            data: vec![EmbeddingData {
                embedding: EmbeddingVector::FloatArray(vec![0.1, 0.2, 0.3]),
                index: 0,
                object: "embedding".to_string(),
            }],
            model: "text-embedding-ada-002".to_string(),
            usage: EmbeddingUsage {
                prompt_tokens: 1,
                total_tokens: 1,
            },
            object: "list".to_string(),
        };
        let result = process_embedding_response(response);
        let embedding_response = result.unwrap();
        assert_eq!(embedding_response.embeddings.len(), 1);
        assert_eq!(embedding_response.embeddings[0].index, 0);
        assert_eq!(embedding_response.embeddings[0].vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(embedding_response.provider_metadata_json, None);
    }

    #[test]
    fn test_create_request() {
        let inputs = vec![ContentPart::Text("Hello, world!".to_string())];
        let config = Config {
            model: Some("text-embedding-ada-002".to_string()),
            dimensions: Some(1536),
            user: Some("test_user".to_string()),
            task_type: Some(TaskType::RetrievalQuery),
            truncation: Some(false),
            output_format: Some(OutputFormat::FloatArray),
            output_dtype: Some(OutputDtype::FloatArray),
            provider_options: vec![],
        };
        let request = create_request(inputs, config);
        let embedding_request = request.unwrap();
        assert_eq!(embedding_request.input, "Hello, world!");
        assert_eq!(embedding_request.model, "text-embedding-ada-002");
        assert_eq!(embedding_request.dimension, Some(1536));
        assert_eq!(embedding_request.user, Some("test_user".to_string()));
        assert_eq!(
            embedding_request.encoding_format,
            Some(EncodingFormat::Float)
        );
    }

    #[test]
    fn test_create_request_with_image() {
        // OpenAI does not support image embeddings so this should return an error
        let inputs = vec![ContentPart::Image(ImageUrl {
            url: "https://example.com/image.png".to_string(),
        })];
        let config = Config {
            model: Some("text-embedding-ada-002".to_string()),
            dimensions: Some(1536),
            user: Some("test_user".to_string()),
            task_type: Some(TaskType::RetrievalQuery),
            truncation: Some(false),
            output_format: Some(OutputFormat::FloatArray),
            output_dtype: Some(OutputDtype::FloatArray),
            provider_options: vec![],
        };
        let request = create_request(inputs, config);
        assert!(request.is_err());
    }
}
