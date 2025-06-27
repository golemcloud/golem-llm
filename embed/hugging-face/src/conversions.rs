use golem_embed::error::unsupported;
use golem_embed::golem::embed::embed::{
    Config, ContentPart, EmbeddingResponse as GolemEmbeddingResponse, Error,
};

use crate::client::{EmbeddingRequest, EmbeddingResponse};

pub fn create_embedding_request(
    inputs: Vec<ContentPart>,
    config: Config,
) -> Result<(EmbeddingRequest, String), Error> {
    let mut input_texts = Vec::new();
    for content in inputs {
        match content {
            ContentPart::Text(text) => input_texts.push(text),
            ContentPart::Image(_) => {
                return Err(unsupported(
                    "Image embeddings are not supported by Hugging Face.",
                ))
            }
        }
    }

    let model = config
        .model
        .unwrap_or_else(|| "sentence-transformers/all-MiniLM-L6-v2".to_string());

    let request = EmbeddingRequest {
        inputs: input_texts,
        normalize: Some(true),
        prompt_name: None,
        truncate: config.truncation,
        truncate_direction: None,
    };

    Ok((request, model))
}

pub fn process_embedding_response(
    response: EmbeddingResponse,
    model: String,
) -> Result<GolemEmbeddingResponse, Error> {
    let mut embeddings = Vec::new();
    for (index, embedding_vec) in response.iter().enumerate() {
        embeddings.push(golem_embed::golem::embed::embed::Embedding {
            index: index as u32,
            vector: embedding_vec.clone(),
        });
    }

    Ok(GolemEmbeddingResponse {
        embeddings,
        usage: None,
        model,
        provider_metadata_json: None,
    })
}

#[cfg(test)]
mod tests {
    use golem_embed::golem::embed::embed::{ImageUrl, OutputDtype, OutputFormat, TaskType};

    use super::*;

    #[test]
    fn test_create_embedding_request() {
        let inputs = vec![ContentPart::Text("Hello, world!".to_string())];
        let config = Config {
            model: Some("sentence-transformers/all-MiniLM-L6-v2".to_string()),
            dimensions: Some(384),
            user: Some("test_user".to_string()),
            task_type: Some(TaskType::RetrievalQuery),
            truncation: Some(false),
            output_format: Some(OutputFormat::FloatArray),
            output_dtype: Some(OutputDtype::FloatArray),
            provider_options: vec![],
        };
        let result = create_embedding_request(inputs, config);
        let (request, model) = result.unwrap();
        assert_eq!(request.inputs, vec!["Hello, world!"]);
        assert_eq!(model, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(request.normalize, Some(true));
        assert_eq!(request.truncate, Some(false));
    }

    #[test]
    fn test_process_embedding_response() {
        let response: EmbeddingResponse = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let result = process_embedding_response(response, model.clone());
        let embedding_response = result.unwrap();
        assert_eq!(embedding_response.embeddings.len(), 2);
        assert_eq!(embedding_response.embeddings[0].index, 0);
        assert_eq!(embedding_response.embeddings[0].vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(embedding_response.embeddings[1].index, 1);
        assert_eq!(embedding_response.embeddings[1].vector, vec![0.4, 0.5, 0.6]);
        assert_eq!(embedding_response.model, model);
        assert!(embedding_response.usage.is_none());
    }

    #[test]
    fn test_create_embedding_request_with_image() {
        let inputs = vec![ContentPart::Image(ImageUrl {
            url: "https://example.com/image.png".to_string(),
        })];
        let config = Config {
            model: Some("sentence-transformers/all-MiniLM-L6-v2".to_string()),
            dimensions: Some(384),
            user: Some("test_user".to_string()),
            task_type: Some(TaskType::RetrievalQuery),
            truncation: Some(false),
            output_format: Some(OutputFormat::FloatArray),
            output_dtype: Some(OutputDtype::FloatArray),
            provider_options: vec![],
        };
        let request = create_embedding_request(inputs, config);
        assert!(request.is_err());
    }
}
