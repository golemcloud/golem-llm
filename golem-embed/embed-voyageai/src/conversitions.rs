use golem_embed::{
    error::unsupported,
    golem::embed::embed::{
        Config, ContentPart, Embedding, EmbeddingResponse as GolemEmbeddingResponse, Error,
        OutputDtype as GolemOutputDtype, OutputFormat as GolemOutputFormat,
        RerankResponse as GolemRerankResponse, RerankResult as GolemRerankResult, TaskType, Usage,
    },
};

use crate::client::{
    EmbeddingRequest, EmbeddingResponse, EncodingFormat, InputType, OutputDtype, RerankRequest,
    RerankResponse,
};

pub fn create_embedding_request(
    inputs: Vec<ContentPart>,
    config: Config,
) -> Result<EmbeddingRequest, Error> {
    let mut text_inputs = Vec::new();

    for input in inputs {
        match input {
            ContentPart::Text(text) => text_inputs.push(text),
            ContentPart::Image(_) => {
                return Err(unsupported(
                    "VoyageAI text embeddings do not support image inputs. Use multimodal embeddings instead.",
                ));
            }
        }
    }

    let model = config
        .model
        .unwrap_or_else(|| "voyage-3.5-lite".to_string());

    let input_type = match config.task_type {
        Some(TaskType::RetrievalQuery) => Some(InputType::Query),
        Some(TaskType::RetrievalDocument) => Some(InputType::Document),
        _ => return Err(unsupported("Unsupported task type")),
    };

    let output_dtype = match config.output_dtype {
        Some(GolemOutputDtype::FloatArray) => Some(OutputDtype::Float),
        Some(GolemOutputDtype::Int8) => Some(OutputDtype::Int8),
        Some(GolemOutputDtype::Uint8) => Some(OutputDtype::Uint8),
        Some(GolemOutputDtype::Binary) => Some(OutputDtype::Binary),
        Some(GolemOutputDtype::Ubinary) => Some(OutputDtype::Ubinary),
        _ => None,
    };

    let encoding_format = match config.output_format {
        Some(GolemOutputFormat::Base64) => Some(EncodingFormat::Base64),
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
        top_k: None,
        return_documents: Some(true),
        truncation: config.truncation,
    })
}

pub fn process_rerank_response(response: RerankResponse) -> Result<GolemRerankResponse, Error> {
    let mut results = Vec::new();
    for result in response.data {
        results.push(GolemRerankResult {
            index: result.index,
            relevance_score: result.relevance_score,
            document: result.document,
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

        let request = create_embedding_request(inputs.clone(), config.clone());
        match &request {
            Ok(request) => print!("{:?}", request),
            Err(err) => println!("{:?}", err),
        };
        assert!(request.is_ok());
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

        let request = create_rerank_request(query, documents, config);
        match &request {
            Ok(request) => print!("{:?}", &request),
            Err(err) => println!("{:?}", err),
        };
        assert!(request.is_ok());
    }
}
