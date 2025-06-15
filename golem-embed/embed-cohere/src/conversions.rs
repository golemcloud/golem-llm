use std::{fs, path::Path};

use base64::{engine::general_purpose, Engine};
use golem_embed::{
    error::unsupported,
    golem::embed::embed::{
        Config, ContentPart, Embedding, EmbeddingResponse as GolemEmbeddingResponse, Error,
        OutputDtype, RerankResponse as GolemRerankResponse, RerankResult, TaskType, Usage,
    },
};
use log::trace;
use reqwest::{Client, Url};

use crate::client::{
    EmbeddingRequest, EmbeddingResponse, EmbeddingType, InputType, RerankRequest, RerankResponse,
};

fn output_dtype_to_cohere_embedding_type(dtype: OutputDtype) -> EmbeddingType {
    match dtype {
        OutputDtype::FloatArray => EmbeddingType::Float,
        OutputDtype::Int8 => EmbeddingType::Int8,
        OutputDtype::Uint8 => EmbeddingType::Uint8,
        OutputDtype::Binary => EmbeddingType::Binary,
        OutputDtype::Ubinary => EmbeddingType::Ubinary,
    }
}

pub fn create_embed_request(
    inputs: Vec<ContentPart>,
    config: Config,
) -> Result<EmbeddingRequest, Error> {
    let mut text_inputs = Vec::new();
    let mut image_inputs = Vec::new();
    for input in inputs {
        match input {
            ContentPart::Text(text) => text_inputs.push(text),
            ContentPart::Image(image) => match image_to_base64(&image.url) {
                Ok(base64_data) => image_inputs.push(base64_data),
                Err(err) => {
                    trace!("Failed to encode image: {}\nError: {}\n", image.url, err);
                }
            },
        }
    }

    if !text_inputs.is_empty() && !image_inputs.is_empty()
        || text_inputs.is_empty() && image_inputs.is_empty()
    {
        return Err(unsupported(
            "Cohere requires text or image input, not both.",
        ));
    }

    let input_type = if !image_inputs.is_empty() && text_inputs.is_empty() {
        InputType::Image
    } else {
        config
            .task_type
            .map(|task_type| match task_type {
                TaskType::RetrievalQuery => InputType::SearchQuery,
                TaskType::RetrievalDocument => InputType::SearchDocument,
                TaskType::Classification => InputType::Classification,
                TaskType::Clustering => InputType::Clustering,
                _ => InputType::SearchDocument,
            })
            .unwrap()
    };

    let model = config
        .model
        .unwrap_or_else(|| "embed-english-v3.0".to_string());

    let embedding_types = config
        .output_dtype
        .map(|dtype| vec![output_dtype_to_cohere_embedding_type(dtype)]);

    Ok(EmbeddingRequest {
        model,
        input_type,
        embedding_types,
        images: if !image_inputs.is_empty() {
            Some(image_inputs)
        } else {
            None
        },
        texts: if !text_inputs.is_empty() {
            Some(text_inputs)
        } else {
            None
        },
        truncate: None,
        max_tokens: None,
        output_dimension: Some(config.dimensions.unwrap()),
    })
}

pub fn create_rerank_request(
    query: String,
    documents: Vec<String>,
    config: Config,
) -> Result<RerankRequest, Error> {
    let model = config.model.unwrap_or_else(|| "rerank-2-lite".to_string());
    Ok(RerankRequest {
        model,
        query,
        documents,
        top_n: None,
        max_tokens_per_doc: None,
    })
}

pub fn image_to_base64(source: &str) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = if Url::parse(source).is_ok() {
        let client = Client::new();
        let response = client.get(source).send()?;

        response.bytes()?.to_vec()
    } else {
        let path = Path::new(source);

        fs::read(path)?
    };

    let base64_data = general_purpose::STANDARD.encode(&bytes);
    Ok(base64_data)
}

pub fn process_embedding_response(
    response: EmbeddingResponse,
    config: Config,
) -> Result<GolemEmbeddingResponse, Error> {
    let mut embeddings: Vec<Embedding> = Vec::new();
    if let Some(emdeddings_array) = &response.embeddings.int8 {
        for int_embedding in emdeddings_array {
            let float_embedding = int_embedding.iter().map(|&v| v as f32).collect();
            embeddings.push(Embedding {
                index: 0,
                vector: float_embedding,
            });
        }
    };

    if let Some(emdeddings_array) = &response.embeddings.uint8 {
        for uint_embedding in emdeddings_array {
            let float_embedding = uint_embedding.iter().map(|&v| v as f32).collect();
            embeddings.push(Embedding {
                index: 0,
                vector: float_embedding,
            });
        }
    };
    if let Some(emdeddings_array) = &response.embeddings.binary {
        for binary_embedding in emdeddings_array {
            let float_embedding = binary_embedding.iter().map(|&v| v as f32).collect();
            embeddings.push(Embedding {
                index: 0,
                vector: float_embedding,
            });
        }
    };
    if let Some(emdeddings_array) = &response.embeddings.ubinary {
        for ubinary_embedding in emdeddings_array {
            let float_embedding = ubinary_embedding.iter().map(|&v| v as f32).collect();
            embeddings.push(Embedding {
                index: 0,
                vector: float_embedding,
            });
        }
    };
    if let Some(emdeddings_array) = &response.embeddings.float {
        for float_embedding in emdeddings_array {
            embeddings.push(Embedding {
                index: 0,
                vector: float_embedding.to_vec(),
            });
        }
    };

    Ok(GolemEmbeddingResponse {
        embeddings,
        provider_metadata_json: Some(get_embed_provider_metadata(response.clone())),
        model: config
            .model
            .unwrap_or_else(|| "embed-english-v3.0".to_string()),
        usage: Some(Usage {
            input_tokens: response.meta.unwrap().billed_units.unwrap().input_tokens,
            total_tokens: None,
        }),
    })
}

pub fn get_embed_provider_metadata(response: EmbeddingResponse) -> String {
    let meta = serde_json::to_string(&response.meta.unwrap()).unwrap_or_default();
    format!(r#"{{"id":"{}","meta":"{}",}}"#, response.id, meta)
}

pub fn process_rerank_response(
    response: RerankResponse,
    config: Config,
) -> Result<GolemRerankResponse, Error> {
    let results = response
        .clone()
        .results
        .iter()
        .map(|result| RerankResult {
            index: result.index,
            relevance_score: result.relevance_score,
            document: None,
        })
        .collect();

    let usage = if let Some(meta) = response.clone().meta {
        if let Some(billed_units) = meta.billed_units {
            Some(Usage {
                input_tokens: billed_units.input_tokens,
                total_tokens: billed_units.output_tokens,
            })
        } else {
            None
        }
    } else {
        None
    };

    Ok(GolemRerankResponse {
        results,
        usage,
        model: config.model.unwrap_or_else(|| "rerank-2-lite".to_string()),
        provider_metadata_json: Some(get_rerank_provider_metadata(response)),
    })
}

fn get_rerank_provider_metadata(response: RerankResponse) -> String {
    let meta = serde_json::to_string(&response.meta.unwrap()).unwrap_or_default();
    format!(
        r#"{{"id":"{}","meta":"{}",}}"#,
        response.id.unwrap_or_default(),
        meta
    )
}

#[cfg(test)]
mod tests {
    use crate::client::{ApiVersion, BilledUnits, EmbeddingData, Meta, RerankData};

    use super::*;

    #[test]
    fn test_create_embed_request() {
        let inputs = vec![ContentPart::Text("Hello, world!".to_string())];
        let config = Config {
            model: Some("embed-english-v3.0".to_string()),
            task_type: Some(TaskType::RetrievalQuery),
            dimensions: Some(1024),
            truncation: Some(true),
            output_format: None,
            output_dtype: None,
            user: None,
            provider_options: vec![],
        };
        let request = create_embed_request(inputs, config);
        let request = request.unwrap();
        assert_eq!(request.model, "embed-english-v3.0");
        // assert_eq!(request.input_type, InputType::SearchQuery);
        // assert_eq!(request.embedding_types, Some(vec![EmbeddingType::Float]));
        assert_eq!(request.images, None);
        assert_eq!(request.texts, Some(vec!["Hello, world!".to_string()]));
        // assert_eq!(request.truncate, None);
        assert_eq!(request.max_tokens, None);
        assert_eq!(request.output_dimension, Some(1024));
    }

    #[test]
    fn test_embedding_response_conversion() {
        let data = EmbeddingResponse {
            id: "54910170-852f-4322-9767-63d36e55c3bf".to_owned(),
            images: None,
            texts: Some(vec![
                "This is the sentence I want to embed.".to_owned(),
                "Hey !".to_owned(),
            ]),
            embeddings: EmbeddingData {
                float: Some(vec![
                    vec![
                        0.016967773,
                        0.031982422,
                        0.041503906,
                        0.0021514893,
                        0.008178711,
                        -0.029541016,
                        -0.018432617,
                        -0.046875,
                        0.021240234,
                    ],
                    vec![
                        0.013977051,
                        0.012084961,
                        0.005554199,
                        -0.053955078,
                        -0.026977539,
                        -0.008361816,
                        0.02368164,
                        -0.013183594,
                        -0.063964844,
                        0.026611328,
                    ],
                ]),
                int8: Some(vec![
                    vec![
                        -15, -65, 0, -31, -43, -14, -48, 59, -34, 15, 36, 49, -5, 3, -49, -34, -74,
                        21,
                    ],
                    vec![
                        14, 38, -30, -13, -49, 4, -33, -49, 48, 9, -84, 8, 0, -84, -46, -20, 24,
                        -26, -98, 28,
                    ],
                ]),
                uint8: None,
                binary: Some(vec![vec![-54, 99, -87, 60, 15, 10, 93, 97, -42, -51, 9]]),
                ubinary: None,
            },
            meta: Some(Meta {
                api_version: Some(ApiVersion {
                    version: Some("2".to_owned()),
                    is_experimental: None,
                    is_deprecated: None,
                }),
                billed_units: Some(BilledUnits {
                    input_tokens: Some(11),
                    classifications: None,
                    images: None,
                    output_tokens: None,
                    search_units: None,
                }),
                tokens: None,
                warning: None,
            }),
            response_type: Some("embeddings_by_type".to_owned()),
        };

        let config = Config {
            model: Some("embed-english-v3.0".to_string()),
            task_type: None,
            dimensions: None,
            truncation: None,
            output_format: None,
            output_dtype: None,
            user: None,
            provider_options: vec![],
        };

        let result = process_embedding_response(data.clone(), config);
        print!("{:?}", result);
        let embedding_response = result.unwrap();
        assert_eq!(embedding_response.embeddings.len(), 5);
        assert_eq!(embedding_response.embeddings[0].index, 0);
        assert_eq!(
            embedding_response.embeddings[0].vector,
            vec![
                -15.0, -65.0, 0.0, -31.0, -43.0, -14.0, -48.0, 59.0, -34.0, 15.0, 36.0, 49.0, -5.0, 3.0, -49.0, -34.0, -74.0, 21.0
            ]
        );
        assert_eq!(embedding_response.embeddings[1].index, 0);
        assert_eq!(
            embedding_response.embeddings[1].vector,
            vec![
                14.0, 38.0, -30.0, -13.0, -49.0, 4.0, -33.0, -49.0, 48.0, 9.0, -84.0, 8.0, 0.0, -84.0, -46.0, -20.0, 24.0, -26.0, -98.0, 28.0
            ]
        );
        assert_eq!(
            embedding_response.provider_metadata_json,
            Some(get_embed_provider_metadata(data))
        );
        assert_eq!(embedding_response.model, "embed-english-v3.0");
        assert_eq!(
            embedding_response.usage,
            Some(Usage {
                input_tokens: Some(11),
                total_tokens: None,
            })
        );
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
            truncation: None,
            output_format: None,
            output_dtype: None,
            user: None,
            provider_options: vec![],
        };
        let request = create_rerank_request(query, documents.clone(), config);
        let request = request.unwrap();
        assert_eq!(request.model, "rerank-2-lite");
        assert_eq!(request.query, "What is AI?");
        assert_eq!(request.documents, documents);
        assert_eq!(request.top_n, None);
        assert_eq!(request.max_tokens_per_doc, None);
    }

    #[test]
    fn test_rerank_response_conversion() {
        let data = RerankResponse {
            id: Some("54910170-852f-4322-9767-63d36e55c3bf".to_owned()),
            results: vec![RerankData {
                index: 0,
                relevance_score: 0.9,
            }],
            meta: Some(Meta {
                api_version: Some(ApiVersion {
                    version: Some("2".to_owned()),
                    is_experimental: None,
                    is_deprecated: None,
                }),
                billed_units: Some(BilledUnits {
                    input_tokens: Some(11),
                    classifications: None,
                    images: None,
                    output_tokens: Some(111),
                    search_units: None,
                }),
                tokens: None,
                warning: None,
            }),
        };

        let config = Config {
            model: Some("rerank-2-lite".to_string()),
            task_type: None,
            dimensions: None,
            truncation: None,
            output_format: None,
            output_dtype: None,
            user: None,
            provider_options: vec![],
        };
        let result = process_rerank_response(data.clone(), config);
        let rerank_response = result.unwrap();
        assert_eq!(rerank_response.results.len(), 1);
        assert_eq!(rerank_response.results[0].index, 0);
        assert_eq!(rerank_response.results[0].relevance_score, 0.9);
        assert_eq!(rerank_response.model, "rerank-2-lite");
        assert_eq!(
            rerank_response.provider_metadata_json,
            Some(get_rerank_provider_metadata(data))
        );
        assert_eq!(
            rerank_response.usage,
            Some(Usage {
                input_tokens: Some(11),
                total_tokens: Some(111),
            })  
        );
    }
}
