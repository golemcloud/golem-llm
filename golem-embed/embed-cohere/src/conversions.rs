use std::{fs, path::Path};

use base64::{engine::general_purpose, Engine};
use golem_embed::{
    error::unsupported,
    golem::embed::embed::{
        self, Config, ContentPart, Embedding, EmbeddingResponse as GolemEmbeddingResponse, Error,
        OutputDtype, OutputFormat, TaskType, Usage,
    },
};
use log::trace;
use reqwest::{Client, Url};

use crate::client::{EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingType, InputType};

fn output_dtype_to_cohere_embedding_type(dtype: OutputDtype) -> EmbeddingType {
    match dtype {
        OutputDtype::FloatArray => EmbeddingType::Float,
        OutputDtype::Int8 => EmbeddingType::Int8,
        OutputDtype::Uint8 => EmbeddingType::Uint8,
        OutputDtype::Binary => EmbeddingType::Binary,
        OutputDtype::Ubinary => EmbeddingType::Ubinary,
    }
}

pub fn create_request(inputs: Vec<ContentPart>, config: Config) -> Result<EmbeddingRequest, Error> {
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
        output_dimension: Some(config.dimensions.unwrap().to_string()),
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
    }

    if let Some(emdeddings_array) = &response.embeddings.uint8 {
        for uint_embedding in emdeddings_array {
            let float_embedding = uint_embedding.iter().map(|&v| v as f32).collect();
            embeddings.push(Embedding {
                index: 0,
                vector: float_embedding,
            });
        }
    }
    if let Some(emdeddings_array) = &response.embeddings.binary {
        for binary_embedding in emdeddings_array {
            let float_embedding = binary_embedding.iter().map(|&v| v as f32).collect();
            embeddings.push(Embedding {
                index: 0,
                vector: float_embedding,
            });
        }
    }
    if let Some(emdeddings_array) = &response.embeddings.ubinary {
        for ubinary_embedding in emdeddings_array {
            let float_embedding = ubinary_embedding.iter().map(|&v| v as f32).collect();
            embeddings.push(Embedding {
                index: 0,
                vector: float_embedding,
            });
        }
    }
    if let Some(emdeddings_array) = &response.embeddings.float {
        for float_embedding in emdeddings_array {
            embeddings.push(Embedding {
                index: 0,
                vector: float_embedding.to_vec(),
            });
        }
    }

    Ok(GolemEmbeddingResponse {
        embeddings: embeddings,
        usage: todo!(),
        model: todo!(),
        provider_metadata_json: todo!(),
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_json;

    fn test_conversion() {
        let data = EmbeddingResponse {
            id: "54910170-852f-4322-9767-63d36e55c3bf".to_owned(),
            images: None,
            texts: Some(["This is the sentence I want to embed.", "Hey !"]),
            embeddings: EmbeddingData {
                float: Some([
                    [
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
                    [
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
                int8: Some([
                    [
                        -15, -65, 0, -31, -43, -14, -48, 59, -34, 15, 36, 49, -5, 3, -49, -34, -74,
                        21,
                    ],
                    [
                        14, 38, -30, -13, -49, 4, -33, -49, 48, 9, -84, 8, 0, -84, -46, -20, 24,
                        -26, -98, 28,
                    ],
                ]),
                uint8: None,
                binary: Some([[-54, 99, -87, 60, 15, 10, 93, 97, -42, -51, 9]]),
                ubinary: None,
            },
            meta: Some(Meta {
                api_version: Some(ApiVersion { version: Some("2") }),
                billed_units: Some(BilledUnits {
                    input_tokens: Some(11),
                }),
            }),
            response_type: Some("embeddings_by_type".to_owned()),
        };

        let result = process_embedding_response(data);
        print!("{:?}", result);
        assert!(result.is_ok())
    }
}
