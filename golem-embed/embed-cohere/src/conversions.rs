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
