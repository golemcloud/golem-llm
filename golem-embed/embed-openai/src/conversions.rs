use bytemuck::checked::cast_slice;
use golem_embed::golem::embed::embed::{Config, ContentPart, EmbeddingResponse as GolemEmbeddingResponse, Error};
use golem_embed::error::unsupported;

use crate::client::{EmbeddingRequest, EmbeddingResponse, EncodingFormat};






pub fn create_request(inputs: Vec<ContentPart>, config: Config) -> Result<EmbeddingRequest, Error> {
    let mut input = String::new();
    for content in inputs {
        match content {
            ContentPart::Text(text) => input.push_str(&text),
            ContentPart::Image(_) =>
             return Err(unsupported("Image embeddings is not supported by OpenAI.")),
        }
    }


    let model = config.model.unwrap_or_else(|| "text-embedding-ada-002".to_string());

    let encoding_format = config.output_format.map(|format| match format {
        golem_embed::golem::embed::embed::OutputFormat::FloatArray => EncodingFormat::Float,
        golem_embed::golem::embed::embed::OutputFormat::Binary => EncodingFormat::Base64,
        golem_embed::golem::embed::embed::OutputFormat::Base64 => EncodingFormat::Base64,
    });

    let dimension = config.dimensions;

    let user = config.user;

    Ok(EmbeddingRequest {
        input,
        model,
        encoding_format,
        dimension,
        user,
    })

}



fn f32vec_to_binary(f32_vec: &[f32]) -> &[u8] {
    cast_slice(f32_vec)
}

pub fn process_embedding_response(response: EmbeddingResponse) -> Result<GolemEmbeddingResponse, Error> {
    let mut embeddings = Vec::new();
    for embeding_data in &response.data{
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


    Ok(GolemEmbeddingResponse{
        embeddings,
        usage : Some(usage),
        model : response.model,
        provider_metadata_json: None,
    })

}
