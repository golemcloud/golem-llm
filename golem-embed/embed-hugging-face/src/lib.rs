mod client;
mod conversions;

use client::EmbeddingsApi;
use conversions::{create_embedding_request, process_embedding_response};
use golem_embed::{
    config::with_config_key,
    durability::{DurableEmbed, ExtendedGuest},
    golem::embed::embed::{
        Config, ContentPart, EmbeddingResponse, Error, ErrorCode, Guest, RerankResponse,
    },
    LOGGING_STATE,
};

struct HuggingFaceComponent;

impl HuggingFaceComponent {
    const ENV_VAR_NAME: &'static str = "HUGGINGFACE_API_KEY";

    fn embeddings(
        client: EmbeddingsApi,
        inputs: Vec<ContentPart>,
        config: Config,
    ) -> Result<EmbeddingResponse, Error> {
        let (request, model) = create_embedding_request(inputs, config)?;
        match client.generate_embedding(request, &model) {
            Ok(response) => process_embedding_response(response, model),
            Err(err) => Err(err),
        }
    }
}

impl Guest for HuggingFaceComponent {
    fn generate(inputs: Vec<ContentPart>, config: Config) -> Result<EmbeddingResponse, Error> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        with_config_key(Self::ENV_VAR_NAME, Err, |huggingface_api_key| {
            let client = EmbeddingsApi::new(huggingface_api_key);
            Self::embeddings(client, inputs, config)
        })
    }

    fn rerank(
        query: String,
        documents: Vec<String>,
        config: Config,
    ) -> Result<RerankResponse, Error> {
        Err(Error {
            code: ErrorCode::Unsupported,
            message: "Hugging Face inference does not support rerank".to_string(),
            provider_error_json: None,
        })
    }
}

impl ExtendedGuest for HuggingFaceComponent {}

type DurableHuggingFaceComponent = DurableEmbed<HuggingFaceComponent>;

golem_embed::export_embed!(DurableHuggingFaceComponent with_types_in golem_embed);
