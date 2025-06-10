mod client;
mod conversions;

use client::EmbeddingsApi;
use conversions::{
    create_embedding_request, create_rerank_request, process_embedding_response, 
    process_rerank_response,
};
use golem_embed::{
    config::with_config_key,
    durability::{DurableEmbed, ExtendedGuest},
    golem::embed::embed::{
        Config, ContentPart, EmbeddingResponse, Error, Guest, RerankResponse,
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

    fn rerank(
        client: EmbeddingsApi,
        query: String,
        documents: Vec<String>,
        config: Config,
    ) -> Result<RerankResponse, Error> {
        let (request, model) = create_rerank_request(query, documents, config)?;
        match client.rerank(request, &model) {
            Ok(response) => process_rerank_response(response, model),
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
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        with_config_key(Self::ENV_VAR_NAME, Err, |huggingface_api_key| {
            let client = EmbeddingsApi::new(huggingface_api_key);
            Self::rerank(client, query, documents, config)
        })
    }
}

impl ExtendedGuest for HuggingFaceComponent {}

type DurableHuggingFaceComponent = DurableEmbed<HuggingFaceComponent>;

golem_embed::export_embed!(DurableHuggingFaceComponent with_types_in golem_embed);