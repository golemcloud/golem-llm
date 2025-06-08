use client::{EmbeddingResponse, EmbeddingsApi};
use conversions::create_embed_request;
use golem_embed::{
    config::with_config_key, durability::{DurableEmbed, ExtendedGuest}, golem::embed::embed::{
        Config, ContentPart, EmbeddingResponse as GolemEmbeddingResponse, Error, Guest,
        RerankResponse,
    }, LOGGING_STATE
};

use crate::conversions::{
    create_rerank_request, process_embedding_response, process_rerank_response,
};

mod client;
mod conversions;

struct CohereComponent;

impl CohereComponent {
    const ENV_VAR_NAME: &'static str = "COHERE_API_KEY";

    fn embeddings(
        client: EmbeddingsApi,
        inputs: Vec<ContentPart>,
        config: Config,
    ) -> Result<GolemEmbeddingResponse, Error> {
        let request = create_embed_request(inputs, config.clone());
        match request {
            Ok(request) => match client.generate_embeding(request) {
                Ok(response) => process_embedding_response(response, config),
                Err(err) => Err(err),
            },
            Err(err) => Err(err),
        }
    }

    fn rerank(
        client: EmbeddingsApi,
        query: String,
        documents: Vec<String>,
        config: Config,
    ) -> Result<RerankResponse, Error> {
        let request = create_rerank_request(query, documents, config.clone());
        match request {
            Ok(request) => match client.rerank(request) {
                Ok(response) => process_rerank_response(response, config),
                Err(err) => Err(err),
            },
            Err(err) => Err(err),
        }
    }
}

impl Guest for CohereComponent {
    fn generate(inputs: Vec<ContentPart>, config: Config) -> Result<GolemEmbeddingResponse, Error> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        with_config_key(
            Self::ENV_VAR_NAME,
            |error| Err(error),
            |cohere_api_key| {
                let client = EmbeddingsApi::new(cohere_api_key);
                Self::embeddings(client, inputs, config)
            },
        )
    }

    fn rerank(
        query: String,
        documents: Vec<String>,
        config: Config,
    ) -> Result<RerankResponse, Error> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        with_config_key(
            Self::ENV_VAR_NAME,
            |error| Err(error),
            |cohere_api_key| {
                let client = EmbeddingsApi::new(cohere_api_key);
                Self::rerank(client, query, documents, config)
            },
        )
    }
}


impl ExtendedGuest for CohereComponent {}

type DurableCohereComponent = DurableEmbed<CohereComponent>;

golem_embed::export_embed!(DurableCohereComponent with_types_in golem_embed);