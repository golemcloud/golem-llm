use client::{EmbeddingResponse, EmbeddingsApi};
use conversions::create_request;
use golem_embed::{
    config::with_config_key,
    golem::embed::embed::{
        Config, ContentPart, EmbeddingResponse as GolemEmbeddingResponse, Error, Guest,
        RerankResponse,
    },
    LOGGING_STATE,
};

use crate::conversions::process_embedding_response;

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
        let request = create_request(inputs, config.clone());
        match request {
            Ok(request) => match client.generate_embeding(request) {
                Ok(response) => process_embedding_response(response, config),
                Err(err) => Err(err),
            },
            Err(err) => Err(err),
        }
    }

    fn rerank(
        query: String,
        documents: Vec<String>,
        config: Config,
    ) -> Result<RerankResponse, Error> {
        todo!()
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
        todo!()
    }
}
