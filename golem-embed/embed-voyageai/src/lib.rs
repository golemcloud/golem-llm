use golem_embed::{
    config::with_config_key,
    durability::{DurableEmbed, ExtendedGuest},
    golem::embed::embed::{Config, ContentPart, EmbeddingResponse, Error, Guest, RerankResponse}, LOGGING_STATE,
};

use crate::{
    client::VoyageAIApi,
    conversitions::{
        create_embedding_request, create_rerank_request, process_embedding_response,
        process_rerank_response,
    },
};

mod client;
mod conversitions;

struct VoyageAIApiComponent;

impl VoyageAIApiComponent {
    const ENV_VAR_NAME: &'static str = "VOYAGEAI_API_KEY";

    fn embeddings(
        client: VoyageAIApi,
        inputs: Vec<ContentPart>,
        config: Config,
    ) -> Result<EmbeddingResponse, Error> {
        let request = create_embedding_request(inputs, config.clone());
        match request {
            Ok(request) => match client.generate_embedding(request) {
                Ok(response) => process_embedding_response(response),
                Err(err) => Err(err),
            },
            Err(err) => Err(err),
        }
    }

    fn rerank(
        client: VoyageAIApi,
        query: String,
        documents: Vec<String>,
        config: Config,
    ) -> Result<RerankResponse, Error> {
        let request = create_rerank_request(query, documents, config);
        match request {
            Ok(request) => match client.rerank(request) {
                Ok(response) => process_rerank_response(response),
                Err(err) => Err(err),
            },
            Err(err) => Err(err),
        }
    }
}

impl Guest for VoyageAIApiComponent {
    fn generate(inputs: Vec<ContentPart>, config: Config) -> Result<EmbeddingResponse, Error> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, Err, |voyageai_api_key| {
            let client = VoyageAIApi::new(voyageai_api_key);
            Self::embeddings(client, inputs, config)
        })
    }

    fn rerank(
        query: String,
        documents: Vec<String>,
        config: Config,
    ) -> Result<RerankResponse, Error> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, Err, |voyageai_api_key| {
            let client = VoyageAIApi::new(voyageai_api_key);
            Self::rerank(client, query, documents, config)
        })
    }
}

impl ExtendedGuest for VoyageAIApiComponent {}

type DurableVoyageAIApiComponent = DurableEmbed<VoyageAIApiComponent>;

golem_embed::export_embed!(DurableVoyageAIApiComponent with_types_in golem_embed);
