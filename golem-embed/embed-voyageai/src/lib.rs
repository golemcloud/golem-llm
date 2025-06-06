use golem_embed::golem::embed::embed::{EmbeddingResponse, RerankResponse, ContentPart, Config, Error, Guest};

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
                Ok(response) => process_embedding_response(response, config),
                Err(err) => Err(err),
            },
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


impl Guest for VoyageAIApiComponent {
    fn generate(inputs: Vec<ContentPart>, config: Config) -> Result<EmbeddingResponse, Error> {
        let client = VoyageAIApi::new(config.api_key.clone());
        Self::embeddings(client, inputs, config)
    }

    fn rerank(
        query: String,
        documents: Vec<String>,
        config: Config,
    ) -> Result<RerankResponse, Error> {
        todo!()
    }
}

