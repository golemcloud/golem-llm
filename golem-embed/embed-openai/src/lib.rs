use client::EmbeddingsApi;
use conversions::{create_request, process_embedding_response};
use golem_embed::{config::with_config_key, golem::embed::embed::{
    Config, ContentPart, EmbeddingResponse, Error, Guest, RerankResponse
}, LOGGING_STATE};

mod client;
mod conversions;

struct OpenAIComponent;

impl OpenAIComponent {
    const ENV_VAR_NAME: &'static str = "OPENAI_API_KEY";

    fn embeddings(
        client: EmbeddingsApi,
        inputs: Vec<ContentPart>,
        config: Config,
    ) -> Result<EmbeddingResponse, Error> {
        let request = create_request(inputs, config);
        match request {
            Ok(request) => match client.generate_embeding(request) {
                Ok(response) => process_embedding_response(response),
                Err(err) => Err(err),
            },
            Err(err) => Err(err),
        }
    }


    
}



impl Guest for OpenAIComponent {
    fn generate(inputs: Vec<ContentPart>,config:Config,) -> Result<EmbeddingResponse,Error> {
     LOGGING_STATE.with_borrow_mut(|state|state.init());
      with_config_key(
            Self::ENV_VAR_NAME,
            |error|Err(error),
            |openai_api_key| {
                let client = EmbeddingsApi::new(openai_api_key);
                Self::embeddings(client, inputs, config)
            },
        )
   
    }
    
    fn rerank(query:String,documents:Vec::<String>,config:Config,) -> Result<RerankResponse,Error> {
        todo!()
    } 
}

golem_embed::export_embed!(OpenAIComponent with_types_in golem_embed);
