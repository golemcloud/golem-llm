#[allow(static_mut_refs)]
mod bindings;

use crate::bindings::exports::test::embed_exports::test_embed_api::*;
use crate::bindings::golem::embed::embed;
use crate::bindings::golem::embed::embed::{Config, ContentPart, Error, EmbeddingResponse};
use crate::bindings::test::helper_client::test_helper_client::TestHelperApi;
use golem_rust::atomically;

struct Component;

#[cfg(feature = "openai")]
const MODEL: &'static str = "text-embedding-3-small";
#[cfg(feature = "cohere")]
const MODEL: &'static str = "embed-english-v3.0";
#[cfg(feature = "hugging-face")]
const MODEL: &'static str = "sentence-transformers/all-MiniLM-L6-v2";
#[cfg(feature = "voyageai")]
const MODEL: &'static str = "voyage-3";

#[cfg(feature = "openai")]
const RERANKING_MODEL: &'static str = "";
#[cfg(feature = "cohere")]
const RERANKING_MODEL: &'static str = "rerank-v3.5";
#[cfg(feature = "hugging-face")]
const RERANKING_MODEL: &'static str = "cross-encoder/ms-marco-MiniLM-L-2-v2";
#[cfg(feature = "voyageai")]
const RERANKING_MODEL: &'static str = "rerank-1";

impl Guest for Component {
    /// test1 demonstrates text embedding generation.
    fn test1() -> String {
        let config = Config {
            model: Some(MODEL.to_string()),
            task_type: Some(embed::TaskType::RetrievalDocument),
            dimensions: Some(1024),
            truncation: Some(true),
            output_format: Some(embed::OutputFormat::FloatArray),
            output_dtype: Some(embed::OutputDtype::FloatArray),
            user: Some("RutikThakre".to_string()),
            provider_options: vec![],
        };
        println!("Sending text for embedding generation...");
        let response: Result<EmbeddingResponse, Error> = embed::generate(
            &[ContentPart::Text("Hello, world!".to_string())],
            &config,
        );

        match response {
            Ok(response) => {
                format!("Response: {:?}", response)
            }
            Err(error) => {
                format!(
                    "Error: {:?} {} {}",
                    error.code,
                    error.message,
                    error.provider_error_json.unwrap_or_default()
                )
            }
        }
    }

    /// test2 demonstrates embedding's reranking 
    fn test2() -> String {
        let config = Config {
            model: Some(RERANKING_MODEL.to_string()),
            task_type: Some(embed::TaskType::RetrievalDocument),
            dimensions: Some(1024),
            truncation: Some(true),
            output_format: Some(embed::OutputFormat::FloatArray),
            output_dtype: Some(embed::OutputDtype::FloatArray),
            user: Some("RutikThakre".to_string()),
            provider_options: vec![],
        };
        let query = "What is the capital of the United States?";
        let documents = vec![
            "Carson City is the capital city of the American state of Nevada.".to_string(),
            "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.".to_string(),
            "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.".to_string(),
            "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.".to_string(),
            "Capital punishment has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.".to_string()
        ];

        println!("Sending request for reranking...");
        let response = embed::rerank(query, &documents, &config);
        match response {
            Ok(response) => {
                format!("Response: {:?}", response)
            }
            Err(error) => {
                format!(
                    "Error: {:?} {} {}",
                    error.code,
                    error.message,
                    error.provider_error_json.unwrap_or_default()
                )
            }
        }
    }
}

bindings::export!(Component with_types_in bindings);
