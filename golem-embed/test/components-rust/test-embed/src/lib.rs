#[allow(static_mut_refs)]
mod bindings;

use crate::bindings::exports::test::embed_exports::test_embed_api::*;
use crate::bindings::golem::embed::embed;
use crate::bindings::test::helper_client::test_helper_client::TestHelperApi;
use golem_rust::atomically;

struct Component;

#[cfg(feature = "openai")]
const MODEL: &'static str = "text-embedding-3-small";

#[cfg(feature = "openai")]
const IMAGE_MODEL: &'static str = "text-embedding-3-small";

impl Guest for Component {
    /// test1 demonstrates a simple, non-streaming text question-answer interaction with the LLM.
    fn test1() -> String {
        let config = embed::Config {
            model: Some(MODEL.to_string()),
            task_type: None,
            dimensions: None,
            truncation: None,
            output_format: None,
            output_dtype: None,
            user: None,
            provider_options: vec![],
        };
        println!("Sending text to LLM...");
        let response = embed::generate(
            &[embed::ContentPart::Text("Hello, world!".to_string())],
            &config,
        );

        match response {
            Ok(response) => {
                println!("Response: {:?}", response);
                format!("Response: {:?}", response)
            }
            Err(error) => {
                println!("Error: {:?}", error);
                format!(
                    "Error: {:?} {} {}",
                    error.code,
                    error.message,
                    error.provider_error_json.unwrap_or_default()
                )
            }
        }
    }
    fn test2() -> String {
        todo!()
    }
    fn test3() -> String {
        todo!()
    }
}

bindings::export!(Component with_types_in bindings);
