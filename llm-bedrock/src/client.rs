use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::primitives::Blob;
use golem_llm::golem::llm::llm::{Message, Role, ContentPart};
use serde::Deserialize;

use crate::error::BedrockError;

#[derive(Debug)]
pub struct BedrockClient {
    pub client: Client,
    pub model_id: String,
}

#[derive(Debug, Deserialize)]
pub struct BedrockResponse {
    pub completion: String,
    pub stop_reason: Option<String>,
    pub stop: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BedrockUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl BedrockClient {
    pub async fn chat(&self, messages: &[Message]) -> Result<BedrockResponse, BedrockError> {
        let prompt = Self::build_prompt(messages);
        let body = serde_json::json!({
            "prompt": prompt,
            "max_tokens_to_sample": 1000,
            "temperature": 0.7,
            "top_p": 1.0,
            "stop_sequences": ["\n\nHuman:", "\n\nAssistant:"],
        });

        let response = self.client
            .invoke_model()
            .model_id(&self.model_id)
            .body(Blob::new(serde_json::to_vec(&body)?))
            .send()
            .await?;

        let response_body = response.body.as_ref();
        let response: BedrockResponse = serde_json::from_slice(response_body)?;
        Ok(response)
    }

    fn build_prompt(messages: &[Message]) -> String {
        let mut prompt = String::new();
        for message in messages {
            match message.role {
                Role::System => {
                    prompt.push_str("\n\nHuman: System: ");
                    if let Some(ContentPart::Text(text)) = message.content.get(0) {
                        prompt.push_str(text);
                    }
                }
                Role::User => {
                    prompt.push_str("\n\nHuman: ");
                    if let Some(ContentPart::Text(text)) = message.content.get(0) {
                        prompt.push_str(text);
                    }
                }
                Role::Assistant => {
                    prompt.push_str("\n\nAssistant: ");
                    if let Some(ContentPart::Text(text)) = message.content.get(0) {
                        prompt.push_str(text);
                    }
                }
                Role::Tool => {
                    prompt.push_str("\n\nTool: ");
                    if let Some(ContentPart::Text(text)) = message.content.get(0) {
                        prompt.push_str(text);
                    }
                }
            }
        }
        prompt.push_str("\n\nAssistant: ");
        prompt
    }
} 