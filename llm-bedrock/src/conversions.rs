use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use golem_llm::golem::llm::llm::{
    Message, ContentPart, Role, Kv, ResponseMetadata, 
    FinishReason, Usage
};
use crate::client::{BedrockClient, BedrockResponse, BedrockUsage};

pub async fn create_bedrock_client(config: &[Kv]) -> BedrockClient {
    let aws_config = aws_config::defaults(BehaviorVersion::latest())
        .load()
        .await;
    let client = Client::new(&aws_config);
    
    let model_id = config.iter()
        .find(|kv| kv.key == "model_id")
        .map(|kv| kv.value.clone())
        .unwrap_or_else(|| "anthropic.claude-v2".to_string());

    BedrockClient {
        client,
        model_id,
    }
}

impl From<BedrockResponse> for Message {
    fn from(response: BedrockResponse) -> Self {
        Message {
            role: Role::Assistant,
            name: None,
            content: vec![ContentPart::Text(response.completion)],
        }
    }
}

impl From<BedrockUsage> for Usage {
    fn from(usage: BedrockUsage) -> Self {
        Self {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            total_tokens: Some(usage.input_tokens + usage.output_tokens),
        }
    }
}

impl From<BedrockResponse> for ResponseMetadata {
    fn from(_response: BedrockResponse) -> Self {
        ResponseMetadata {
            finish_reason: Some(FinishReason::Stop),
            usage: None,
            provider_id: Some("bedrock".to_string()),
            timestamp: Some(chrono::Utc::now().timestamp().to_string()),
            provider_metadata_json: None,
        }
    }
} 