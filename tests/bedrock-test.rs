use golem_llm::golem::llm::llm::{Message, Role, ContentPart, Config};
use std::fs;

#[tokio::main]
async fn main() {
    // Load configuration
    let config_json = fs::read_to_string("tests/bedrock-test.json").unwrap();
    let config: Config = serde_json::from_str(&config_json).unwrap();

    // Create a test message
    let messages = vec![Message {
        role: Role::User,
        name: None,
        content: vec![ContentPart::Text("Hello, how are you?".to_string())],
    }];

    // Initialize Bedrock client
    let client = llm_bedrock::BedrockComponent::new(&config.provider_options).await;

    // Test chat
    match client.send(&messages, &config).await {
        Ok(response) => {
            println!("Response: {:?}", response);
        }
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
} 