#[cfg(test)]
mod tests {
    use golem_llm::golem::llm::llm::{Config, ContentPart, Message, Role, Kv};
    use golem_llm::golem::llm::llm::ChatEvent;
    use std::env;
    use llm_bedrock::BedrockComponent;

    #[test]
    fn test_bedrock_chat() {
        // Set up AWS credentials from environment variables
        let access_key = env::var("AWS_ACCESS_KEY_ID").expect("AWS_ACCESS_KEY_ID must be set");
        let secret_key = env::var("AWS_SECRET_ACCESS_KEY").expect("AWS_SECRET_ACCESS_KEY must be set");
        let region = env::var("AWS_REGION").expect("AWS_REGION must be set");

        // Create test messages
        let messages = vec![Message {
            role: Role::User,
            name: None,
            content: vec![ContentPart::Text("Hello, how are you?".to_string())],
        }];

        // Create configuration
        let config = Config {
            model: "anthropic.claude-v2".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(100),
            stop_sequences: None,
            tools: vec![],
            tool_choice: None,
            provider_options: vec![
                Kv { key: "top_p".to_string(), value: "0.9".to_string() },
                Kv { key: "top_k".to_string(), value: "250".to_string() },
            ],
        };

        // Test chat functionality
        let response = BedrockComponent::send(messages.clone(), config.clone());
        match response {
            ChatEvent::Message(response) => {
                assert!(!response.content.is_empty());
                println!("Response: {:?}", response);
            }
            ChatEvent::ToolRequest(_) => panic!("Unexpected tool request"),
            ChatEvent::Error(error) => panic!("Error: {:?}", error),
        }

        // Test streaming functionality
        let stream = BedrockComponent::stream(messages, config);
        let events = stream.get_next();
        assert!(events.is_some());
        println!("Stream events: {:?}", events);
    }

    #[test]
    fn test_bedrock_error_handling() {
        // Test with invalid credentials
        env::set_var("AWS_ACCESS_KEY_ID", "invalid");
        env::set_var("AWS_SECRET_ACCESS_KEY", "invalid");
        env::set_var("AWS_REGION", "invalid-region");

        let messages = vec![Message {
            role: Role::User,
            name: None,
            content: vec![ContentPart::Text("Hello".to_string())],
        }];

        let config = Config {
            model: "anthropic.claude-v2".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(100),
            stop_sequences: None,
            tools: vec![],
            tool_choice: None,
            provider_options: vec![],
        };

        let response = BedrockComponent::send(messages, config);
        match response {
            ChatEvent::Error(error) => {
                println!("Expected error: {:?}", error);
                assert!(error.message.contains("authentication"));
            }
            _ => panic!("Expected error response"),
        }
    }

    // ... existing tests ...
} 