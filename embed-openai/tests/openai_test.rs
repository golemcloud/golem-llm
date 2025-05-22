use golem_embed::golem::embed::embed::{Config, ContentPart, Guest};
use std::env;

#[test]
fn test_openai_embeddings() {
    // Load environment variables from .env file
    dotenv::dotenv().ok();
    
    // Create an instance of OpenAIEmbedding
    let mut embedding = golem_embed_openai::OpenAIEmbedding::new();
    
    // Prepare test inputs
    let inputs = vec![
        ContentPart::Text("This is a test sentence for OpenAI embeddings.".to_string()),
    ];
    
    // Create a basic config
    let config = Config {
        model: Some("text-embedding-3-small".to_string()), // Using a smaller model for testing
        task_type: None,
        dimensions: None,
        truncation: None,
        output_format: None,
        output_dtype: None,
        user: None,
        provider_options: vec![],
    };
    
    // Generate embeddings
    let result = embedding.generate(inputs, config);
    
    // Verify the result
    match result {
        Ok(response) => {
            println!("Successfully generated embeddings with model: {}", response.model);
            println!("Number of embeddings: {}", response.embeddings.len());
            println!("First embedding vector length: {}", response.embeddings[0].vector.len());
            
            // Basic assertions
            assert!(!response.embeddings.is_empty(), "Should have at least one embedding");
            assert!(!response.embeddings[0].vector.is_empty(), "Embedding vector should not be empty");
            
            // Check usage information
            if let Some(usage) = response.usage {
                println!("Input tokens: {:?}", usage.input_tokens);
                println!("Total tokens: {:?}", usage.total_tokens);
            }
        },
        Err(err) => {
            panic!("Failed to generate embeddings: {:?}", err);
        }
    }
}