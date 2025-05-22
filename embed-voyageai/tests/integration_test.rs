use golem_embed::golem::embed::embed::{Config, ContentPart, Guest};
use golem_embed_voyageai::VoyageAIEmbedding;
use std::env;

#[test]
fn test_voyageai_embedding() {
    // Skip test if API key is not set
    match env::var("VOYAGEAI_API_KEY") {
        Ok(_) => {}, // Continue with test
        Err(_) => {
            println!("Skipping test: VOYAGEAI_API_KEY not set");
            return;
        }
    }
    
    // Create embedding instance
    let mut embedding = VoyageAIEmbedding::new();
    
    // Test inputs
    let inputs = vec![ContentPart::Text("This is a test sentence for embedding".to_string())];
    
    // Default config
    let config = Config {
        model: Some("voyage-2".to_string()),
        task_type: None,
        dimensions: None,
        truncation: Some(true),
        output_format: None,
        output_dtype: None,
        user: None,
        provider_options: vec![],
    };
    
    // Generate embeddings
    let result = embedding.generate(inputs, config);
    
    // Verify result
    assert!(result.is_ok(), "Failed to generate embeddings: {:?}", result.err());
    
    let response = result.unwrap();
    assert!(!response.embeddings.is_empty(), "No embeddings returned");
    assert!(!response.embeddings[0].vector.is_empty(), "Empty embedding vector");
}

#[test]
fn test_voyageai_rerank() {
    // Skip test if API key is not set
    match env::var("VOYAGEAI_API_KEY") {
        Ok(_) => {}, // Continue with test
        Err(_) => {
            println!("Skipping test: VOYAGEAI_API_KEY not set");
            return;
        }
    }
    
    // Create embedding instance
    let mut embedding = VoyageAIEmbedding::new();
    
    // Test inputs
    let query = "What is machine learning?".to_string();
    let documents = vec![
        "Machine learning is a field of artificial intelligence".to_string(),
        "Deep learning is a subset of machine learning".to_string(),
    ];
    
    // Default config
    let config = Config {
        model: Some("voyage-rerank-2".to_string()),
        task_type: None,
        dimensions: None,
        truncation: None,
        output_format: None,
        output_dtype: None,
        user: None,
        provider_options: vec![],
    };
    
    // Rerank documents
    let result = embedding.rerank(query, documents, config);
    
    // Verify result
    assert!(result.is_ok(), "Failed to rerank documents: {:?}", result.err());
    
    let response = result.unwrap();
    assert!(!response.results.is_empty(), "No rerank results returned");
    assert!(response.results[0].relevance_score > 0.0, "Invalid relevance score");
}