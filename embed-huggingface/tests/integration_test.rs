use golem_embed::golem::embed::embed::{Config, ContentPart, Guest};
use golem_embed_huggingface::HuggingFaceEmbedding;
use std::env;

#[test]
fn test_huggingface_embedding() {
    // Skip test if API key is not set
    match env::var("HUGGINGFACE_API_KEY") {
        Ok(_) => {}, // Continue with test
        Err(_) => {
            println!("Skipping test: HUGGINGFACE_API_KEY not set");
            return;
        }
    }
    
    // Create embedding instance
    let mut embedding = HuggingFaceEmbedding::new();
    
    // Test inputs
    let inputs = vec![ContentPart::Text("This is a test sentence for embedding".to_string())];
    
    // Default config
    let config = Config {
        model: Some("sentence-transformers/all-MiniLM-L6-v2".to_string()),
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
fn test_huggingface_rerank_unsupported() {
    // Skip test if API key is not set
    match env::var("HUGGINGFACE_API_KEY") {
        Ok(_) => {}, // Continue with test
        Err(_) => {
            println!("Skipping test: HUGGINGFACE_API_KEY not set");
            return;
        }
    }
    
    // Create embedding instance
    let mut embedding = HuggingFaceEmbedding::new();
    
    // Test inputs
    let query = "What is machine learning?".to_string();
    let documents = vec![
        "Machine learning is a field of artificial intelligence".to_string(),
        "Deep learning is a subset of machine learning".to_string(),
    ];
    
    // Default config
    let config = Config {
        model: None,
        task_type: None,
        dimensions: None,
        truncation: None,
        output_format: None,
        output_dtype: None,
        user: None,
        provider_options: vec![],
    };
    
    // Try to rerank documents
    let result = embedding.rerank(query, documents, config);
    
    // Verify result is an error with Unsupported code
    assert!(result.is_err(), "Expected rerank to be unsupported");
    if let Err(err) = result {
        assert_eq!(err.code, golem_embed::golem::embed::embed::ErrorCode::Unsupported);
    }
}