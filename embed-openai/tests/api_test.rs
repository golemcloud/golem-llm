use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: u32,
    object: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    model: String,
    object: String,
    usage: EmbeddingUsage,
}

#[tokio::test]
async fn test_openai_api_directly() {
    // Load environment variables from .env file
    dotenv::dotenv().ok();
    
    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    
    // Create HTTP client
    let client = reqwest::Client::new();
    
    // Set up headers
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap(),
    );
    
    // Create request payload
    let request = EmbeddingRequest {
        model: "text-embedding-3-small".to_string(),
        input: vec!["This is a test sentence for OpenAI embeddings.".to_string()],
    };
    
    // Send request to OpenAI API
    let response = client
        .post("https://api.openai.com/v1/embeddings")
        .headers(headers)
        .json(&request)
        .send()
        .await
        .expect("Failed to send request");
    
    // Check response status
    assert!(response.status().is_success(), 
           "API request failed with status: {}", response.status());
    
    // Parse response
    let embedding_response = response
        .json::<EmbeddingResponse>()
        .await
        .expect("Failed to parse response");
    
    // Verify response
    println!("Successfully generated embeddings with model: {}", embedding_response.model);
    println!("Number of embeddings: {}", embedding_response.data.len());
    println!("First embedding vector length: {}", embedding_response.data[0].embedding.len());
    
    // Basic assertions
    assert!(!embedding_response.data.is_empty(), "Should have at least one embedding");
    assert!(!embedding_response.data[0].embedding.is_empty(), "Embedding vector should not be empty");
    
    // Check usage information
    println!("Input tokens: {}", embedding_response.usage.prompt_tokens);
    println!("Total tokens: {}", embedding_response.usage.total_tokens);
}