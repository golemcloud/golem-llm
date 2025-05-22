# Golem Embeddings

WebAssembly Components providing a unified API for various embedding providers. This repository implements embedding functionality for multiple providers, allowing you to generate vector embeddings for text using different services.

## Supported Embedding Providers

| Provider | Status | API Reference |
|----------|--------|---------------|
| OpenAI | ✅ Implemented | [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) |
| Cohere | ✅ Implemented | [Cohere Embed API](https://docs.cohere.com/reference/embed) |
| Voyage AI | ✅ Implemented | [Voyage AI API](https://docs.voyageai.com/reference/embeddings) |
| Hugging Face | ✅ Implemented | [Hugging Face API](https://huggingface.co/docs/api-inference/detailed_parameters#feature-extraction-task) |

## Installation

To use the embedding functionality, you need to set up the appropriate API keys for the providers you want to use.

### API Keys Configuration

Create a `.env` file in your project root with the following variables (add only the ones you need):

```
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
VOYAGEAI_API_KEY=your_voyageai_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

## Usage Examples

### OpenAI Embeddings

```rust
use embed_openai::client::{create_embeddings, EmbeddingRequest};
use dotenv::dotenv;
use std::env;

async fn generate_embeddings() {
    // Load environment variables from .env file
    dotenv().ok();
    
    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    
    // Create request payload
    let request = EmbeddingRequest {
        model: "text-embedding-3-small".to_string(),
        input: vec!["This is a test sentence for OpenAI embeddings.".to_string()],
    };
    
    // Generate embeddings
    let response = create_embeddings(&api_key, None, &request).await.expect("Failed to generate embeddings");
    
    // Use the embeddings
    println!("Model: {}", response.model);
    println!("Number of embeddings: {}", response.data.len());
    println!("Embedding dimensions: {}", response.data[0].embedding.len());
}
```

### Voyage AI Embeddings

```rust
use embed_voyageai::client::{create_embeddings, EmbeddingRequest};
use dotenv::dotenv;
use std::env;

async fn generate_embeddings() {
    // Load environment variables from .env file
    dotenv().ok();
    
    // Get API key from environment
    let api_key = env::var("VOYAGEAI_API_KEY").expect("VOYAGEAI_API_KEY must be set");
    
    // Create request payload
    let request = EmbeddingRequest {
        model: "voyage-2".to_string(),
        input: vec!["This is a test sentence for Voyage AI embeddings.".to_string()],
    };
    
    // Generate embeddings
    let response = create_embeddings(&api_key, &request).await.expect("Failed to generate embeddings");
    
    // Use the embeddings
    println!("Model: {}", response.model);
    println!("Number of embeddings: {}", response.data.len());
    println!("Embedding dimensions: {}", response.data[0].embedding.len());
}
```

## Testing

You can test the embedding functionality using the provided test scripts. For example, to test OpenAI embeddings:

### Python Test Script

The repository includes a Python test script (`test_openai_embeddings.py`) that demonstrates how to use the OpenAI embeddings API:

```python
python test_openai_embeddings.py
```

Example output:

```
Using API key: sk-ab...wxyz

✅ API request successful!
Model: text-embedding-3-small
Number of embeddings: 1
Embedding dimensions: 1536
Usage - Prompt tokens: 8
Usage - Total tokens: 8

The OpenAI API key is working correctly with the embeddings service.
This confirms that the bounty issue can be solved with this API key.
```

### Rust Integration Tests

The repository also includes Rust integration tests for each provider. To run the OpenAI tests:

```bash
cd embed-openai
cargo test
```

Example output from the OpenAI API test:

```
Successfully generated embeddings with model: text-embedding-3-small
Number of embeddings: 1
First embedding vector length: 1536
Input tokens: 8
Total tokens: 8
```

## Implementation Details

Each embedding provider is implemented as a separate Rust crate with a consistent API structure:

- `client.rs`: Contains the HTTP client implementation for making API requests
- `conversions.rs`: Handles data type conversions between the API and internal representations
- `lib.rs`: Exports the public API

All providers implement a common error handling pattern and return consistent response structures, making it easy to switch between different embedding services.

## License

This project is licensed under the same terms as the main Golem LLM project.