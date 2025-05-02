use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::primitives::Blob;
use dotenv::dotenv;
use serde_json::json;

#[tokio::main]
async fn main() {
    dotenv().ok();

    let aws_config = aws_config::defaults(BehaviorVersion::latest())
        .load()
        .await;
    let client = Client::new(&aws_config);

    let body = json!({
        "prompt": "\n\nHuman: Hello, how are you?\n\nAssistant: ",
        "max_tokens_to_sample": 1000,
        "temperature": 0.7,
        "top_p": 1.0,
        "stop_sequences": ["\n\nHuman:", "\n\nAssistant:"],
    });

    let response = client
        .invoke_model()
        .model_id("anthropic.claude-v2")
        .body(Blob::new(serde_json::to_vec(&body).unwrap()))
        .send()
        .await;

    match response {
        Ok(response) => {
            let response_body = response.body.as_ref();
            let response_str = String::from_utf8_lossy(response_body);
            println!("Response: {}", response_str);
        }
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
} 