// Placeholder for Bedrock API client logic

// TODO: Implement functions to interact with AWS Bedrock API
// using the AWS SDK for Rust (e.g., aws-sdk-bedrockruntime)
// - Authentication (likely handled by aws-config based on environment)
// - InvokeModel for non-streaming
// - InvokeModelWithResponseStream for streaming 

use std::sync::Arc;
use tokio::sync::OnceCell; // Using tokio's OnceCell for async initialization

use aws_config::{BehaviorVersion, Region, SdkConfig};
use aws_sdk_bedrockruntime::{
    config::{Builder as BedrockConfigBuilder},
    Client as BedrockClient,
};
use aws_smithy_runtime_api::client::http::SdkHttpConnector;
use aws_smithy_async::time::StaticTimeSource;
use aws_smithy_wasm::wasi::WasiHttpConnector;

use golem_llm::golem::llm::llm::{
    ChatEvent, Config as LlmConfig, Error as LlmError, ErrorCode as LlmErrorCode, Message,
    ToolCall, ToolResult,
};

use crate::conversions; // Assuming this will handle request/response conversions

// Type alias for the Bedrock stream receiver
pub(crate) type AwsBedrockStreamReceiver = aws_sdk_bedrockruntime::output::converse_stream::Receiver;

// Global, async-initialized Bedrock client
static BEDROCK_CLIENT: OnceCell<Arc<BedrockClient>> = OnceCell::const_new();

async fn init_bedrock_client_internal() -> Result<Arc<BedrockClient>, String> {
    log::debug!("Initializing Bedrock client with WASI HTTP connector.");

    let http_connector = WasiHttpConnector::new().map_err(|e| format!("Failed to create WasiHttpConnector: {}", e))?;
    let sdk_http_connector = SdkHttpConnector::new(http_connector);
    let time_source = StaticTimeSource::new(std::time::SystemTime::now());

    let aws_sdk_config = aws_config::defaults(BehaviorVersion::latest())
        .http_connector(sdk_http_connector)
        .time_source(time_source)
        // Optionally set region if not expected from environment or to provide a default
        // .region(Region::new("us-east-1")) 
        .load()
        .await;

    let bedrock_config = BedrockConfigBuilder::from(&aws_sdk_config).build();
    let client = BedrockClient::from_conf(bedrock_config);
    log::info!("Bedrock client initialized successfully.");
    Ok(Arc::new(client))
}

async fn get_bedrock_client() -> Result<Arc<BedrockClient>, LlmError> {
    BEDROCK_CLIENT
        .get_or_try_init(|| async { init_bedrock_client_internal().await })
        .await
        .map_err(|e_str| {
            log::error!("Failed to initialize Bedrock client: {}", e_str);
            LlmError {
                code: LlmErrorCode::InternalError,
                message: format!("Bedrock client initialization failed: {}", e_str),
                provider_error_json: None,
            }
        })
}


pub(crate) async fn perform_converse_request(
    messages: Vec<Message>,
    llm_config: LlmConfig,
    tool_results: Option<Vec<(ToolCall, ToolResult)>>,
) -> Result<ChatEvent, LlmError> {
    log::debug!("perform_converse_request_async called with config: {:?}", llm_config);
    let client = get_bedrock_client().await?;

    let bedrock_request_body = conversions::golem_to_bedrock_converse_request_body(messages, &llm_config, tool_results)?;
    
    let model_id = llm_config.model.clone(); // Get model_id from LlmConfig

    match client
        .converse()
        .model_id(model_id)
        .body(bedrock_request_body)
        // .content_type("application/json") // Usually set by SDK based on model/body
        .send()
        .await {
        Ok(bedrock_response) => {
            log::debug!("Bedrock converse response received successfully.");
            conversions::bedrock_converse_output_to_golem_chat_event(bedrock_response)
        }
        Err(sdk_err) => {
            log::error!("Bedrock converse SDK error: {:#?}", sdk_err);
            Err(conversions::sdk_error_to_llm_error(
                sdk_err,
                "converse",
            ))
        }
    }
}


pub(crate) async fn perform_converse_stream_request(
    messages: Vec<Message>,
    llm_config: LlmConfig,
) -> Result<AwsBedrockStreamReceiver, LlmError> {
    log::debug!("perform_converse_stream_request_async called with config: {:?}", llm_config);
    let client = get_bedrock_client().await?;

    let bedrock_request_body = conversions::golem_to_bedrock_converse_request_body(messages, &llm_config, None)?;
    let model_id = llm_config.model.clone();

    match client
        .converse_stream()
        .model_id(model_id)
        .body(bedrock_request_body)
        // .content_type("application/json")
        .send()
        .await {
        Ok(bedrock_stream_output) => {
            log::debug!("Bedrock converse_stream response received successfully.");
            Ok(bedrock_stream_output.into_stream()) // Convert Output to Receiver
        }
        Err(sdk_err) => {
            log::error!("Bedrock converse_stream SDK error: {:#?}", sdk_err);
            Err(conversions::sdk_error_to_llm_error(
                sdk_err,
                "converse_stream",
            ))
        }
    }
}

// This function will be called by BedrockChatStream in lib.rs
// It needs to be callable from a potentially synchronous context in LlmChatStream::get_next
pub(crate) async fn receive_next_stream_event(
    receiver: &mut AwsBedrockStreamReceiver,
) -> Result<Option<aws_sdk_bedrockruntime::types::ConverseStreamOutput>, LlmError> {
    match receiver.recv().await {
        Ok(Some(event)) => Ok(Some(event)),
        Ok(None) => Ok(None), // Stream gracefully ended
        Err(e) => {
            log::error!("Error receiving from Bedrock stream: {:?}", e);
            // Convert `aws_smithy_http::event_stream::RawMessageError` to LlmError
            Err(LlmError {
                code: LlmErrorCode::InternalError,
                message: format!("Bedrock stream receive error: {}", e),
                provider_error_json: None, // Optionally serialize `e` if useful
            })
        }
    }
}

// Need to add tracing for span creation
use tracing::span as tracing_span; 