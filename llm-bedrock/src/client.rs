use aws_config::meta::region::RegionProviderChain;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::Client as AwsBedrockClient;
use aws_sdk_bedrockruntime::error::ProvideErrorMetadata;
use aws_types::SdkConfig;
use golem_llm::golem::llm::llm::{Error, ErrorCode};
use log::{debug, error, trace};

pub struct BedrockClient {
    client: AwsBedrockClient,
}

// Helper to construct SDK config using SmithyWasmClient
async fn new_bedrock_sdk_config(aws_region_opt: Option<String>) -> Result<SdkConfig, Error> {
    let region_provider = RegionProviderChain::first_try(aws_region_opt.map(aws_types::region::Region::new))
        .or_default_provider()
        .or_else(aws_types::region::Region::new("us-east-1")); // Default fallback region

    let sdk_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(region_provider)
        .load()
        .await;
    Ok(sdk_config)
}

impl BedrockClient {
    pub async fn new(aws_region_from_config: Option<String>) -> Result<Self, Error> {
        debug!(
            "Initializing BedrockClient with region from config: {:?}",
            aws_region_from_config
        );
        match new_bedrock_sdk_config(aws_region_from_config).await {
            Ok(sdk_config) => {
                let client = AwsBedrockClient::new(&sdk_config);
                Ok(Self { client })
            }
            Err(e) => {
                error!("Failed to initialize Bedrock SDK config: {}", e.message);
                Err(e)
            }
        }
    }

    pub async fn invoke_model(
        &self,
        model_id: String,
        body: serde_json::Value,
        accept: String,
        content_type: String,
    ) -> Result<serde_json::Value, Error> {
        trace!(
            "Invoking Bedrock model. Model ID: {}, Content-Type: {}, Accept: {}, Body: {}",
            model_id, content_type, accept, body
        );

        let body_blob = Blob::new(body.to_string());

        let response = self
            .client
            .invoke_model()
            .model_id(model_id)
            .body(body_blob)
            .content_type(content_type)
            .accept(accept)
            .send()
            .await
            .map_err(|sdk_err| {
                let error_message = format!("Bedrock InvokeModel SDK error: {:?}", sdk_err);
                error!("{}", error_message);
                let provider_error_json = Some(error_message.clone());
                let message = sdk_err.message().unwrap_or("Unknown Bedrock SDK error").to_string();

                let code = match sdk_err.as_service_error() {
                    Some(err) => match err {
                        aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ValidationException(_) => ErrorCode::InvalidRequest,
                        aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::AccessDeniedException(_) => ErrorCode::AuthenticationFailed,
                        aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ResourceNotFoundException(_) => ErrorCode::InvalidRequest,
                        aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ThrottlingException(_) => ErrorCode::RateLimitExceeded,
                        aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ServiceQuotaExceededException(_) => ErrorCode::RateLimitExceeded,
                        aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ModelTimeoutException(_) => ErrorCode::InternalError,
                        aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::InternalServerException(_) => ErrorCode::InternalError,
                        aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ModelNotReadyException(_) => ErrorCode::Unsupported,
                        aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ModelErrorException(_) => ErrorCode::InternalError,
                        _ => ErrorCode::InternalError,
                    },
                    None => ErrorCode::InternalError,
                };
                Error {
                    code,
                    message,
                    provider_error_json,
                }
            })?;

        let output_body_bytes = response.body.into_inner();
        let output_body_str = std::str::from_utf8(&output_body_bytes).map_err(|e| {
            let msg = format!("Failed to parse Bedrock response body as UTF-8: {e}");
            error!("{}", msg);
            Error {
                code: ErrorCode::InternalError,
                message: msg,
                provider_error_json: None,
            }
        })?;

        trace!("Received Bedrock response body: {}", output_body_str);

        serde_json::from_str(output_body_str).map_err(|e| {
            let msg = format!("Failed to parse Bedrock response JSON: {e}");
            error!("{} Body was: {}", msg, output_body_str);
            Error {
                code: ErrorCode::InternalError,
                message: msg,
                provider_error_json: Some(output_body_str.to_string()),
            }
        })
    }

    pub async fn invoke_model_with_response_stream(
        &self,
        model_id: String,
        body: serde_json::Value,
        accept: String,
        content_type: String,
    ) -> Result<
        aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamOutput,
        Error
    > {
        trace!(
            "Invoking Bedrock model with response stream. Model ID: {}, Body: {}",
            model_id, body
        );
        let body_blob = Blob::new(body.to_string());

        self.client
            .invoke_model_with_response_stream()
            .model_id(model_id)
            .body(body_blob)
            .content_type(content_type)
            .accept(accept)
            .send()
            .await
            .map_err(|sdk_err| {
                let error_message = format!(
                    "Bedrock InvokeModelWithResponseStream SDK error: {:?}",
                    sdk_err
                );
                error!("{}", error_message);
                let provider_error_json = Some(error_message.clone());
                let message = sdk_err.message().unwrap_or("Unknown Bedrock SDK error for stream").to_string();

                let code = match sdk_err.as_service_error() {
                    Some(err) => match err {
                        aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ValidationException(_) => ErrorCode::InvalidRequest,
                        aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::AccessDeniedException(_) => ErrorCode::AuthenticationFailed,
                        aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ResourceNotFoundException(_) => ErrorCode::InvalidRequest,
                        aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ThrottlingException(_) => ErrorCode::RateLimitExceeded,
                        aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ServiceQuotaExceededException(_) => ErrorCode::RateLimitExceeded,
                        aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ModelTimeoutException(_) => ErrorCode::InternalError,
                        aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::InternalServerException(_) => ErrorCode::InternalError,
                        aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ModelNotReadyException(_) => ErrorCode::Unsupported,
                        aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ModelErrorException(_) => ErrorCode::InternalError,
                        aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ModelStreamErrorException(_) => ErrorCode::InternalError,
                        _ => ErrorCode::InternalError,
                    },
                    None => ErrorCode::InternalError,
                };
                Error {
                    code,
                    message,
                    provider_error_json,
                }
            })
    }
} 