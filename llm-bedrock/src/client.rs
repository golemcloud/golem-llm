use aws_config::meta::region::RegionProviderChain;
use aws_sdk_bedrockruntime::config::Credentials;
use aws_sdk_bedrockruntime::error::ProvideErrorMetadata;
use aws_sdk_bedrockruntime::operation::converse::ConverseOutput;
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamOutput;
use aws_sdk_bedrockruntime::types::{Message, SystemContentBlock};
use aws_sdk_bedrockruntime::Client as AwsBedrockClient;
use aws_smithy_wasm::wasi::WasiHttpClientBuilder;
use golem_llm::golem::llm::llm::{Error, ErrorCode};
use log::{debug, error, trace};

pub struct BedrockClient {
    client: AwsBedrockClient,
}

#[derive(Debug, Clone)]
pub struct BedrockRequest {
    pub model_id: String,
    pub messages: Vec<Message>,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

impl BedrockClient {
    pub fn new(
        aws_region_opt: Option<String>,
        aws_credentials_opt: Option<AwsCredentials>,
    ) -> Result<Self, Error> {
        debug!(
            "Initializing BedrockClient with region: {:?} and credentials: {}",
            aws_region_opt,
            aws_credentials_opt.is_some()
        );

        // This will be executed in a Tokio context, but synchronously
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error {
                code: ErrorCode::InternalError,
                message: format!("Failed to create Tokio runtime: {e}"),
                provider_error_json: None,
            })?;

        let client = rt.block_on(async {
            // Create WASI HTTP client
            let wasi_client = WasiHttpClientBuilder::new().build();

            // Set up region provider chain
            let region_provider =
                RegionProviderChain::first_try(aws_region_opt.map(aws_types::region::Region::new))
                    .or_default_provider()
                    .or_else(aws_types::region::Region::new("us-east-1"));

            // Build SDK config with WASI client
            let mut config_builder = aws_config::defaults(aws_config::BehaviorVersion::latest())
                .region(region_provider)
                .http_client(wasi_client);

            // Add credentials if provided
            if let Some(creds) = aws_credentials_opt {
                let credentials = Credentials::new(
                    &creds.access_key_id,
                    &creds.secret_access_key,
                    creds.session_token,
                    None, // Expiry
                    "golem-llm-bedrock",
                );
                config_builder = config_builder.credentials_provider(credentials);
            }

            let sdk_config = config_builder.load().await;
            AwsBedrockClient::new(&sdk_config)
        });

        Ok(Self { client })
    }

    pub fn converse(&self, request: BedrockRequest) -> Result<ConverseOutput, Error> {
        trace!(
            "Bedrock converse request. Model ID: {}, Messages: {}",
            request.model_id,
            request.messages.len()
        );

        // Create a runtime for this specific call
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error {
                code: ErrorCode::InternalError,
                message: format!("Failed to create Tokio runtime: {e}"),
                provider_error_json: None,
            })?;

        rt.block_on(async {
            let mut converse_request = self.client
                .converse()
                .model_id(&request.model_id)
                .set_messages(Some(request.messages));

            // Add system prompt if provided
            if let Some(system_prompt) = request.system_prompt {
                converse_request = converse_request.system(SystemContentBlock::Text(system_prompt));
            }

            // Add inference configuration if provided
            if request.max_tokens.is_some() || request.temperature.is_some() {
                let mut inference_config = aws_sdk_bedrockruntime::types::InferenceConfiguration::builder();
                
                if let Some(max_tokens) = request.max_tokens {
                    inference_config = inference_config.max_tokens(max_tokens);
                }
                
                if let Some(temperature) = request.temperature {
                    inference_config = inference_config.temperature(temperature);
                }
                
                converse_request = converse_request.inference_config(inference_config.build());
            }

            converse_request.send().await
        }).map_err(|sdk_err| {
            let error_message = format!("Bedrock Converse SDK error: {sdk_err:?}");
            error!("{error_message}");
            let provider_error_json = Some(error_message.clone());
            let message = sdk_err.message().unwrap_or("Unknown Bedrock SDK error").to_string();

            let code = match sdk_err.as_service_error() {
                Some(err) => match err {
                    aws_sdk_bedrockruntime::operation::converse::ConverseError::ValidationException(_) => ErrorCode::InvalidRequest,
                    aws_sdk_bedrockruntime::operation::converse::ConverseError::AccessDeniedException(_) => ErrorCode::AuthenticationFailed,
                    aws_sdk_bedrockruntime::operation::converse::ConverseError::ResourceNotFoundException(_) => ErrorCode::InvalidRequest,
                    aws_sdk_bedrockruntime::operation::converse::ConverseError::ThrottlingException(_) => ErrorCode::RateLimitExceeded,
                    aws_sdk_bedrockruntime::operation::converse::ConverseError::ModelTimeoutException(_) => ErrorCode::InternalError,
                    aws_sdk_bedrockruntime::operation::converse::ConverseError::InternalServerException(_) => ErrorCode::InternalError,
                    aws_sdk_bedrockruntime::operation::converse::ConverseError::ModelNotReadyException(_) => ErrorCode::Unsupported,
                    aws_sdk_bedrockruntime::operation::converse::ConverseError::ModelErrorException(_) => ErrorCode::InternalError,
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

    pub fn converse_stream(&self, request: BedrockRequest) -> Result<ConverseStreamOutput, Error> {
        trace!(
            "Bedrock converse_stream request. Model ID: {}, Messages: {}",
            request.model_id,
            request.messages.len()
        );

        // Create a runtime for this specific call
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error {
                code: ErrorCode::InternalError,
                message: format!("Failed to create Tokio runtime: {e}"),
                provider_error_json: None,
            })?;

        rt.block_on(async {
            let mut converse_request = self.client
                .converse_stream()
                .model_id(&request.model_id)
                .set_messages(Some(request.messages));

            // Add system prompt if provided
            if let Some(system_prompt) = request.system_prompt {
                converse_request = converse_request.system(SystemContentBlock::Text(system_prompt));
            }

            // Add inference configuration if provided
            if request.max_tokens.is_some() || request.temperature.is_some() {
                let mut inference_config = aws_sdk_bedrockruntime::types::InferenceConfiguration::builder();
                
                if let Some(max_tokens) = request.max_tokens {
                    inference_config = inference_config.max_tokens(max_tokens);
                }
                
                if let Some(temperature) = request.temperature {
                    inference_config = inference_config.temperature(temperature);
                }
                
                converse_request = converse_request.inference_config(inference_config.build());
            }

            converse_request.send().await
        }).map_err(|sdk_err| {
            let error_message = format!("Bedrock ConverseStream SDK error: {sdk_err:?}");
            error!("{error_message}");
            let provider_error_json = Some(error_message.clone());
            let message = sdk_err.message().unwrap_or("Unknown Bedrock SDK error for stream").to_string();

            let code = match sdk_err.as_service_error() {
                Some(err) => match err {
                    aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError::ValidationException(_) => ErrorCode::InvalidRequest,
                    aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError::AccessDeniedException(_) => ErrorCode::AuthenticationFailed,
                    aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError::ResourceNotFoundException(_) => ErrorCode::InvalidRequest,
                    aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError::ThrottlingException(_) => ErrorCode::RateLimitExceeded,
                    aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError::ModelTimeoutException(_) => ErrorCode::InternalError,
                    aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError::InternalServerException(_) => ErrorCode::InternalError,
                    aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError::ModelNotReadyException(_) => ErrorCode::Unsupported,
                    aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError::ModelErrorException(_) => ErrorCode::InternalError,
                    aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError::ModelStreamErrorException(_) => ErrorCode::InternalError,
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
