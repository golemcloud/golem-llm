use std::fmt::Display;
use aws_sdk_bedrockruntime::error::SdkError;
use aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError;
use golem_llm::golem::llm::llm::{Error, ErrorCode};
use serde_json::Error as JsonError;

#[derive(Debug)]
pub enum BedrockError {
    AwsSdk(String),
    Json(String),
    Llm(Error),
}

impl Display for BedrockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BedrockError::AwsSdk(msg) => write!(f, "AWS SDK error: {}", msg),
            BedrockError::Json(msg) => write!(f, "JSON error: {}", msg),
            BedrockError::Llm(err) => write!(f, "LLM error: {:?}", err),
        }
    }
}

impl std::error::Error for BedrockError {}

impl From<SdkError<InvokeModelError>> for BedrockError {
    fn from(err: SdkError<InvokeModelError>) -> Self {
        BedrockError::AwsSdk(err.to_string())
    }
}

impl From<JsonError> for BedrockError {
    fn from(err: JsonError) -> Self {
        BedrockError::Json(err.to_string())
    }
}

impl From<Error> for BedrockError {
    fn from(err: Error) -> Self {
        BedrockError::Llm(err)
    }
}

impl From<BedrockError> for Error {
    fn from(err: BedrockError) -> Self {
        match err {
            BedrockError::AwsSdk(msg) => Error {
                code: ErrorCode::InternalError,
                message: msg,
                provider_error_json: None,
            },
            BedrockError::Json(msg) => Error {
                code: ErrorCode::InvalidRequest,
                message: msg,
                provider_error_json: None,
            },
            BedrockError::Llm(err) => err,
        }
    }
} 