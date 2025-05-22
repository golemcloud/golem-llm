use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbedError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    #[error("Provider error: {0}")]
    ProviderError(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

pub fn map_error_code(error: &EmbedError) -> golem_rust::bindings::golem::embed::embed::ErrorCode {
    use golem_rust::bindings::golem::embed::embed::ErrorCode;
    match error {
        EmbedError::InvalidRequest(_) => ErrorCode::InvalidRequest,
        EmbedError::ModelNotFound(_) => ErrorCode::ModelNotFound,
        EmbedError::Unsupported(_) => ErrorCode::Unsupported,
        EmbedError::ProviderError(_) => ErrorCode::ProviderError,
        EmbedError::RateLimitExceeded(_) => ErrorCode::RateLimitExceeded,
        EmbedError::InternalError(_) => ErrorCode::InternalError,
        EmbedError::Unknown(_) => ErrorCode::Unknown,
    }
}