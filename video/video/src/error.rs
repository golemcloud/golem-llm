use crate::exports::golem::video::video::VideoError;
use reqwest::StatusCode;

/// Creates a `VideoError` value representing invalid input
pub fn invalid_input(details: impl AsRef<str>) -> VideoError {
    VideoError::InvalidInput(details.as_ref().to_string())
}

/// Creates a `VideoError` value representing an unsupported feature
pub fn unsupported_feature(what: impl AsRef<str>) -> VideoError {
    VideoError::UnsupportedFeature(what.as_ref().to_string())
}

/// Creates a `VideoError` value representing quota exceeded
pub fn quota_exceeded() -> VideoError {
    VideoError::QuotaExceeded
}

/// Creates a `VideoError` value representing generation failure
pub fn generation_failed(details: impl AsRef<str>) -> VideoError {
    VideoError::GenerationFailed(details.as_ref().to_string())
}

/// Creates a `VideoError` value representing cancellation
pub fn cancelled() -> VideoError {
    VideoError::Cancelled
}

/// Creates a `VideoError` value representing internal error
pub fn internal_error(details: impl AsRef<str>) -> VideoError {
    VideoError::InternalError(details.as_ref().to_string())
}

/// Creates a `VideoError` from a reqwest error
pub fn from_reqwest_error(details: impl AsRef<str>, err: reqwest::Error) -> VideoError {
    VideoError::InternalError(format!("{}: {err}", details.as_ref()))
}

/// Maps HTTP status codes to appropriate video error types
pub fn video_error_from_status(status: StatusCode, message: impl AsRef<str>) -> VideoError {
    let msg = message.as_ref().to_string();

    if status == StatusCode::TOO_MANY_REQUESTS {
        VideoError::QuotaExceeded
    } else if status == StatusCode::BAD_REQUEST {
        VideoError::InvalidInput(msg)
    } else if status == StatusCode::UNPROCESSABLE_ENTITY {
        VideoError::GenerationFailed(msg)
    } else if status == StatusCode::NOT_IMPLEMENTED || status == StatusCode::METHOD_NOT_ALLOWED {
        VideoError::UnsupportedFeature(msg)
    } else {
        VideoError::InternalError(format!("HTTP {}: {}", status, msg))
    }
}
