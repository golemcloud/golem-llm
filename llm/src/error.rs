use crate::event_source;
use crate::golem::llm::llm::{Error, ErrorCode};
use reqwest::StatusCode;

/// Creates an `Error` value representing that something is unsuported
pub fn unsupported(what: impl AsRef<str>) -> Error {
    Error {
        code: ErrorCode::Unsupported,
        message: format!("Unsupported: {}", what.as_ref()),
        provider_error_json: None,
    }
}

pub fn from_reqwest_error(details: impl AsRef<str>, err: reqwest::Error) -> Error {
    Error {
        code: ErrorCode::InternalError,
        message: format!("{}: {err}", details.as_ref()),
        provider_error_json: None,
    }
}

pub fn from_event_source_error(details: impl AsRef<str>, err: event_source::error::Error) -> Error {
    Error {
        code: ErrorCode::InternalError,
        message: format!("{}: {err}", details.as_ref()),
        provider_error_json: None,
    }
}

pub fn error_code_from_status(status: StatusCode) -> ErrorCode {
    if status == StatusCode::TOO_MANY_REQUESTS {
        ErrorCode::RateLimitExceeded
    } else if status == StatusCode::UNAUTHORIZED
        || status == StatusCode::FORBIDDEN
        || status == StatusCode::PAYMENT_REQUIRED
    {
        ErrorCode::AuthenticationFailed
    } else if status.is_client_error() {
        ErrorCode::InvalidRequest
    } else {
        ErrorCode::InternalError
    }
}
