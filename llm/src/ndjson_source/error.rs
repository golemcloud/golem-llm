use crate::ndjson_source::ndjson_stream::NdjsonStreamError;
use core::fmt;
use golem_rust::bindings::wasi::io::streams::StreamError;
use reqwest::Error as ReqwestError;
use reqwest::Response;
use reqwest::StatusCode;
use std::string::FromUtf8Error;
use thiserror::Error;

/// Error raised when a [`RequestBuilder`] cannot be cloned. See [`RequestBuilder::try_clone`] for
/// more information
#[derive(Debug, Clone, Copy)]
pub struct CannotCloneRequestError;

impl fmt::Display for CannotCloneRequestError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("expected a cloneable request")
    }
}

impl std::error::Error for CannotCloneRequestError {}

/// Error raised by the NDJSON stream fetching and parsing
#[derive(Debug, Error)]
pub enum Error {
    /// Source stream is not valid UTF8
    #[error(transparent)]
    Utf8(FromUtf8Error),
    /// JSON parsing error
    #[error("JSON parser error: {0}")]
    JsonParser(serde_json::Error),
    /// The HTTP Request could not be completed
    #[error(transparent)]
    Transport(ReqwestError),
    /// Underlying HTTP response stream error
    #[error("Transport stream error: {0}")]
    TransportStream(String),
    /// The status code returned by the server is invalid
    #[error("Invalid status code: {0}")]
    InvalidStatusCode(StatusCode, Response),
    /// The stream ended
    #[error("Stream ended")]
    StreamEnded,
}

impl From<NdjsonStreamError<ReqwestError>> for Error {
    fn from(err: NdjsonStreamError<ReqwestError>) -> Self {
        match err {
            NdjsonStreamError::Utf8(err) => Self::Utf8(err),
            NdjsonStreamError::JsonParser(err) => Self::JsonParser(err),
            NdjsonStreamError::Transport(err) => Self::Transport(err),
        }
    }
}

impl From<NdjsonStreamError<StreamError>> for Error {
    fn from(err: NdjsonStreamError<StreamError>) -> Self {
        match err {
            NdjsonStreamError::Utf8(err) => Self::Utf8(err),
            NdjsonStreamError::JsonParser(err) => Self::JsonParser(err),
            NdjsonStreamError::Transport(err) => match err {
                StreamError::Closed => Self::StreamEnded,
                StreamError::LastOperationFailed(err) => {
                    Self::TransportStream(err.to_debug_string())
                }
            },
        }
    }
} 