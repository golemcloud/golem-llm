use crate::event_source::event_stream::EventStreamError;
use core::fmt;
use golem_rust::bindings::wasi::io::streams::StreamError;
use nom::error::Error as NomError;
use reqwest::header::HeaderValue;
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

/// Error raised by the EventSource stream fetching and parsing
#[derive(Debug, Error)]
pub enum Error {
    /// Source stream is not valid UTF8
    #[error(transparent)]
    Utf8(FromUtf8Error),
    /// Source stream is not a valid EventStream
    #[error("Protocol parser error: {0:?}")]
    Parser(NomError<String>),
    /// The HTTP Request could not be completed
    #[error(transparent)]
    Transport(ReqwestError),
    /// Underlying HTTP response stream error
    #[error("Transport stream error: {0}")]
    TransportStream(String),
    /// The `Content-Type` returned by the server is invalid
    #[error("Invalid header value: {0:?}")]
    InvalidContentType(HeaderValue, Response),
    /// The status code returned by the server is invalid
    #[error("Invalid status code: {0}")]
    InvalidStatusCode(StatusCode, Response),
    /// The `Last-Event-ID` cannot be formed into a Header to be submitted to the server
    #[error("Invalid `Last-Event-ID`: {0}")]
    InvalidLastEventId(String),
    /// The stream ended
    #[error("Stream ended")]
    StreamEnded,
}

impl From<EventStreamError<ReqwestError>> for Error {
    fn from(err: EventStreamError<ReqwestError>) -> Self {
        match err {
            EventStreamError::Utf8(err) => Self::Utf8(err),
            EventStreamError::Parser(err) => Self::Parser(err),
            EventStreamError::Transport(err) => Self::Transport(err),
        }
    }
}

impl From<EventStreamError<StreamError>> for Error {
    fn from(err: EventStreamError<StreamError>) -> Self {
        match err {
            EventStreamError::Utf8(err) => Self::Utf8(err),
            EventStreamError::Parser(err) => Self::Parser(err),
            EventStreamError::Transport(err) => match err {
                StreamError::Closed => Self::StreamEnded,
                StreamError::LastOperationFailed(err) => {
                    Self::TransportStream(err.to_debug_string())
                }
            },
        }
    }
}
