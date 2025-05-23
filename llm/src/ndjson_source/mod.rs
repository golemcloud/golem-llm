// Based on the existing event_source implementation but adapted for NDJSON streaming
// NDJSON (Newline Delimited JSON) streams contain one JSON object per line

pub mod error;
mod ndjson_stream;
mod utf8_stream;

use crate::ndjson_source::error::Error;
use crate::ndjson_source::ndjson_stream::NdjsonStream;
pub use crate::ndjson_source::utf8_stream::Utf8Stream;
use golem_rust::wasm_rpc::Pollable;
use reqwest::{Response, StatusCode};
use serde_json::Value as JsonValue;
use std::task::Poll;

/// The ready state of an [`NdjsonSource`]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum ReadyState {
    /// The NdjsonSource is waiting on a response from the endpoint
    Connecting = 0,
    /// The NdjsonSource is connected
    Open = 1,
    /// The NdjsonSource is closed and no longer emitting JSON objects
    Closed = 2,
}

/// A JSON object from an NDJSON stream
#[derive(Debug, Clone, PartialEq)]
pub struct JsonEvent {
    /// The raw JSON string
    pub raw: String,
    /// The parsed JSON value
    pub data: JsonValue,
}

/// Events created by the [`NdjsonSource`]
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// The event fired when the connection is opened
    Open,
    /// The event fired when a JSON object is received
    Json(JsonEvent),
}

impl From<JsonEvent> for Event {
    fn from(event: JsonEvent) -> Self {
        Event::Json(event)
    }
}

pub struct NdjsonSource {
    stream: NdjsonStream,
    response: Response,
    is_closed: bool,
}

impl NdjsonSource {
    #[allow(clippy::result_large_err)]
    pub fn new(response: Response) -> Result<Self, Error> {
        match check_response(response) {
            Ok(mut response) => {
                let handle = unsafe {
                    std::mem::transmute::<
                        reqwest::InputStream,
                        golem_rust::bindings::wasi::io::streams::InputStream,
                    >(response.get_raw_input_stream())
                };
                let stream = NdjsonStream::new(handle);
                Ok(Self {
                    response,
                    stream,
                    is_closed: false,
                })
            }
            Err(err) => Err(err),
        }
    }

    /// Close the NdjsonSource stream
    pub fn close(&mut self) {
        self.is_closed = true;
    }

    /// Get the current ready state
    pub fn ready_state(&self) -> ReadyState {
        if self.is_closed {
            ReadyState::Closed
        } else {
            ReadyState::Open
        }
    }

    pub fn subscribe(&self) -> Pollable {
        self.stream.subscribe()
    }

    pub fn poll_next(&mut self) -> Poll<Option<Result<Event, Error>>> {
        if self.is_closed {
            return Poll::Ready(None);
        }

        match self.stream.poll_next() {
            Poll::Ready(Some(Err(err))) => {
                let err = err.into();
                self.is_closed = true;
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(Some(Ok(event))) => Poll::Ready(Some(Ok(event.into()))),
            Poll::Ready(None) => {
                let err = Error::StreamEnded;
                self.is_closed = true;
                Poll::Ready(Some(Err(err)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[allow(clippy::result_large_err)]
fn check_response(response: Response) -> Result<Response, Error> {
    match response.status() {
        StatusCode::OK => Ok(response),
        status => Err(Error::InvalidStatusCode(status, response)),
    }
} 