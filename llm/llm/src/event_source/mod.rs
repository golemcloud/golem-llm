// Based on https://github.com/jpopesculian/eventsource-stream and https://github.com/jpopesculian/reqwest-eventsource
// modified to use the wasi-http based reqwest, and wasi pollables

pub mod error;
mod event_stream;
mod message_event;
mod ndjson_stream;
mod parser;
mod stream;
mod utf8_stream;

use crate::event_source::error::Error;
use crate::event_source::event_stream::EventStream;
use golem_rust::wasm_rpc::Pollable;
pub use message_event::MessageEvent;
use ndjson_stream::NdJsonStream;
use reqwest::header::HeaderValue;
use reqwest::{Response, StatusCode};
use std::task::Poll;
use stream::{LlmStream, StreamType};

/// The ready state of an [`EventSource`]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum ReadyState {
    /// The EventSource is waiting on a response from the endpoint
    Connecting = 0,
    /// The EventSource is connected
    Open = 1,
    /// The EventSource is closed and no longer emitting Events
    Closed = 2,
}

pub struct EventSource {
    /// stream is the type which implements Stream trait
    stream: StreamType,
    response: Response,
    is_closed: bool,
}

impl EventSource {
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

                let stream = if response
                    .headers()
                    .get(&reqwest::header::CONTENT_TYPE)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .contains("ndjson")
                {
                    StreamType::NdJsonStream(NdJsonStream::new(handle))
                } else {
                    StreamType::EventStream(EventStream::new(handle))
                };
                Ok(Self {
                    response,
                    stream,
                    is_closed: false,
                })
            }
            Err(err) => Err(err),
        }
    }

    /// Close the EventSource stream and stop trying to reconnect
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
        match &self.stream {
            StreamType::EventStream(stream) => stream.subscribe(),
            StreamType::NdJsonStream(stream) => stream.subscribe(),
        }
    }

    pub fn poll_next(&mut self) -> Poll<Option<Result<Event, Error>>> {
        if self.is_closed {
            return Poll::Ready(None);
        }

        match &mut self.stream {
            StreamType::EventStream(stream) => match stream.poll_next() {
                Poll::Ready(Some(Ok(event))) => Poll::Ready(Some(Ok(Event::Message(event)))),
                Poll::Ready(Some(Err(err))) => Poll::Ready(Some(Err(err.into()))),
                Poll::Ready(None) => Poll::Ready(None),
                Poll::Pending => Poll::Pending,
            },
            StreamType::NdJsonStream(stream) => match stream.poll_next() {
                Poll::Ready(Some(Ok(event))) => Poll::Ready(Some(Ok(Event::Message(event)))),
                Poll::Ready(Some(Err(err))) => Poll::Ready(Some(Err(err.into()))),
                Poll::Ready(None) => Poll::Ready(None),
                Poll::Pending => Poll::Pending,
            },
        }
    }
}

#[allow(clippy::result_large_err)]
fn check_response(response: Response) -> Result<Response, Error> {
    match response.status() {
        StatusCode::OK => {}
        status => {
            return Err(Error::InvalidStatusCode(status, response));
        }
    }
    let content_type =
        if let Some(content_type) = response.headers().get(&reqwest::header::CONTENT_TYPE) {
            content_type
        } else {
            return Err(Error::InvalidContentType(
                HeaderValue::from_static(""),
                response,
            ));
        };
    if content_type
        .to_str()
        .map_err(|_| ())
        .and_then(|s| s.parse::<mime::Mime>().map_err(|_| ()))
        .map(|mime_type| {
            matches!(
                (mime_type.type_(), mime_type.subtype()),
                (mime::TEXT, mime::EVENT_STREAM)
            ) || mime_type.subtype().as_str().contains("ndjson")
        })
        .unwrap_or(false)
    {
        Ok(response)
    } else {
        Err(Error::InvalidContentType(content_type.clone(), response))
    }
}

/// Events created by the [`EventSource`]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Event {
    /// The event fired when the connection is opened
    Open,
    /// The event fired when a [`MessageEvent`] is received
    Message(MessageEvent),
}

impl From<MessageEvent> for Event {
    fn from(event: MessageEvent) -> Self {
        Event::Message(event)
    }
}
