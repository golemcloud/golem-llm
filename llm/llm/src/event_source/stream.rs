use core::fmt;
use std::{string::FromUtf8Error, task::Poll};

use super::{
    event_stream::EventStream, ndjson_stream::NdJsonStream, utf8_stream::Utf8StreamError,
    MessageEvent,
};
use golem_rust::{
    bindings::wasi::io::streams::{InputStream, StreamError as WasiStreamError},
    wasm_rpc::Pollable,
};
use nom::error::Error as NomError;

pub enum StreamType {
    EventStream(EventStream),
    NdJsonStream(NdJsonStream),
}

pub trait LlmStream {
    fn new(stream: InputStream) -> Self;
    fn set_last_event_id(&mut self, id: impl Into<String>);
    fn last_event_id(&self) -> &str;
    fn subscribe(&self) -> Pollable;
    fn poll_next(&mut self) -> Poll<Option<Result<MessageEvent, StreamError<WasiStreamError>>>>;
}

/// Error thrown while parsing an event line
#[derive(Debug, PartialEq)]
pub enum StreamError<E> {
    /// Source stream is not valid UTF8
    Utf8(FromUtf8Error),
    /// Source stream is not a valid EventStream
    Parser(NomError<String>),
    /// Underlying source stream error
    Transport(E),
}

impl<E> From<Utf8StreamError<E>> for StreamError<E> {
    fn from(err: Utf8StreamError<E>) -> Self {
        match err {
            Utf8StreamError::Utf8(err) => Self::Utf8(err),
            Utf8StreamError::Transport(err) => Self::Transport(err),
        }
    }
}

impl<E> From<NomError<&str>> for StreamError<E> {
    fn from(err: NomError<&str>) -> Self {
        StreamError::Parser(NomError::new(err.input.to_string(), err.code))
    }
}

impl<E> fmt::Display for StreamError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Utf8(err) => f.write_fmt(format_args!("UTF8 error: {err}")),
            Self::Parser(err) => f.write_fmt(format_args!("Parse error: {err}")),
            Self::Transport(err) => f.write_fmt(format_args!("Transport error: {err}")),
        }
    }
}

impl<E> std::error::Error for StreamError<E> where E: fmt::Display + fmt::Debug + Send + Sync {}
