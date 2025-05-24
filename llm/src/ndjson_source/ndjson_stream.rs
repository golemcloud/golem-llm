use crate::ndjson_source::utf8_stream::{Utf8Stream, Utf8StreamError};
use crate::ndjson_source::JsonEvent;
use core::fmt;
use golem_rust::bindings::wasi::io::streams::{InputStream, StreamError};
use golem_rust::wasm_rpc::Pollable;
use log::trace;
use serde_json::Value as JsonValue;
use std::string::FromUtf8Error;
use std::task::Poll;

#[derive(Debug, Clone, Copy)]
pub enum NdjsonStreamState {
    NotStarted,
    Started,
    Terminated,
}

impl NdjsonStreamState {
    fn is_terminated(self) -> bool {
        matches!(self, Self::Terminated)
    }
}

/// A Stream of NDJSON objects
pub struct NdjsonStream {
    stream: Utf8Stream,
    buffer: String,
    state: NdjsonStreamState,
}

impl NdjsonStream {
    /// Initialize the NdjsonStream with a Stream
    pub fn new(stream: InputStream) -> Self {
        Self {
            stream: Utf8Stream::new(stream),
            buffer: String::new(),
            state: NdjsonStreamState::NotStarted,
        }
    }

    pub fn subscribe(&self) -> Pollable {
        self.stream.subscribe()
    }

    pub fn poll_next(&mut self) -> Poll<Option<Result<JsonEvent, NdjsonStreamError<StreamError>>>> {
        trace!("Polling for next NDJSON event");

        // Try to parse any complete JSON objects from the current buffer
        if let Some(event) = self.try_parse_json_from_buffer()? {
            return Poll::Ready(Some(Ok(event)));
        }

        if self.state.is_terminated() {
            return Poll::Ready(None);
        }

        loop {
            match self.stream.poll_next() {
                Poll::Ready(Some(Ok(string))) => {
                    if string.is_empty() {
                        continue;
                    }

                    self.state = NdjsonStreamState::Started;
                    self.buffer.push_str(&string);

                    // Try to parse JSON objects from the updated buffer
                    if let Some(event) = self.try_parse_json_from_buffer()? {
                        return Poll::Ready(Some(Ok(event)));
                    }
                }
                Poll::Ready(Some(Err(err))) => return Poll::Ready(Some(Err(err.into()))),
                Poll::Ready(None) => {
                    self.state = NdjsonStreamState::Terminated;

                    // Process any remaining JSON in the buffer
                    if !self.buffer.trim().is_empty() {
                        if let Some(event) = self.try_parse_json_from_buffer()? {
                            return Poll::Ready(Some(Ok(event)));
                        }
                    }

                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }

    /// Try to parse a complete JSON object from the buffer
    /// Returns the first complete JSON object found, or None if no complete object is available
    /// No Size Limit on the JSON object, Potential Memory Leak as Buffer grows indefinitely
    fn try_parse_json_from_buffer(
        &mut self,
    ) -> Result<Option<JsonEvent>, NdjsonStreamError<StreamError>> {
        loop {
            if let Some(newline_pos) = self.buffer.find('\n') {
                // Extract the line up to the newline
                let line = self.buffer[..newline_pos].trim().to_string();

                // Remove the processed line from the buffer
                self.buffer.drain(..=newline_pos);

                // Skip empty lines
                if line.is_empty() {
                    continue;
                }

                // Try to parse the line as JSON
                match serde_json::from_str::<JsonValue>(&line) {
                    Ok(data) => {
                        trace!("Successfully parsed NDJSON object: {}", line);
                        return Ok(Some(JsonEvent { raw: line, data }));
                    }
                    Err(err) => {
                        trace!("Failed to parse NDJSON line: {}, error: {}", line, err);
                        return Err(NdjsonStreamError::JsonParser(err));
                    }
                }
            } else {
                // No complete line available
                return Ok(None);
            }
        }
    }
}

/// Error thrown while parsing an NDJSON line
#[derive(Debug)]
pub enum NdjsonStreamError<E> {
    /// Source stream is not valid UTF8
    Utf8(FromUtf8Error),
    /// JSON parsing error
    JsonParser(serde_json::Error),
    /// Underlying source stream error
    Transport(E),
}

impl<E> From<Utf8StreamError<E>> for NdjsonStreamError<E> {
    fn from(err: Utf8StreamError<E>) -> Self {
        match err {
            Utf8StreamError::Utf8(err) => Self::Utf8(err),
            Utf8StreamError::Transport(err) => Self::Transport(err),
        }
    }
}

impl<E> fmt::Display for NdjsonStreamError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Utf8(err) => f.write_fmt(format_args!("UTF8 error: {}", err)),
            Self::JsonParser(err) => f.write_fmt(format_args!("JSON parse error: {}", err)),
            Self::Transport(err) => f.write_fmt(format_args!("Transport error: {}", err)),
        }
    }
}

impl<E> std::error::Error for NdjsonStreamError<E> where E: fmt::Display + fmt::Debug + Send + Sync {}
