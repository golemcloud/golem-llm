use super::stream::{LlmStream, StreamError as NdJsonStreamError};
use crate::event_source::utf8_stream::Utf8Stream;
use crate::event_source::MessageEvent;
use golem_rust::bindings::wasi::io::streams::{InputStream, StreamError};
use golem_rust::wasm_rpc::Pollable;
use log::trace;
use std::task::Poll;

#[derive(Debug, Clone, Copy)]
pub enum NdJsonStreamState {
    NotStarted,
    Started,
    Terminated,
}

impl NdJsonStreamState {
    fn is_terminated(self) -> bool {
        matches!(self, Self::Terminated)
    }
}

/// A Stream of NDJSON events (newline-delimited JSON)
pub struct NdJsonStream {
    stream: Utf8Stream,
    buffer: String,
    state: NdJsonStreamState,
    last_event_id: String,
}

impl LlmStream for NdJsonStream {
    /// Initialize the NdJsonStream with a Stream
    fn new(stream: InputStream) -> Self {
        Self {
            stream: Utf8Stream::new(stream),
            buffer: String::new(),
            state: NdJsonStreamState::NotStarted,
            last_event_id: String::new(),
        }
    }

    /// Set the last event ID of the stream
    fn set_last_event_id(&mut self, id: impl Into<String>) {
        self.last_event_id = id.into();
    }

    /// Get the last event ID of the stream
    fn last_event_id(&self) -> &str {
        &self.last_event_id
    }

    fn subscribe(&self) -> Pollable {
        self.stream.subscribe()
    }

    fn poll_next(&mut self) -> Poll<Option<Result<MessageEvent, NdJsonStreamError<StreamError>>>> {
        trace!("Polling for next NDJSON event");

        // Try to parse a complete line from the current buffer
        if let Some(event) = try_parse_line(self)? {
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

                    if !self.state.is_terminated() {
                        self.state = NdJsonStreamState::Started;
                    }

                    self.buffer.push_str(&string);

                    // Try to parse complete lines from the updated buffer
                    if let Some(event) = try_parse_line(self)? {
                        return Poll::Ready(Some(Ok(event)));
                    }
                }
                Poll::Ready(Some(Err(err))) => return Poll::Ready(Some(Err(err.into()))),
                Poll::Ready(None) => {
                    self.state = NdJsonStreamState::Terminated;

                    // Process any remaining content in buffer before terminating
                    if !self.buffer.trim().is_empty() {
                        let remaining = std::mem::take(&mut self.buffer);
                        let event = MessageEvent {
                            event: "message".to_string(),
                            data: remaining.trim().to_string(),
                            id: self.last_event_id.clone(),
                            retry: None,
                        };
                        return Poll::Ready(Some(Ok(event)));
                    }

                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Try to parse a complete line from the buffer
/// Returns Ok(Some(event)) if a complete line was found and parsed
/// Returns Ok(None) if no complete line is available
/// Returns Err if there was a parsing error
fn try_parse_line(
    stream: &mut NdJsonStream,
) -> Result<Option<MessageEvent>, NdJsonStreamError<StreamError>> {
    // Look for a complete line (ending with \n)
    if let Some(newline_pos) = stream.buffer.find('\n') {
        // Extract the line (without the newline)
        let line = stream.buffer[..newline_pos].trim().to_string();

        // Remove the processed line from the buffer (including the newline)
        stream.buffer.drain(..=newline_pos);

        // Skip empty lines
        if line.is_empty() {
            return Ok(None);
        }

        trace!("Parsed NDJSON line: {line}");

        // Create a MessageEvent with the JSON line as data
        let event = MessageEvent {
            event: "message".to_string(),
            data: line,
            id: stream.last_event_id.clone(),
            retry: None,
        };

        return Ok(Some(event));
    }

    Ok(None)
}
