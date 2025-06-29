use std::task::Poll;

use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use aws_smithy_types::event_stream::HeaderValue;
use base64::Engine;
use golem_rust::{
    bindings::wasi::io::streams::{InputStream, StreamError},
    wasm_rpc::Pollable,
};
use log::trace;

use crate::event_source::{
    stream::{LlmStream, StreamError as AwsEventStreamError},
    MessageEvent,
};

#[derive(Debug, Clone, Copy)]
pub enum AwsEventStreamState {
    NotStarted,
    Started,
    Terminated,
}

impl AwsEventStreamState {
    fn is_terminated(self) -> bool {
        matches!(self, Self::Terminated)
    }
}

/// A Stream of AWS EventStream events using the vnd.amazon.eventstream binary format
pub struct AwsEventStream {
    stream: InputStream,
    decoder: MessageFrameDecoder,
    buffer: Vec<u8>,
    state: AwsEventStreamState,
    last_event_id: String,
    subscription: Pollable,
}

impl LlmStream for AwsEventStream {
    fn new(stream: InputStream) -> Self {
        let subscription = stream.subscribe();
        Self {
            decoder: MessageFrameDecoder::new(),
            buffer: Vec::new(),
            state: AwsEventStreamState::NotStarted,
            last_event_id: String::new(),
            stream,
            subscription,
        }
    }

    fn set_last_event_id(&mut self, id: impl Into<String>) {
        self.last_event_id = id.into();
    }

    fn last_event_id(&self) -> &str {
        &self.last_event_id
    }

    fn subscribe(&self) -> Pollable {
        self.stream.subscribe()
    }

    fn poll_next(
        &mut self,
    ) -> Poll<Option<Result<MessageEvent, AwsEventStreamError<StreamError>>>> {
        trace!("Polling for next AWS EventStream event");

        if let Some(event) = try_decode_message(self)? {
            return Poll::Ready(Some(Ok(event)));
        }

        if self.state.is_terminated() {
            return Poll::Ready(None);
        }

        loop {
            if self.subscription.ready() {
                match self.stream.read(8192) {
                    Ok(bytes) => {
                        if bytes.is_empty() {
                            continue;
                        }

                        if !self.state.is_terminated() {
                            self.state = AwsEventStreamState::Started;
                        }

                        self.buffer.extend_from_slice(&bytes);

                        // Try to decode complete messages from the updated buffer
                        if let Some(event) = try_decode_message(self)? {
                            return Poll::Ready(Some(Ok(event)));
                        }
                    }
                    Err(StreamError::Closed) => {
                        trace!("AWS EventStream closed");
                        self.state = AwsEventStreamState::Terminated;
                        return Poll::Ready(None);
                    }
                    Err(err) => return Poll::Ready(Some(Err(AwsEventStreamError::Transport(err)))),
                }
            } else {
                return Poll::Pending;
            }
        }
    }
}

fn try_decode_message(
    stream: &mut AwsEventStream,
) -> Result<Option<MessageEvent>, AwsEventStreamError<StreamError>> {
    if stream.buffer.is_empty() {
        return Ok(None);
    }

    let mut buffer_slice = stream.buffer.as_slice();
    let original_len = buffer_slice.len();

    match stream.decoder.decode_frame(&mut buffer_slice) {
        Ok(DecodedFrame::Complete(message)) => {
            trace!(
                "Decoded AWS EventStream message with {} byte payload",
                message.payload().len()
            );

            let consumed = original_len - buffer_slice.len();

            let event_type = message
                .headers()
                .iter()
                .find(|header| header.name().as_str() == ":event-type")
                .and_then(|header| {
                    if let HeaderValue::String(s) = header.value() {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .unwrap_or("message");

            let data = match std::str::from_utf8(message.payload()) {
                Ok(s) => s.to_string(),
                Err(_) => base64::engine::general_purpose::STANDARD.encode(message.payload()),
            };

            if let Some(id_header) = message
                .headers()
                .iter()
                .find(|header| header.name().as_str() == ":event-id")
            {
                if let HeaderValue::String(id) = id_header.value() {
                    stream.last_event_id = id.as_str().to_string();
                }
            }

            stream.buffer.drain(..consumed);

            let event = MessageEvent {
                event: event_type.to_string(),
                data,
                id: stream.last_event_id.clone(),
                retry: None,
            };

            Ok(Some(event))
        }
        Ok(DecodedFrame::Incomplete) => Ok(None),
        Err(err) => Err(AwsEventStreamError::Parser(nom::error::Error::new(
            format!("AWS EventStream decode error: {}", err),
            nom::error::ErrorKind::Tag,
        ))),
    }
}
