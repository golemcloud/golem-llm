use crate::event_source::{Event, EventSource, MessageEvent};
use crate::golem::llm::llm::{Error, ErrorCode, GuestChatStream, StreamEvent};
use crate::ndjson_source::{Event as NdjsonEvent, JsonEvent, NdjsonSource};
use golem_rust::wasm_rpc::Pollable;
use std::cell::{Ref, RefMut};
use std::task::Poll;

// Original SSE-based trait
pub trait LlmChatStreamState: 'static {
    fn failure(&self) -> &Option<Error>;
    fn is_finished(&self) -> bool;
    fn set_finished(&self);
    fn stream(&self) -> Ref<Option<EventSource>>;
    fn stream_mut(&self) -> RefMut<Option<EventSource>>;
    fn decode_message(&self, raw: &str) -> Result<Option<StreamEvent>, String>;
}

// New NDJSON-based trait
pub trait LlmNdjsonStreamState: 'static {
    fn failure(&self) -> &Option<Error>;
    fn is_finished(&self) -> bool;
    fn set_finished(&self);
    fn stream(&self) -> Ref<Option<NdjsonSource>>;
    fn stream_mut(&self) -> RefMut<Option<NdjsonSource>>;
    fn decode_message(&self, raw: &str) -> Result<Option<StreamEvent>, String>;
}

pub struct LlmChatStream<T> {
    implementation: T,
}

impl<T: LlmChatStreamState> LlmChatStream<T> {
    pub fn new(implementation: T) -> Self {
        Self { implementation }
    }

    pub fn subscribe(&self) -> Pollable {
        if let Some(stream) = self.implementation.stream().as_ref() {
            stream.subscribe()
        } else {
            golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0)
        }
    }
}

// NDJSON-specific chat stream
pub struct LlmNdjsonChatStream<T> {
    implementation: T,
}

impl<T: LlmNdjsonStreamState> LlmNdjsonChatStream<T> {
    pub fn new(implementation: T) -> Self {
        Self { implementation }
    }

    pub fn subscribe(&self) -> Pollable {
        if let Some(stream) = self.implementation.stream().as_ref() {
            stream.subscribe()
        } else {
            golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0)
        }
    }
}

impl<T: LlmChatStreamState> GuestChatStream for LlmChatStream<T> {
    fn get_next(&self) -> Option<Vec<StreamEvent>> {
        if self.implementation.is_finished() {
            return Some(vec![]);
        }

        let mut stream = self.implementation.stream_mut();
        if let Some(stream) = stream.as_mut() {
            match stream.poll_next() {
                Poll::Ready(None) => {
                    self.implementation.set_finished();
                    Some(vec![])
                }
                Poll::Ready(Some(Err(crate::event_source::error::Error::StreamEnded))) => {
                    self.implementation.set_finished();
                    Some(vec![])
                }
                Poll::Ready(Some(Err(error))) => Some(vec![StreamEvent::Error(Error {
                    code: ErrorCode::InternalError,
                    message: error.to_string(),
                    provider_error_json: None,
                })]),
                Poll::Ready(Some(Ok(event))) => {
                    let mut events = vec![];

                    match event {
                        Event::Open => {}
                        Event::Message(MessageEvent { data, .. }) => {
                            if data != "[DONE]" {
                                match self.implementation.decode_message(&data) {
                                    Ok(Some(stream_event)) => {
                                        if matches!(stream_event, StreamEvent::Finish(_)) {
                                            self.implementation.set_finished();
                                        }
                                        events.push(stream_event);
                                    }
                                    Ok(None) => {
                                        // Ignored event
                                    }
                                    Err(error) => {
                                        events.push(StreamEvent::Error(Error {
                                            code: ErrorCode::InternalError,
                                            message: error,
                                            provider_error_json: None,
                                        }));
                                    }
                                }
                            }
                        }
                    }

                    if events.is_empty() {
                        None
                    } else {
                        Some(events)
                    }
                }
                Poll::Pending => None,
            }
        } else if let Some(error) = self.implementation.failure().clone() {
            self.implementation.set_finished();
            Some(vec![StreamEvent::Error(error)])
        } else {
            None
        }
    }

    fn blocking_get_next(&self) -> Vec<StreamEvent> {
        let pollable = self.subscribe();
        let mut result = Vec::new();
        loop {
            pollable.block();
            match self.get_next() {
                Some(events) => {
                    result.extend(events);
                    break result;
                }
                None => continue,
            }
        }
    }
}

impl<T: LlmNdjsonStreamState> GuestChatStream for LlmNdjsonChatStream<T> {
    fn get_next(&self) -> Option<Vec<StreamEvent>> {
        if self.implementation.is_finished() {
            return Some(vec![]);
        }

        let mut stream = self.implementation.stream_mut();
        if let Some(stream) = stream.as_mut() {
            match stream.poll_next() {
                Poll::Ready(None) => {
                    self.implementation.set_finished();
                    Some(vec![])
                }
                Poll::Ready(Some(Err(crate::ndjson_source::error::Error::StreamEnded))) => {
                    self.implementation.set_finished();
                    Some(vec![])
                }
                Poll::Ready(Some(Err(error))) => Some(vec![StreamEvent::Error(Error {
                    code: ErrorCode::InternalError,
                    message: error.to_string(),
                    provider_error_json: None,
                })]),
                Poll::Ready(Some(Ok(event))) => {
                    let mut events = vec![];

                    match event {
                        NdjsonEvent::Open => {}
                        NdjsonEvent::Json(JsonEvent { raw, .. }) => {
                            match self.implementation.decode_message(&raw) {
                                Ok(Some(stream_event)) => {
                                    if matches!(stream_event, StreamEvent::Finish(_)) {
                                        self.implementation.set_finished();
                                    }
                                    events.push(stream_event);
                                }
                                Ok(None) => {
                                    // Ignored event
                                }
                                Err(error) => {
                                    events.push(StreamEvent::Error(Error {
                                        code: ErrorCode::InternalError,
                                        message: error,
                                        provider_error_json: None,
                                    }));
                                }
                            }
                        }
                    }

                    if events.is_empty() {
                        None
                    } else {
                        Some(events)
                    }
                }
                Poll::Pending => None,
            }
        } else if let Some(error) = self.implementation.failure().clone() {
            self.implementation.set_finished();
            Some(vec![StreamEvent::Error(error)])
        } else {
            None
        }
    }

    fn blocking_get_next(&self) -> Vec<StreamEvent> {
        let pollable = self.subscribe();
        let mut result = Vec::new();
        loop {
            pollable.block();
            match self.get_next() {
                Some(events) => {
                    result.extend(events);
                    break result;
                }
                None => continue,
            }
        }
    }
}
