use crate::event_source::parser::{is_bom, is_lf, line, RawEventLine};
use crate::event_source::MessageEvent;
use crate::utf8_stream::{Utf8Stream, Utf8StreamError};
use core::fmt;
use core::time::Duration;
use golem_rust::bindings::wasi::io::streams::{InputStream, StreamError};
use golem_rust::wasm_rpc::Pollable;
use log::trace;
use nom::error::Error as NomError;
use std::string::FromUtf8Error;
use std::task::Poll;

#[derive(Default, Debug)]
struct EventBuilder {
    event: MessageEvent,
    is_complete: bool,
}

impl EventBuilder {
    /// From the HTML spec
    ///
    /// -> If the field name is "event"
    ///    Set the event type buffer to field value.
    ///
    /// -> If the field name is "data"
    ///    Append the field value to the data buffer, then append a single U+000A LINE FEED (LF)
    ///    character to the data buffer.
    ///
    /// -> If the field name is "id"
    ///    If the field value does not contain U+0000 NULL, then set the last event ID buffer
    ///    to the field value. Otherwise, ignore the field.
    ///
    /// -> If the field name is "retry"
    ///    If the field value consists of only ASCII digits, then interpret the field value as
    ///    an integer in base ten, and set the event stream's reconnection time to that integer.
    ///    Otherwise, ignore the field.
    ///
    /// -> Otherwise
    ///    The field is ignored.
    fn add(&mut self, line: RawEventLine) {
        match line {
            RawEventLine::Field(field, val) => {
                let val = val.unwrap_or("");
                match field {
                    "event" => {
                        self.event.event = val.to_string();
                    }
                    "data" => {
                        self.event.data.push_str(val);
                        self.event.data.push('\u{000A}');
                    }
                    "id" => {
                        if !val.contains('\u{0000}') {
                            self.event.id = val.to_string()
                        }
                    }
                    "retry" => {
                        if let Ok(val) = val.parse::<u64>() {
                            self.event.retry = Some(Duration::from_millis(val))
                        }
                    }
                    _ => {}
                }
            }
            RawEventLine::Comment(_) => {}
            RawEventLine::Empty => self.is_complete = true,
        }
    }

    /// From the HTML spec
    ///
    /// 1. Set the last event ID string of the event source to the value of the last event ID
    ///    buffer. The buffer does not get reset, so the last event ID string of the event source
    ///    remains set to this value until the next time it is set by the server.
    /// 2. If the data buffer is an empty string, set the data buffer and the event type buffer
    ///    to the empty string and return.
    /// 3. If the data buffer's last character is a U+000A LINE FEED (LF) character, then remove
    ///    the last character from the data buffer.
    /// 4. Let event be the result of creating an event using MessageEvent, in the relevant Realm
    ///    of the EventSource object.
    /// 5. Initialize event's type attribute to message, its data attribute to data, its origin
    ///    attribute to the serialization of the origin of the event stream's final URL (i.e., the
    ///    URL after redirects), and its lastEventId attribute to the last event ID string of the
    ///    event source.
    /// 6. If the event type buffer has a value other than the empty string, change the type of
    ///    the newly created event to equal the value of the event type buffer.
    /// 7. Set the data buffer and the event type buffer to the empty string.
    /// 8. Queue a task which, if the readyState attribute is set to a value other than CLOSED,
    ///    dispatches the newly created event at the EventSource object.
    fn dispatch(&mut self) -> Option<MessageEvent> {
        let builder = core::mem::take(self);
        let mut event = builder.event;
        self.event.id = event.id.clone();

        if event.data.is_empty() {
            return None;
        }

        if is_lf(event.data.chars().next_back().unwrap()) {
            event.data.pop();
        }

        if event.event.is_empty() {
            event.event = "message".to_string();
        }

        Some(event)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EventStreamState {
    NotStarted,
    Started,
    Terminated,
}

impl EventStreamState {
    fn is_terminated(self) -> bool {
        matches!(self, Self::Terminated)
    }
    fn is_started(self) -> bool {
        matches!(self, Self::Started)
    }
}

/// A Stream of events
pub struct EventStream {
    stream: Utf8Stream,
    buffer: String,
    builder: EventBuilder,
    state: EventStreamState,
    last_event_id: String,
}

impl EventStream {
    /// Initialize the EventStream with a Stream
    pub fn new(stream: InputStream) -> Self {
        Self {
            stream: Utf8Stream::new(stream),
            buffer: String::new(),
            builder: EventBuilder::default(),
            state: EventStreamState::NotStarted,
            last_event_id: String::new(),
        }
    }

    /// Set the last event ID of the stream. Useful for initializing the stream with a previous
    /// last event ID
    pub fn set_last_event_id(&mut self, id: impl Into<String>) {
        self.last_event_id = id.into();
    }

    /// Get the last event ID of the stream
    pub fn last_event_id(&self) -> &str {
        &self.last_event_id
    }

    pub fn subscribe(&self) -> Pollable {
        self.stream.subscribe()
    }

    pub fn poll_next(
        &mut self,
    ) -> Poll<Option<Result<MessageEvent, EventStreamError<StreamError>>>> {
        trace!("Polling for next event");

        match parse_event(&mut self.buffer, &mut self.builder) {
            Ok(Some(event)) => {
                self.last_event_id = event.id.clone();
                return Poll::Ready(Some(Ok(event)));
            }
            Err(err) => return Poll::Ready(Some(Err(err))),
            _ => {}
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

                    let slice = if self.state.is_started() {
                        &string
                    } else {
                        self.state = EventStreamState::Started;
                        if is_bom(string.chars().next().unwrap()) {
                            &string[1..]
                        } else {
                            &string
                        }
                    };
                    self.buffer.push_str(slice);

                    match parse_event(&mut self.buffer, &mut self.builder) {
                        Ok(Some(event)) => {
                            self.last_event_id = event.id.clone();
                            return Poll::Ready(Some(Ok(event)));
                        }
                        Err(err) => return Poll::Ready(Some(Err(err))),
                        _ => {}
                    }
                }
                Poll::Ready(Some(Err(err))) => return Poll::Ready(Some(Err(err.into()))),
                Poll::Ready(None) => {
                    self.state = EventStreamState::Terminated;
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Error thrown while parsing an event line
#[derive(Debug, PartialEq)]
pub enum EventStreamError<E> {
    /// Source stream is not valid UTF8
    Utf8(FromUtf8Error),
    /// Source stream is not a valid EventStream
    Parser(NomError<String>),
    /// Underlying source stream error
    Transport(E),
}

impl<E> From<Utf8StreamError<E>> for EventStreamError<E> {
    fn from(err: Utf8StreamError<E>) -> Self {
        match err {
            Utf8StreamError::Utf8(err) => Self::Utf8(err),
            Utf8StreamError::Transport(err) => Self::Transport(err),
        }
    }
}

impl<E> From<NomError<&str>> for EventStreamError<E> {
    fn from(err: NomError<&str>) -> Self {
        EventStreamError::Parser(NomError::new(err.input.to_string(), err.code))
    }
}

impl<E> fmt::Display for EventStreamError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Utf8(err) => f.write_fmt(format_args!("UTF8 error: {}", err)),
            Self::Parser(err) => f.write_fmt(format_args!("Parse error: {}", err)),
            Self::Transport(err) => f.write_fmt(format_args!("Transport error: {}", err)),
        }
    }
}

impl<E> std::error::Error for EventStreamError<E> where E: fmt::Display + fmt::Debug + Send + Sync {}

fn parse_event<E>(
    buffer: &mut String,
    builder: &mut EventBuilder,
) -> Result<Option<MessageEvent>, EventStreamError<E>> {
    if buffer.is_empty() {
        return Ok(None);
    }
    loop {
        match line(buffer.as_ref()) {
            Ok((rem, next_line)) => {
                builder.add(next_line);
                let consumed = buffer.len() - rem.len();
                let rem = buffer.split_off(consumed);
                *buffer = rem;
                if builder.is_complete {
                    if let Some(event) = builder.dispatch() {
                        return Ok(Some(event));
                    }
                }
            }
            Err(nom::Err::Incomplete(_)) => return Ok(None),
            Err(nom::Err::Error(err)) | Err(nom::Err::Failure(err)) => return Err(err.into()),
        }
    }
}
