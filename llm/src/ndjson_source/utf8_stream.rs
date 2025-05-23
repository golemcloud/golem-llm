use golem_rust::bindings::wasi::io::streams::{InputStream, StreamError};
use golem_rust::wasm_rpc::Pollable;
use log::trace;
use std::string::FromUtf8Error;
use std::task::Poll;

pub struct Utf8Stream {
    subscription: Pollable,
    stream: InputStream,
    buffer: Vec<u8>,
    terminated: bool,
}

impl Utf8Stream {
    const CHUNK_SIZE: u64 = 1024;

    pub fn new(stream: InputStream) -> Self {
        let subscription = stream.subscribe();
        Self {
            stream,
            subscription,
            buffer: Vec::new(),
            terminated: false,
        }
    }

    pub fn subscribe(&self) -> Pollable {
        self.stream.subscribe()
    }

    pub fn poll_next(&mut self) -> Poll<Option<Result<String, Utf8StreamError<StreamError>>>> {
        if !self.terminated && self.subscription.ready() {
            match self.stream.read(Self::CHUNK_SIZE) {
                Ok(bytes) => {
                    trace!("Read {} bytes from NDJSON response stream", bytes.len());

                    self.buffer.extend_from_slice(bytes.as_ref());
                    let bytes = core::mem::take(&mut self.buffer);
                    match String::from_utf8(bytes) {
                        Ok(string) => Poll::Ready(Some(Ok(string))),
                        Err(err) => {
                            let valid_size = err.utf8_error().valid_up_to();
                            let mut bytes = err.into_bytes();
                            let rem = bytes.split_off(valid_size);
                            self.buffer = rem;
                            Poll::Ready(Some(Ok(unsafe { String::from_utf8_unchecked(bytes) })))
                        }
                    }
                }
                Err(StreamError::Closed) => {
                    trace!("NDJSON response stream closed");

                    self.terminated = true;
                    if self.buffer.is_empty() {
                        Poll::Ready(None)
                    } else {
                        Poll::Ready(Some(
                            String::from_utf8(core::mem::take(&mut self.buffer))
                                .map_err(Utf8StreamError::Utf8),
                        ))
                    }
                }
                Err(err) => Poll::Ready(Some(Err(Utf8StreamError::Transport(err)))),
            }
        } else {
            Poll::Pending
        }
    }
}

#[derive(Debug)]
pub enum Utf8StreamError<E> {
    Utf8(FromUtf8Error),
    Transport(E),
}

impl<E> From<FromUtf8Error> for Utf8StreamError<E> {
    fn from(err: FromUtf8Error) -> Self {
        Self::Utf8(err)
    }
}
