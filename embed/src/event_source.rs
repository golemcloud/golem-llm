use log::{debug, trace};
use nom::bytes::complete::{tag, take_until};
use nom::combinator::opt;
use nom::sequence::tuple;
use nom::IResult;
use reqwest::Response;
use std::cell::RefCell;
use std::io::Read;

// Define our own Pollable trait instead of importing it
pub trait Pollable {
    fn ready(&self) -> bool;
}

pub struct EventSource {
    response: RefCell<Response>,
    buffer: RefCell<Vec<u8>>,
    current_offset: RefCell<usize>,
}

impl EventSource {
    pub fn new(response: Response) -> Self {
        Self {
            response: RefCell::new(response),
            buffer: RefCell::new(Vec::new()),
            current_offset: RefCell::new(0),
        }
    }

    pub fn poll_next_event(&self) -> Option<String> {
        let mut response = self.response.borrow_mut();
        let mut buffer = self.buffer.borrow_mut();
        let mut current_offset = self.current_offset.borrow_mut();

        // Try to parse an event from the existing buffer
        if let Some((new_offset, event)) = parse_event(&buffer, *current_offset) {
            *current_offset = new_offset;
            return Some(event);
        }

        // If we couldn't parse an event, read more data
        let mut chunk = [0u8; 1024];
        match response.read(&mut chunk) {
            Ok(0) => {
                // End of stream
                debug!("End of event stream");
                None
            }
            Ok(n) => {
                trace!("Read {} bytes from event stream", n);
                buffer.extend_from_slice(&chunk[..n]);
                if let Some((new_offset, event)) = parse_event(&buffer, *current_offset) {
                    *current_offset = new_offset;
                    Some(event)
                } else {
                    // Not enough data yet
                    None
                }
            }
            Err(e) => {
                debug!("Error reading from event stream: {}", e);
                None
            }
        }
    }
}

impl Pollable for EventSource {
    fn ready(&self) -> bool {
        self.poll_next_event().is_some()
    }
}

fn parse_event(buffer: &[u8], offset: usize) -> Option<(usize, String)> {
    if offset >= buffer.len() {
        return None;
    }

    let slice = &buffer[offset..];
    let result: IResult<&[u8], (&[u8], Option<&[u8]>)> =
        tuple((take_until("\n"), opt(tag("\n"))))(slice);

    match result {
        Ok((remaining, (line, _))) => {
            let new_offset = buffer.len() - remaining.len();
            if line.starts_with(b"data: ") {
                let data = &line[6..];
                if let Ok(s) = std::str::from_utf8(data) {
                    return Some((new_offset, s.to_string()));
                }
            }
            Some((new_offset, String::new()))
        }
        Err(_) => None,
    }
}