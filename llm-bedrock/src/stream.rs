use aws_sdk_bedrockruntime::{
    self as bedrock, primitives::event_stream::EventReceiver,
    types::error::ConverseStreamOutputError,
};
use golem_llm::golem::llm::llm;
use std::cell::{RefCell, RefMut};

use crate::{
    client::get_async_runtime,
    conversions::{converse_stream_output_to_stream_event, custom_error},
};

type BedrockEventSource =
    EventReceiver<bedrock::types::ConverseStreamOutput, ConverseStreamOutputError>;

pub struct BedrockChatStream {
    stream: RefCell<Option<BedrockEventSource>>,
    failure: Option<llm::Error>,
    finished: RefCell<bool>,
}

impl BedrockChatStream {
    pub fn new(stream: BedrockEventSource) -> BedrockChatStream {
        BedrockChatStream {
            stream: RefCell::new(Some(stream)),
            failure: None,
            finished: RefCell::new(false),
        }
    }

    pub fn failed(error: llm::Error) -> BedrockChatStream {
        BedrockChatStream {
            stream: RefCell::new(None),
            failure: Some(error),
            finished: RefCell::new(true),
        }
    }

    fn stream_mut(&self) -> RefMut<Option<BedrockEventSource>> {
        self.stream.borrow_mut()
    }

    fn failure(&self) -> &Option<llm::Error> {
        &self.failure
    }

    fn is_finished(&self) -> bool {
        *self.finished.borrow()
    }

    fn set_finished(&self) {
        *self.finished.borrow_mut() = true;
    }
}

impl llm::GuestChatStream for BedrockChatStream {
    fn get_next(&self) -> Option<Vec<llm::StreamEvent>> {
        if self.is_finished() {
            return Some(vec![]);
        }

        if let Some(stream) = self.stream_mut().as_mut() {
            let runtime = get_async_runtime();

            runtime.block_on(async {
                let token = stream.recv().await;

                match token {
                    Ok(Some(output)) => {
                        log::trace!("Processing bedrock stream event: {output:?}");
                        converse_stream_output_to_stream_event(output)
                    }
                    Ok(None) => {
                        self.set_finished();
                        Some(vec![])
                    }
                    Err(error) => {
                        self.set_finished();
                        Some(vec![llm::StreamEvent::Error(custom_error(
                            llm::ErrorCode::InternalError,
                            format!("An error occurred while reading event stream: {error}"),
                        ))])
                    }
                }
            })
        } else if let Some(error) = self.failure() {
            self.set_finished();
            Some(vec![llm::StreamEvent::Error(error.clone())])
        } else {
            None
        }
    }

    fn blocking_get_next(&self) -> Vec<llm::StreamEvent> {
        let mut result = Vec::new();
        loop {
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
