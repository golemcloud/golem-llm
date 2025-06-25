use aws_sdk_bedrockruntime::{
    self as bedrock, primitives::event_stream::EventReceiver,
    types::error::ConverseStreamOutputError,
};
use golem_llm::golem::llm::llm;
use std::cell::{RefCell, RefMut};

use crate::{
    client::get_async_runtime,
    conversions::{converse_stream_output_to_stream_event, custom_error, merge_metadata},
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
    fn get_single_event(&self) -> Option<llm::StreamEvent> {
        if let Some(stream) = self.stream_mut().as_mut() {
            let runtime = get_async_runtime();

            runtime.block_on(async move {
                let token = stream.recv().await;
                log::trace!("Bedrock stream event: {token:?}");

                match token {
                    Ok(Some(output)) => {
                        log::trace!("Processing bedrock stream event: {output:?}");
                        converse_stream_output_to_stream_event(output)
                    }
                    Ok(None) => {
                        log::trace!("running set_finished on stream due to None event received");
                        self.set_finished();
                        None
                    }
                    Err(error) => {
                        log::trace!("running set_finished on stream due to error: {error:?}");
                        self.set_finished();
                        Some(llm::StreamEvent::Error(custom_error(
                            llm::ErrorCode::InternalError,
                            format!("An error occurred while reading event stream: {error}"),
                        )))
                    }
                }
            })
        } else if let Some(error) = self.failure() {
            self.set_finished();
            Some(llm::StreamEvent::Error(error.clone()))
        } else {
            None
        }
    }
}

impl llm::GuestChatStream for BedrockChatStream {
    fn get_next(&self) -> Option<Vec<llm::StreamEvent>> {
        if self.is_finished() {
            return Some(vec![]);
        }
        self.get_single_event().map(|event| {
            if let llm::StreamEvent::Finish(metadata) = event.clone() {
                if let Some(llm::StreamEvent::Finish(final_metadata)) = self.get_single_event() {
                    return vec![llm::StreamEvent::Finish(merge_metadata(
                        metadata,
                        final_metadata,
                    ))];
                }
            }
            vec![event]
        })
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
