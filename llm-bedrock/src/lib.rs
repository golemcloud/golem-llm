use std::{cell::RefCell, sync::Arc};

use client::Bedrock;
use golem_llm::{
    durability::{DurableLLM, ExtendedGuest},
    golem::llm::llm::{self, ChatEvent, ChatStream, Config, Guest, Message, ToolCall, ToolResult},
    LOGGING_STATE,
};
use stream::BedrockChatStream;

mod client;
mod conversions;
mod stream;

struct BedrockComponent;

impl Guest for BedrockComponent {
    type ChatStream = BedrockChatStream;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let bedrock = get_bedrock_client();

        match bedrock {
            Ok(client) => client.converse(messages, config, None),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let bedrock = get_bedrock_client();

        match bedrock {
            Ok(client) => client.converse(messages, config, Some(tool_results)),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for BedrockComponent {
    fn unwrapped_stream(
        messages: Vec<golem_llm::golem::llm::llm::Message>,
        config: golem_llm::golem::llm::llm::Config,
    ) -> Self::ChatStream {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let bedrock = get_bedrock_client();

        match bedrock {
            Ok(client) => client.converse_stream(messages, config),
            Err(err) => BedrockChatStream::failed(err),
        }
    }

    fn subscribe(_stream: &Self::ChatStream) -> golem_rust::wasm_rpc::Pollable {
        unimplemented!()
    }
}

fn get_bedrock_client() -> Result<Arc<Bedrock>, llm::Error> {
    BEDROCK_CLIENT.with_borrow_mut(|client_opt| {
        if client_opt.is_none() {
            *client_opt = Some(Arc::new(Bedrock::new()?));
        }
        Ok(client_opt.as_ref().map(Arc::clone).unwrap())
    })
}

thread_local! {
    static BEDROCK_CLIENT: RefCell<Option<Arc<Bedrock>>> = const { RefCell::new(None) };
}

type DurableBedrockComponent = DurableLLM<BedrockComponent>;

golem_llm::export_llm!(DurableBedrockComponent with_types_in golem_llm);
