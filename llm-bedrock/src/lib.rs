use client::Bedrock;
use golem_llm::{
    durability::{DurableLLM, ExtendedGuest},
    golem::llm::llm::{ChatEvent, ChatStream, Config, Guest, Message, ToolCall, ToolResult},
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

        let bedrock = Bedrock::new();
        if let Err(err) = bedrock {
            return ChatEvent::Error(err);
        }

        bedrock.unwrap().converse(messages, config, None)
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let bedrock = Bedrock::new();
        if let Err(err) = bedrock {
            return ChatEvent::Error(err);
        }

        bedrock
            .unwrap()
            .converse(messages, config, Some(tool_results))
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

        let bedrock = Bedrock::new();
        if let Err(err) = bedrock {
            return BedrockChatStream::failed(err);
        }

        bedrock.unwrap().converse_stream(messages, config)
    }

    fn subscribe(_stream: &Self::ChatStream) -> golem_rust::wasm_rpc::Pollable {
        unimplemented!()
    }
}

type DurableBedrockComponent = DurableLLM<BedrockComponent>;

golem_llm::export_llm!(DurableBedrockComponent with_types_in golem_llm);
