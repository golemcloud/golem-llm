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

    fn retry_prompt(
        original_messages: &[Message],
        partial_result: &[llm::StreamDelta],
    ) -> Vec<Message> {
        let mut extended_messages = Vec::new();
        extended_messages.push(Message {
            role: llm::Role::System,
            name: None,
            content: vec![
                llm::ContentPart::Text(
                    "You were asked the same question previously, but the response was interrupted before completion. \
                     Please continue your response from where you left off. \
                     Do not include the part of the response that was already seen. If the response starts with a new word and no punctuation then add a space to the beginning".to_string()),
            ],
        });
        extended_messages.push(Message {
            role: llm::Role::User,
            name: None,
            content: vec![llm::ContentPart::Text(
                "Here is the original question:".to_string(),
            )],
        });
        extended_messages.extend_from_slice(original_messages);

        let mut partial_result_as_content = Vec::new();
        for delta in partial_result {
            if let Some(contents) = &delta.content {
                partial_result_as_content.extend_from_slice(contents);
            }
            if let Some(tool_calls) = &delta.tool_calls {
                for tool_call in tool_calls {
                    partial_result_as_content.push(llm::ContentPart::Text(format!(
                        "<tool-call id=\"{}\" name=\"{}\" arguments=\"{}\"/>",
                        tool_call.id, tool_call.name, tool_call.arguments_json,
                    )));
                }
            }
        }

        extended_messages.push(Message {
            role: llm::Role::User,
            name: None,
            content: vec![llm::ContentPart::Text(
                "Here is the partial response that was successfully received:".to_string(),
            )]
            .into_iter()
            .chain(partial_result_as_content)
            .collect(),
        });
        extended_messages
    }

    fn subscribe(_stream: &Self::ChatStream) -> golem_rust::wasm_rpc::Pollable {
        unimplemented!()
    }
}

fn get_bedrock_client() -> Result<Arc<Bedrock>, llm::Error> {
    BEDROCK_CLIENT.with_borrow_mut(|client_opt| match client_opt {
        Some(client) => Ok(client.clone()),
        None => {
            let client = Arc::new(Bedrock::new()?);
            *client_opt = Some(client.clone());
            Ok(client)
        }
    })
}

thread_local! {
    static BEDROCK_CLIENT: RefCell<Option<Arc<Bedrock>>> = const { RefCell::new(None) };
}

type DurableBedrockComponent = DurableLLM<BedrockComponent>;

golem_llm::export_llm!(DurableBedrockComponent with_types_in golem_llm);
