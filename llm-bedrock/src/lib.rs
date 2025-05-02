mod client;
mod conversions;
mod error;

use std::cell::RefCell;

use golem_llm::chat_stream::{LlmChatStream, LlmChatStreamState};
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, CompleteResponse, Config, ContentPart, Error, ErrorCode,
    Guest, Message, ResponseMetadata, Role, StreamDelta, StreamEvent, ToolCall, ToolResult, Usage,
};
use tokio::runtime::Builder;
use uuid::Uuid;

use crate::client::BedrockClient;
use crate::conversions::create_bedrock_client;
use crate::error::BedrockError;

pub struct BedrockChatStream {
    done: RefCell<bool>,
    failure: Option<Error>,
    response: Option<Message>,
    stream: RefCell<Option<golem_llm::event_source::EventSource>>,
}

impl BedrockChatStream {
    pub fn new(response: Message) -> Self {
        Self {
            done: RefCell::new(false),
            failure: None,
            response: Some(response),
            stream: RefCell::new(None),
        }
    }
}

impl LlmChatStreamState for BedrockChatStream {
    fn failure(&self) -> &Option<Error> {
        &self.failure
    }

    fn is_finished(&self) -> bool {
        *self.done.borrow()
    }

    fn set_finished(&self) {
        *self.done.borrow_mut() = true;
    }

    fn stream(&self) -> std::cell::Ref<Option<golem_llm::event_source::EventSource>> {
        self.stream.borrow()
    }

    fn stream_mut(&self) -> std::cell::RefMut<Option<golem_llm::event_source::EventSource>> {
        self.stream.borrow_mut()
    }

    fn decode_message(&self, _raw: &str) -> Result<Option<StreamEvent>, String> {
        if let Some(message) = &self.response {
            if let Some(ContentPart::Text(text)) = message.content.get(0) {
                return Ok(Some(StreamEvent::Delta(StreamDelta {
                    content: Some(vec![ContentPart::Text(text.clone())]),
                    tool_calls: None,
                })));
            }
        }
        Ok(None)
    }
}

pub struct BedrockComponent {
    client: BedrockClient,
}

impl BedrockComponent {
    pub async fn new(config: &[golem_llm::golem::llm::llm::Kv]) -> Result<Self, BedrockError> {
        let client = create_bedrock_client(config).await;
        Ok(Self { client })
    }
}

impl Guest for BedrockComponent {
    type ChatStream = LlmChatStream<BedrockChatStream>;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        let rt = match Builder::new_current_thread()
            .enable_all()
            .build() {
                Ok(rt) => rt,
                Err(e) => return ChatEvent::Error(Error {
                    code: ErrorCode::InternalError,
                    message: format!("Failed to create runtime: {:?}", e),
                    provider_error_json: None,
                }),
            };

        let component = match rt.block_on(Self::new(&config.provider_options)) {
            Ok(component) => component,
            Err(e) => return ChatEvent::Error(e.into()),
        };

        match rt.block_on(component.client.chat(&messages)) {
            Ok(response) => {
                let message: Message = response.into();
                ChatEvent::Message(CompleteResponse {
                    id: Uuid::new_v4().to_string(),
                    content: message.content,
                    tool_calls: vec![],
                    metadata: ResponseMetadata {
                        finish_reason: None,
                        usage: Some(Usage {
                            input_tokens: None,
                            output_tokens: None,
                            total_tokens: None,
                        }),
                        provider_id: Some(config.model),
                        timestamp: Some(chrono::Utc::now().to_rfc3339()),
                        provider_metadata_json: None,
                    },
                })
            }
            Err(e) => ChatEvent::Error(e.into()),
        }
    }

    fn continue_(messages: Vec<Message>, tool_results: Vec<(ToolCall, ToolResult)>, config: Config) -> ChatEvent {
        let mut all_messages = messages;
        for (tool_call, tool_result) in tool_results {
            all_messages.push(Message {
                role: Role::Assistant,
                name: None,
                content: vec![ContentPart::Text(format!("Tool call: {:?}", tool_call))],
            });
            all_messages.push(Message {
                role: Role::Tool,
                name: None,
                content: vec![ContentPart::Text(format!("Tool result: {:?}", tool_result))],
            });
        }
        Self::send(all_messages, config)
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        let rt = match Builder::new_current_thread()
            .enable_all()
            .build() {
                Ok(rt) => rt,
                Err(e) => panic!("Failed to create runtime: {:?}", e),
            };

        let component = match rt.block_on(Self::new(&config.provider_options)) {
            Ok(component) => component,
            Err(e) => panic!("Failed to create component: {:?}", e),
        };

        let response = match rt.block_on(component.client.chat(&messages)) {
            Ok(response) => response,
            Err(e) => panic!("Failed to chat: {:?}", e),
        };

        let message: Message = response.into();
        let stream = BedrockChatStream::new(message);
        ChatStream::new(LlmChatStream::new(stream))
    }
}

impl golem_llm::durability::ExtendedGuest for BedrockComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> Self::ChatStream {
        let rt = match Builder::new_current_thread()
            .enable_all()
            .build() {
                Ok(rt) => rt,
                Err(e) => panic!("Failed to create runtime: {:?}", e),
            };

        let component = match rt.block_on(Self::new(&config.provider_options)) {
            Ok(component) => component,
            Err(e) => panic!("Failed to create component: {:?}", e),
        };

        let response = match rt.block_on(component.client.chat(&messages)) {
            Ok(response) => response,
            Err(e) => panic!("Failed to chat: {:?}", e),
        };

        let message: Message = response.into();
        let stream = BedrockChatStream::new(message);
        LlmChatStream::new(stream)
    }
}

type DurableBedrockComponent = golem_llm::durability::DurableLLM<BedrockComponent>;

golem_llm::export_llm!(DurableBedrockComponent with_types_in golem_llm);

