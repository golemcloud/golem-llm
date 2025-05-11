use crate::client::{
    CreateModelResponseResponse, InputItem, OutputItem, ResponseOutputItemDone,
    ResponseOutputTextDelta, ResponsesApi,
};
use crate::conversions::{
    create_request, create_response_metadata, messages_to_input_items, parse_error_code,
    process_model_response, tool_defs_to_tools, tool_results_to_input_items,
};
use golem_llm::chat_stream::{LlmChatStream, LlmChatStreamState};
use golem_llm::config::with_config_key;
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, ContentPart, Error, ErrorCode, Guest, Message, StreamDelta,
    StreamEvent, ToolCall, ToolResult,
};
use golem_llm::LOGGING_STATE;
use golem_rust::wasm_rpc::Pollable;
use log::trace;
use std::cell::{Ref, RefCell, RefMut};

mod client;
mod conversions;

struct OpenAIChatStream {
    stream: RefCell<Option<EventSource>>,
    failure: Option<Error>,
    finished: RefCell<bool>,
}

impl OpenAIChatStream {
    pub fn new(stream: EventSource) -> LlmChatStream<Self> {
        LlmChatStream::new(OpenAIChatStream {
            stream: RefCell::new(Some(stream)),
            failure: None,
            finished: RefCell::new(false),
        })
    }

    pub fn failed(error: Error) -> LlmChatStream<Self> {
        LlmChatStream::new(OpenAIChatStream {
            stream: RefCell::new(None),
            failure: Some(error),
            finished: RefCell::new(false),
        })
    }
}

impl LlmChatStreamState for OpenAIChatStream {
    fn failure(&self) -> &Option<Error> {
        &self.failure
    }

    fn is_finished(&self) -> bool {
        *self.finished.borrow()
    }

    fn set_finished(&self) {
        *self.finished.borrow_mut() = true;
    }

    fn stream(&self) -> Ref<Option<EventSource>> {
        self.stream.borrow()
    }

    fn stream_mut(&self) -> RefMut<Option<EventSource>> {
        self.stream.borrow_mut()
    }

    fn decode_message(&self, raw: &str) -> Result<Option<StreamEvent>, String> {
        trace!("Received raw stream event: {raw}");
        let json: serde_json::Value = serde_json::from_str(raw)
            .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;

        let typ = json
            .as_object()
            .and_then(|obj| obj.get("type"))
            .and_then(|v| v.as_str());
        match typ {
            Some("response.failed") => {
                let response = json
                    .as_object()
                    .and_then(|obj| obj.get("response"))
                    .ok_or_else(|| {
                        "Unexpected stream event format, does not have 'response' field".to_string()
                    })?;
                let decoded =
                    serde_json::from_value::<CreateModelResponseResponse>(response.clone())
                        .map_err(|err| {
                            format!("Failed to deserialize stream event's response field: {err}")
                        })?;

                if let Some(error) = decoded.error {
                    Ok(Some(StreamEvent::Error(Error {
                        code: parse_error_code(error.code),
                        message: error.message,
                        provider_error_json: None,
                    })))
                } else {
                    Ok(Some(StreamEvent::Error(Error {
                        code: ErrorCode::InternalError,
                        message: "Unknown error".to_string(),
                        provider_error_json: None,
                    })))
                }
            }
            Some("response.completed") => {
                let response = json
                    .as_object()
                    .and_then(|obj| obj.get("response"))
                    .ok_or_else(|| {
                        "Unexpected stream event format, does not have 'response' field".to_string()
                    })?;
                let decoded =
                    serde_json::from_value::<CreateModelResponseResponse>(response.clone())
                        .map_err(|err| {
                            format!("Failed to deserialize stream event's response field: {err}")
                        })?;
                Ok(Some(StreamEvent::Finish(create_response_metadata(
                    &decoded,
                ))))
            }
            Some("response.output_text.delta") => {
                let decoded = serde_json::from_value::<ResponseOutputTextDelta>(json)
                    .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;
                Ok(Some(StreamEvent::Delta(StreamDelta {
                    content: Some(vec![ContentPart::Text(decoded.delta)]),
                    tool_calls: None,
                })))
            }
            Some("response.output_item.done") => {
                let decoded = serde_json::from_value::<ResponseOutputItemDone>(json)
                    .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;
                if let OutputItem::ToolCall {
                    arguments,
                    call_id,
                    name,
                    ..
                } = decoded.item
                {
                    Ok(Some(StreamEvent::Delta(StreamDelta {
                        content: None,
                        tool_calls: Some(vec![ToolCall {
                            id: call_id,
                            name,
                            arguments_json: arguments,
                        }]),
                    })))
                } else {
                    Ok(None)
                }
            }
            Some(_) => Ok(None),
            None => Err("Unexpected stream event format, does not have 'type' field".to_string()),
        }
    }
}

struct OpenAIComponent;

impl OpenAIComponent {
    const ENV_VAR_NAME: &'static str = "OPENAI_API_KEY";

    fn request(client: ResponsesApi, items: Vec<InputItem>, config: Config) -> ChatEvent {
        match tool_defs_to_tools(&config.tools) {
            Ok(tools) => {
                let request = create_request(items, config, tools);
                match client.create_model_response(request) {
                    Ok(response) => process_model_response(response),
                    Err(error) => ChatEvent::Error(error),
                }
            }
            Err(error) => ChatEvent::Error(error),
        }
    }

    fn streaming_request(
        client: ResponsesApi,
        items: Vec<InputItem>,
        config: Config,
    ) -> LlmChatStream<OpenAIChatStream> {
        match tool_defs_to_tools(&config.tools) {
            Ok(tools) => {
                let mut request = create_request(items, config, tools);
                request.stream = true;
                match client.stream_model_response(request) {
                    Ok(stream) => OpenAIChatStream::new(stream),
                    Err(error) => OpenAIChatStream::failed(error),
                }
            }
            Err(error) => OpenAIChatStream::failed(error),
        }
    }
}

impl Guest for OpenAIComponent {
    type ChatStream = LlmChatStream<OpenAIChatStream>;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |openai_api_key| {
            let client = ResponsesApi::new(openai_api_key);

            let items = messages_to_input_items(messages);
            Self::request(client, items, config)
        })
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |openai_api_key| {
            let client = ResponsesApi::new(openai_api_key);

            let mut items = messages_to_input_items(messages);
            items.extend(tool_results_to_input_items(tool_results));
            Self::request(client, items, config)
        })
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for OpenAIComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> Self::ChatStream {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(
            Self::ENV_VAR_NAME,
            OpenAIChatStream::failed,
            |openai_api_key| {
                let client = ResponsesApi::new(openai_api_key);

                let items = messages_to_input_items(messages);
                Self::streaming_request(client, items, config)
            },
        )
    }

    fn subscribe(stream: &Self::ChatStream) -> Pollable {
        stream.subscribe()
    }
}

type DurableOpenAIComponent = DurableLLM<OpenAIComponent>;

golem_llm::export_llm!(DurableOpenAIComponent with_types_in golem_llm);
