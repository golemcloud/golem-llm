use crate::golem::llm::llm::{Config, ContentPart, Guest, Message, Role, StreamDelta};
use golem_rust::wasm_rpc::Pollable;
use std::marker::PhantomData;

/// Wraps an LLM implementation with custom durability
pub struct DurableLLM<Impl> {
    phantom: PhantomData<Impl>,
}

/// Trait to be implemented in addition to the LLM `Guest` trait when wrapping it with `DurableLLM`.
pub trait ExtendedGuest: Guest + 'static {
    /// Creates an instance of the LLM specific `ChatStream` without wrapping it in a `Resource`
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> Self::ChatStream;

    /// Creates the retry prompt with a combination of the original messages, and the partially received
    /// streaming responses. There is a default implementation here, but it can be overridden with provider-specific
    /// prompts if needed.
    fn retry_prompt(original_messages: &[Message], partial_result: &[StreamDelta]) -> Vec<Message> {
        let mut extended_messages = Vec::new();
        extended_messages.push(Message {
            role: Role::System,
            name: None,
            content: vec![
                ContentPart::Text(
                    "You were asked the same question previously, but the response was interrupted before completion. \
                                        Please continue your response from where you left off. \
                                        Do not include the part of the response that was already seen.".to_string()),
                ContentPart::Text("Here is the original question:".to_string()),
            ],
        });
        extended_messages.extend_from_slice(original_messages);

        let mut partial_result_as_content = Vec::new();
        for delta in partial_result {
            if let Some(contents) = &delta.content {
                partial_result_as_content.extend_from_slice(contents);
            }
            if let Some(tool_calls) = &delta.tool_calls {
                for tool_call in tool_calls {
                    partial_result_as_content.push(ContentPart::Text(format!(
                        "<tool-call id=\"{}\" name=\"{}\" arguments=\"{}\"/>",
                        tool_call.id, tool_call.name, tool_call.arguments_json,
                    )));
                }
            }
        }

        extended_messages.push(Message {
            role: Role::System,
            name: None,
            content: vec![ContentPart::Text(
                "Here is the partial response that was successfully received:".to_string(),
            )]
            .into_iter()
            .chain(partial_result_as_content)
            .collect(),
        });
        extended_messages
    }

    fn subscribe(stream: &Self::ChatStream) -> Pollable;
}

/// When the durability feature flag is off, wrapping with `DurableLLM` is just a passthrough
#[cfg(not(feature = "durability"))]
mod passthrough_impl {
    use crate::durability::{DurableLLM, ExtendedGuest};
    use crate::golem::llm::llm::{
        ChatEvent, ChatStream, Config, Guest, Message, ToolCall, ToolResult,
    };

    impl<Impl: ExtendedGuest> Guest for DurableLLM<Impl> {
        type ChatStream = Impl::ChatStream;

        fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
            Impl::send(messages, config)
        }

        fn continue_(
            messages: Vec<Message>,
            tool_results: Vec<(ToolCall, ToolResult)>,
            config: Config,
        ) -> ChatEvent {
            Impl::continue_(messages, tool_results, config)
        }

        fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
            Impl::stream(messages, config)
        }
    }
}

/// When the durability feature flag is on, wrapping with `DurableLLM` adds custom durability
/// on top of the provider-specific LLM implementation using Golem's special host functions and
/// the `golem-rust` helper library.
///
/// There will be custom durability entries saved in the oplog, with the full LLM request and configuration
/// stored as input, and the full response stored as output. To serialize these in a way it is
/// observable by oplog consumers, each relevant data type has to be converted to/from `ValueAndType`
/// which is implemented using the type classes and builder in the `golem-rust` library.
#[cfg(feature = "durability")]
mod durable_impl {
    use crate::durability::{DurableLLM, ExtendedGuest};
    use crate::golem::llm::llm::{
        ChatEvent, ChatStream, Config, Guest, GuestChatStream, Message, StreamDelta, StreamEvent,
        ToolCall, ToolResult,
    };
    use golem_rust::bindings::golem::durability::durability::{
        DurableFunctionType, LazyInitializedPollable,
    };
    use golem_rust::durability::Durability;
    use golem_rust::wasm_rpc::Pollable;
    use golem_rust::{with_persistence_level, FromValueAndType, IntoValue, PersistenceLevel};
    use std::cell::RefCell;
    use std::fmt::{Display, Formatter};

    impl<Impl: ExtendedGuest> Guest for DurableLLM<Impl> {
        type ChatStream = DurableChatStream<Impl>;

        fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
            let durability = Durability::<ChatEvent, UnusedError>::new(
                "golem_llm",
                "send",
                DurableFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    Impl::send(messages.clone(), config.clone())
                });
                durability.persist_infallible(SendInput { messages, config }, result)
            } else {
                durability.replay_infallible()
            }
        }

        fn continue_(
            messages: Vec<Message>,
            tool_results: Vec<(ToolCall, ToolResult)>,
            config: Config,
        ) -> ChatEvent {
            let durability = Durability::<ChatEvent, UnusedError>::new(
                "golem_llm",
                "continue",
                DurableFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    Impl::continue_(messages.clone(), tool_results.clone(), config.clone())
                });
                durability.persist_infallible(
                    ContinueInput {
                        messages,
                        tool_results,
                        config,
                    },
                    result,
                )
            } else {
                durability.replay_infallible()
            }
        }

        fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
            let durability = Durability::<NoOutput, UnusedError>::new(
                "golem_llm",
                "stream",
                DurableFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    ChatStream::new(DurableChatStream::<Impl>::live(Impl::unwrapped_stream(
                        messages.clone(),
                        config.clone(),
                    )))
                });
                let _ = durability.persist_infallible(SendInput { messages, config }, NoOutput);
                result
            } else {
                let _: NoOutput = durability.replay_infallible();
                ChatStream::new(DurableChatStream::<Impl>::replay(messages, config))
            }
        }
    }

    /// Represents the durable chat stream's state
    ///
    /// In live mode it directly calls the underlying LLM stream which is implemented on
    /// top of an SSE parser using the wasi-http response body stream.
    ///
    /// In replay mode it buffers the replayed messages, and also tracks the created pollables
    /// to be able to reattach them to the new live stream when the switch to live mode
    /// happens.
    ///
    /// When reaching the end of the replay mode, if the replayed stream was not finished yet,
    /// the replay prompt implemented in `ExtendedGuest` is used to create a new LLM response
    /// stream and continue the response seamlessly.
    enum DurableChatStreamState<Impl: ExtendedGuest> {
        Live {
            stream: Impl::ChatStream,
            pollables: Vec<LazyInitializedPollable>,
        },
        Replay {
            original_messages: Vec<Message>,
            config: Config,
            pollables: Vec<LazyInitializedPollable>,
            partial_result: Vec<StreamDelta>,
            finished: bool,
        },
    }

    pub struct DurableChatStream<Impl: ExtendedGuest> {
        state: RefCell<Option<DurableChatStreamState<Impl>>>,
        subscription: RefCell<Option<Pollable>>,
    }

    impl<Impl: ExtendedGuest> DurableChatStream<Impl> {
        fn live(stream: Impl::ChatStream) -> Self {
            Self {
                state: RefCell::new(Some(DurableChatStreamState::Live {
                    stream,
                    pollables: Vec::new(),
                })),
                subscription: RefCell::new(None),
            }
        }

        fn replay(original_messages: Vec<Message>, config: Config) -> Self {
            Self {
                state: RefCell::new(Some(DurableChatStreamState::Replay {
                    original_messages,
                    config,
                    pollables: Vec::new(),
                    partial_result: Vec::new(),
                    finished: false,
                })),
                subscription: RefCell::new(None),
            }
        }

        fn subscribe(&self) -> Pollable {
            let mut state = self.state.borrow_mut();
            match &mut *state {
                Some(DurableChatStreamState::Live { stream, .. }) => Impl::subscribe(stream),
                Some(DurableChatStreamState::Replay { pollables, .. }) => {
                    let lazy_pollable = LazyInitializedPollable::new();
                    let pollable = lazy_pollable.subscribe();
                    pollables.push(lazy_pollable);
                    pollable
                }
                None => {
                    unreachable!()
                }
            }
        }
    }

    impl<Impl: ExtendedGuest> Drop for DurableChatStream<Impl> {
        fn drop(&mut self) {
            let _ = self.subscription.take();
            match self.state.take() {
                Some(DurableChatStreamState::Live {
                    mut pollables,
                    stream,
                }) => {
                    with_persistence_level(PersistenceLevel::PersistNothing, move || {
                        pollables.clear();
                        drop(stream);
                    });
                }
                Some(DurableChatStreamState::Replay { mut pollables, .. }) => {
                    pollables.clear();
                }
                None => {}
            }
        }
    }

    impl<Impl: ExtendedGuest> GuestChatStream for DurableChatStream<Impl> {
        fn get_next(&self) -> Option<Vec<StreamEvent>> {
            let durability = Durability::<Option<Vec<StreamEvent>>, UnusedError>::new(
                "golem_llm",
                "get_next",
                DurableFunctionType::ReadRemote,
            );
            if durability.is_live() {
                let mut state = self.state.borrow_mut();
                let (result, new_live_stream) = match &*state {
                    Some(DurableChatStreamState::Live { stream, .. }) => {
                        let result =
                            with_persistence_level(PersistenceLevel::PersistNothing, || {
                                stream.get_next()
                            });
                        (durability.persist_infallible(NoInput, result.clone()), None)
                    }
                    Some(DurableChatStreamState::Replay {
                        original_messages,
                        config,
                        pollables,
                        partial_result,
                        finished,
                    }) => {
                        if *finished {
                            (None, None)
                        } else {
                            let extended_messages =
                                Impl::retry_prompt(original_messages, partial_result);

                            let (stream, first_live_result) =
                                with_persistence_level(PersistenceLevel::PersistNothing, || {
                                    let stream = <Impl as ExtendedGuest>::unwrapped_stream(
                                        extended_messages,
                                        config.clone(),
                                    );

                                    for lazy_initialized_pollable in pollables {
                                        lazy_initialized_pollable.set(Impl::subscribe(&stream));
                                    }

                                    let next = stream.get_next();
                                    (stream, next)
                                });
                            durability.persist_infallible(NoInput, first_live_result.clone());

                            (first_live_result, Some(stream))
                        }
                    }
                    None => {
                        unreachable!()
                    }
                };

                if let Some(stream) = new_live_stream {
                    let pollables = match state.take() {
                        Some(DurableChatStreamState::Live { pollables, .. }) => pollables,
                        Some(DurableChatStreamState::Replay { pollables, .. }) => pollables,
                        None => {
                            unreachable!()
                        }
                    };
                    *state = Some(DurableChatStreamState::Live { stream, pollables });
                }

                result
            } else {
                let result: Option<Vec<StreamEvent>> = durability.replay_infallible();
                let mut state = self.state.borrow_mut();
                match &mut *state {
                    Some(DurableChatStreamState::Live { .. }) => {
                        unreachable!("Durable chat stream cannot be in live mode during replay")
                    }
                    Some(DurableChatStreamState::Replay {
                        partial_result,
                        finished,
                        ..
                    }) => {
                        if let Some(result) = &result {
                            for event in result {
                                match event {
                                    StreamEvent::Delta(delta) => {
                                        partial_result.push(delta.clone());
                                    }
                                    StreamEvent::Finish(_) => {
                                        *finished = true;
                                    }
                                    StreamEvent::Error(_) => {
                                        *finished = true;
                                    }
                                }
                            }
                        }
                    }
                    None => {
                        unreachable!()
                    }
                }
                result
            }
        }

        fn blocking_get_next(&self) -> Vec<StreamEvent> {
            let mut subscription = self.subscription.borrow_mut();
            if subscription.is_none() {
                *subscription = Some(self.subscribe());
            }
            let subscription = subscription.as_mut().unwrap();
            let mut result = Vec::new();
            loop {
                subscription.block();
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

    #[derive(Debug, Clone, PartialEq, IntoValue)]
    struct SendInput {
        messages: Vec<Message>,
        config: Config,
    }

    #[derive(Debug, IntoValue)]
    struct ContinueInput {
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    }

    #[derive(Debug, IntoValue)]
    struct NoInput;

    #[derive(Debug, Clone, FromValueAndType, IntoValue)]
    struct NoOutput;

    #[derive(Debug, FromValueAndType, IntoValue)]
    struct UnusedError;

    impl Display for UnusedError {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "UnusedError")
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::durability::durable_impl::SendInput;
        use crate::golem::llm::llm::{
            ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, FinishReason,
            ImageDetail, ImageReference, ImageSource, ImageUrl, Message, ResponseMetadata, Role,
            ToolCall, Usage,
        };
        use golem_rust::value_and_type::{FromValueAndType, IntoValueAndType};
        use golem_rust::wasm_rpc::WitTypeNode;
        use std::fmt::Debug;

        fn roundtrip_test<T: Debug + Clone + PartialEq + IntoValueAndType + FromValueAndType>(
            value: T,
        ) {
            let vnt = value.clone().into_value_and_type();
            let extracted = T::from_value_and_type(vnt).unwrap();
            assert_eq!(value, extracted);
        }

        #[test]
        fn image_detail_roundtrip() {
            roundtrip_test(ImageDetail::Low);
            roundtrip_test(ImageDetail::High);
            roundtrip_test(ImageDetail::Auto);
        }

        #[test]
        fn error_roundtrip() {
            roundtrip_test(Error {
                code: ErrorCode::InvalidRequest,
                message: "Invalid request".to_string(),
                provider_error_json: Some("Provider error".to_string()),
            });
            roundtrip_test(Error {
                code: ErrorCode::AuthenticationFailed,
                message: "Authentication failed".to_string(),
                provider_error_json: None,
            });
        }

        #[test]
        fn image_url_roundtrip() {
            roundtrip_test(ImageUrl {
                url: "https://example.com/image.png".to_string(),
                detail: Some(ImageDetail::High),
            });
            roundtrip_test(ImageUrl {
                url: "https://example.com/image.png".to_string(),
                detail: None,
            });
        }

        #[test]
        fn image_source_roundtrip() {
            roundtrip_test(ImageSource {
                data: vec![0, 1, 2, 3, 4, 5],
                mime_type: "image/jpeg".to_string(),
                detail: Some(ImageDetail::High),
            });
            roundtrip_test(ImageSource {
                data: vec![255, 254, 253, 252],
                mime_type: "image/png".to_string(),
                detail: None,
            });
        }

        #[test]
        fn content_part_roundtrip() {
            roundtrip_test(ContentPart::Text("Hello".to_string()));
            roundtrip_test(ContentPart::Image(ImageReference::Url(ImageUrl {
                url: "https://example.com/image.png".to_string(),
                detail: Some(ImageDetail::Low),
            })));
            roundtrip_test(ContentPart::Image(ImageReference::Inline(ImageSource {
                data: vec![0, 1, 2, 3, 4, 5],
                mime_type: "image/jpeg".to_string(),
                detail: Some(ImageDetail::Auto),
            })));
        }

        #[test]
        fn usage_roundtrip() {
            roundtrip_test(Usage {
                input_tokens: Some(100),
                output_tokens: Some(200),
                total_tokens: Some(300),
            });
            roundtrip_test(Usage {
                input_tokens: None,
                output_tokens: None,
                total_tokens: None,
            });
        }

        #[test]
        fn response_metadata_roundtrip() {
            roundtrip_test(ResponseMetadata {
                finish_reason: Some(FinishReason::Stop),
                usage: Some(Usage {
                    input_tokens: Some(100),
                    output_tokens: None,
                    total_tokens: Some(100),
                }),
                provider_id: Some("provider_id".to_string()),
                timestamp: Some("2023-10-01T00:00:00Z".to_string()),
                provider_metadata_json: Some("{\"key\": \"value\"}".to_string()),
            });
            roundtrip_test(ResponseMetadata {
                finish_reason: None,
                usage: None,
                provider_id: None,
                timestamp: None,
                provider_metadata_json: None,
            });
        }

        #[test]
        fn complete_response_roundtrip() {
            roundtrip_test(CompleteResponse {
                id: "response_id".to_string(),
                content: vec![
                    ContentPart::Text("Hello".to_string()),
                    ContentPart::Image(ImageReference::Url(ImageUrl {
                        url: "https://example.com/image.png".to_string(),
                        detail: Some(ImageDetail::High),
                    })),
                ],
                tool_calls: vec![ToolCall {
                    id: "x".to_string(),
                    name: "y".to_string(),
                    arguments_json: "\"z\"".to_string(),
                }],
                metadata: ResponseMetadata {
                    finish_reason: Some(FinishReason::Stop),
                    usage: None,
                    provider_id: None,
                    timestamp: None,
                    provider_metadata_json: None,
                },
            });
        }

        #[test]
        fn chat_event_roundtrip() {
            roundtrip_test(ChatEvent::Message(CompleteResponse {
                id: "response_id".to_string(),
                content: vec![
                    ContentPart::Text("Hello".to_string()),
                    ContentPart::Image(ImageReference::Url(ImageUrl {
                        url: "https://example.com/image.png".to_string(),
                        detail: Some(ImageDetail::High),
                    })),
                ],
                tool_calls: vec![ToolCall {
                    id: "x".to_string(),
                    name: "y".to_string(),
                    arguments_json: "\"z\"".to_string(),
                }],
                metadata: ResponseMetadata {
                    finish_reason: Some(FinishReason::Stop),
                    usage: None,
                    provider_id: None,
                    timestamp: None,
                    provider_metadata_json: None,
                },
            }));
            roundtrip_test(ChatEvent::ToolRequest(vec![ToolCall {
                id: "x".to_string(),
                name: "y".to_string(),
                arguments_json: "\"z\"".to_string(),
            }]));
            roundtrip_test(ChatEvent::Error(Error {
                code: ErrorCode::InvalidRequest,
                message: "Invalid request".to_string(),
                provider_error_json: Some("Provider error".to_string()),
            }));
        }

        #[test]
        fn send_input_encoding() {
            let input = SendInput {
                messages: vec![
                    Message {
                        role: Role::User,
                        name: Some("user".to_string()),
                        content: vec![ContentPart::Text("Hello".to_string())],
                    },
                    Message {
                        role: Role::Assistant,
                        name: None,
                        content: vec![ContentPart::Image(ImageReference::Url(ImageUrl {
                            url: "https://example.com/image.png".to_string(),
                            detail: Some(ImageDetail::High),
                        }))],
                    },
                    Message {
                        role: Role::User,
                        name: None,
                        content: vec![
                            ContentPart::Text("Analyze this image:".to_string()),
                            ContentPart::Image(ImageReference::Inline(ImageSource {
                                data: vec![0, 1, 2, 3, 4, 5],
                                mime_type: "image/jpeg".to_string(),
                                detail: None,
                            })),
                        ],
                    },
                ],
                config: Config {
                    model: "gpt-3.5-turbo".to_string(),
                    temperature: Some(0.7),
                    max_tokens: Some(100),
                    stop_sequences: Some(vec!["\n".to_string()]),
                    tools: vec![],
                    tool_choice: None,
                    provider_options: vec![],
                },
            };

            let encoded = input.into_value_and_type();
            println!("{encoded:#?}");

            for wit_type in encoded.typ.nodes {
                if let WitTypeNode::ListType(idx) = wit_type {
                    assert!(idx >= 0);
                }
            }
        }
    }
}
