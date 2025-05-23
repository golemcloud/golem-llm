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
        ChatEvent, ChatStream, CompleteResponse, Config, ContentPart, Error, ErrorCode,
        FinishReason, Guest, GuestChatStream, ImageDetail, ImageReference, ImageSource, ImageUrl,
        Kv, Message, ResponseMetadata, Role, StreamDelta, StreamEvent, ToolCall, ToolDefinition,
        ToolFailure, ToolResult, ToolSuccess, Usage,
    };
    use golem_rust::bindings::golem::durability::durability::{
        DurableFunctionType, LazyInitializedPollable,
    };
    use golem_rust::durability::Durability;
    use golem_rust::value_and_type::type_builder::TypeNodeBuilder;
    use golem_rust::value_and_type::{FromValueAndType, IntoValue};
    use golem_rust::wasm_rpc::{NodeBuilder, Pollable, WitValueExtractor};
    use golem_rust::{with_persistence_level, PersistenceLevel};
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

    // variant stream-event {
    //   delta(stream-delta),
    //   finish(response-metadata),
    //   error(error),
    // }
    impl IntoValue for StreamEvent {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                StreamEvent::Delta(stream_delta) => {
                    let builder = builder.variant(0);
                    stream_delta.add_to_builder(builder).finish()
                }
                StreamEvent::Finish(response_metadata) => {
                    let builder = builder.variant(1);
                    response_metadata.add_to_builder(builder).finish()
                }
                StreamEvent::Error(error) => {
                    let builder = builder.variant(2);
                    error.add_to_builder(builder).finish()
                }
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.variant();
            builder = StreamDelta::add_to_type_builder(builder.case("delta"));
            builder = ResponseMetadata::add_to_type_builder(builder.case("finish"));
            builder = Error::add_to_type_builder(builder.case("error"));
            builder.finish()
        }
    }

    impl FromValueAndType for StreamEvent {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            match extractor.variant() {
                Some((0, inner)) => Ok(StreamEvent::Delta(StreamDelta::from_extractor(
                    &inner.ok_or_else(|| "Missing stream-delta body".to_string())?,
                )?)),
                Some((1, inner)) => Ok(StreamEvent::Finish(ResponseMetadata::from_extractor(
                    &inner.ok_or_else(|| "Missing response-metadata body".to_string())?,
                )?)),
                Some((2, inner)) => Ok(StreamEvent::Error(Error::from_extractor(
                    &inner.ok_or_else(|| "Missing error body".to_string())?,
                )?)),
                _ => Err("StreamEvent is not a variant".to_string()),
            }
        }
    }

    // record stream-delta {
    //   content: option<list<content-part>>,
    //   tool-calls: option<list<tool-call>>,
    // }
    impl IntoValue for StreamDelta {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.content.add_to_builder(builder.item());
            builder = self.tool_calls.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = TypeNodeBuilder::finish(
                ContentPart::add_to_type_builder(builder.field("content").option().list()).finish(),
            );
            builder = TypeNodeBuilder::finish(
                ToolCall::add_to_type_builder(builder.field("tool-calls").option().list()).finish(),
            );
            builder.finish()
        }
    }

    impl FromValueAndType for StreamDelta {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                content: Option::<Vec<ContentPart>>::from_extractor(
                    &extractor
                        .field(0)
                        .ok_or_else(|| "Missing content field".to_string())?,
                )?,
                tool_calls: Option::<Vec<ToolCall>>::from_extractor(
                    &extractor
                        .field(1)
                        .ok_or_else(|| "Missing tool-calls field".to_string())?,
                )?,
            })
        }
    }

    //   variant chat-event {
    //     message(complete-response),
    //     tool-request(list<tool-call>),
    //     error(error),
    //   }
    impl IntoValue for ChatEvent {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                ChatEvent::Message(complete_response) => {
                    let builder = builder.variant(0);
                    complete_response.add_to_builder(builder).finish()
                }
                ChatEvent::ToolRequest(tool_calls) => {
                    let builder = builder.variant(1);
                    tool_calls.add_to_builder(builder).finish()
                }
                ChatEvent::Error(error) => {
                    let builder = builder.variant(2);
                    error.add_to_builder(builder).finish()
                }
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.variant();
            builder = CompleteResponse::add_to_type_builder(builder.case("message"));
            builder = Vec::<ToolCall>::add_to_type_builder(builder.case("tool-request"));
            builder = Error::add_to_type_builder(builder.case("error"));
            builder.finish()
        }
    }

    impl FromValueAndType for ChatEvent {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            let (idx, inner) = extractor
                .variant()
                .ok_or_else(|| "ChatEvent should be variant".to_string())?;
            match idx {
                0 => Ok(ChatEvent::Message(CompleteResponse::from_extractor(
                    &inner.ok_or_else(|| "Missing message body".to_string())?,
                )?)),
                1 => Ok(ChatEvent::ToolRequest(Vec::<ToolCall>::from_extractor(
                    &inner.ok_or_else(|| "Missing tool request body".to_string())?,
                )?)),
                2 => Ok(ChatEvent::Error(Error::from_extractor(
                    &inner.ok_or_else(|| "Missing error body".to_string())?,
                )?)),
                _ => Err(format!("Invalid ChatEvent variant: {idx}")),
            }
        }
    }

    //   record complete-response {
    //     id: string,
    //     content: list<content-part>,
    //     tool-calls: list<tool-call>,
    //     metadata: response-metadata,
    //   }
    impl IntoValue for CompleteResponse {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.id.add_to_builder(builder.item());
            builder = self.content.add_to_builder(builder.item());
            builder = self.tool_calls.add_to_builder(builder.item());
            builder = self.metadata.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("id").string();
            builder = Vec::<ContentPart>::add_to_type_builder(builder.field("content"));
            builder = Vec::<ToolCall>::add_to_type_builder(builder.field("tool-calls"));
            builder = ResponseMetadata::add_to_type_builder(builder.field("metadata"));
            builder.finish()
        }
    }

    impl FromValueAndType for CompleteResponse {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                id: String::from_extractor(
                    &extractor
                        .field(0)
                        .ok_or_else(|| "Missing id field".to_string())?,
                )?,
                content: Vec::<ContentPart>::from_extractor(
                    &extractor
                        .field(1)
                        .ok_or_else(|| "Missing content field".to_string())?,
                )?,
                tool_calls: Vec::<ToolCall>::from_extractor(
                    &extractor
                        .field(2)
                        .ok_or_else(|| "Missing tool-calls field".to_string())?,
                )?,
                metadata: ResponseMetadata::from_extractor(
                    &extractor
                        .field(3)
                        .ok_or_else(|| "Missing metadata field".to_string())?,
                )?,
            })
        }
    }

    //   record tool-call {
    //     id: string,
    //     name: string,
    //     arguments-json: string,
    //   }
    impl IntoValue for ToolCall {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.id.add_to_builder(builder.item());
            builder = self.name.add_to_builder(builder.item());
            builder = self.arguments_json.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("id").string();
            builder = builder.field("name").string();
            builder = builder.field("arguments-json").string();
            builder.finish()
        }
    }

    impl FromValueAndType for ToolCall {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                id: String::from_extractor(
                    &extractor
                        .field(0)
                        .ok_or_else(|| "Missing id field".to_string())?,
                )?,
                name: String::from_extractor(
                    &extractor
                        .field(1)
                        .ok_or_else(|| "Missing name field".to_string())?,
                )?,
                arguments_json: String::from_extractor(
                    &extractor
                        .field(2)
                        .ok_or_else(|| "Missing arguments-json field".to_string())?,
                )?,
            })
        }
    }

    // variant tool-result {
    //  success(tool-success),
    //  error(tool-failure),
    //}
    impl IntoValue for ToolResult {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                ToolResult::Success(success) => {
                    let builder = builder.variant(0);
                    success.add_to_builder(builder).finish()
                }
                ToolResult::Error(error) => {
                    let builder = builder.variant(1);
                    error.add_to_builder(builder).finish()
                }
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.variant();
            builder = ToolSuccess::add_to_type_builder(builder.case("success"));
            builder = ToolFailure::add_to_type_builder(builder.case("error"));
            builder.finish()
        }
    }

    // record tool-success {
    //   id: string,
    //   name: string,
    //   result-json: string,
    //   execution-time-ms: option<u32>,
    // }
    impl IntoValue for ToolSuccess {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.id.add_to_builder(builder.item());
            builder = self.name.add_to_builder(builder.item());
            builder = self.result_json.add_to_builder(builder.item());
            builder = self.execution_time_ms.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("id").string();
            builder = builder.field("name").string();
            builder = builder.field("result-json").string();
            builder = TypeNodeBuilder::finish(builder.field("execution-time-ms").option().u32());
            builder.finish()
        }
    }

    // record tool-failure {
    //   id: string,
    //   name: string,
    //   error-message: string,
    //   error-code: option<string>,
    // }
    impl IntoValue for ToolFailure {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.id.add_to_builder(builder.item());
            builder = self.name.add_to_builder(builder.item());
            builder = self.error_message.add_to_builder(builder.item());
            builder = self.error_code.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("id").string();
            builder = builder.field("name").string();
            builder = builder.field("error-message").string();
            builder = TypeNodeBuilder::finish(builder.field("error-code").option().string());
            builder.finish()
        }
    }

    //   record response-metadata {
    //     finish-reason: option<finish-reason>,
    //     usage: option<usage>,
    //     provider-id: option<string>,
    //     timestamp: option<string>,
    //     provider-metadata-json: option<string>,
    //   }
    impl IntoValue for ResponseMetadata {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.finish_reason.add_to_builder(builder.item());
            builder = self.usage.add_to_builder(builder.item());
            builder = self.provider_id.add_to_builder(builder.item());
            builder = self.timestamp.add_to_builder(builder.item());
            builder = self.provider_metadata_json.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = Option::<FinishReason>::add_to_type_builder(builder.field("finish-reason"));
            builder = Option::<Usage>::add_to_type_builder(builder.field("usage"));
            builder = TypeNodeBuilder::finish(builder.field("provider-id").option().string());
            builder = TypeNodeBuilder::finish(builder.field("timestamp").option().string());
            builder =
                TypeNodeBuilder::finish(builder.field("provider-metadata-json").option().string());
            builder.finish()
        }
    }

    impl FromValueAndType for ResponseMetadata {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                finish_reason: Option::<FinishReason>::from_extractor(
                    &extractor
                        .field(0)
                        .ok_or_else(|| "Missing finish-reason field".to_string())?,
                )?,
                usage: Option::<Usage>::from_extractor(
                    &extractor
                        .field(1)
                        .ok_or_else(|| "Missing usage field".to_string())?,
                )?,
                provider_id: Option::<String>::from_extractor(
                    &extractor
                        .field(2)
                        .ok_or_else(|| "Missing provider-id field".to_string())?,
                )?,
                timestamp: Option::<String>::from_extractor(
                    &extractor
                        .field(3)
                        .ok_or_else(|| "Missing timestamp field".to_string())?,
                )?,
                provider_metadata_json: Option::<String>::from_extractor(
                    &extractor
                        .field(4)
                        .ok_or_else(|| "Missing provider-metadata-json field".to_string())?,
                )?,
            })
        }
    }

    // record usage {
    //     input-tokens: option<u32>,
    //     output-tokens: option<u32>,
    //     total-tokens: option<u32>,
    //   }
    impl IntoValue for Usage {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder
                .item()
                .option_fn(self.input_tokens.is_some(), |inner| {
                    inner.u32(self.input_tokens.unwrap())
                });
            builder = builder
                .item()
                .option_fn(self.output_tokens.is_some(), |inner| {
                    inner.u32(self.output_tokens.unwrap())
                });
            builder = builder
                .item()
                .option_fn(self.total_tokens.is_some(), |inner| {
                    inner.u32(self.total_tokens.unwrap())
                });
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = TypeNodeBuilder::finish(builder.field("input-tokens").option().u32());
            builder = TypeNodeBuilder::finish(builder.field("output-tokens").option().u32());
            builder = TypeNodeBuilder::finish(builder.field("total-tokens").option().u32());
            builder.finish()
        }
    }

    impl FromValueAndType for Usage {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                input_tokens: extractor
                    .field(0)
                    .ok_or_else(|| "Missing input-tokens field".to_string())?
                    .option()
                    .ok_or_else(|| "input-tokens is not an option".to_string())?
                    .map(|inner| {
                        inner
                            .u32()
                            .ok_or_else(|| "input-tokens is not u32".to_string())
                    })
                    .transpose()?,
                output_tokens: extractor
                    .field(1)
                    .ok_or_else(|| "Missing output-tokens field".to_string())?
                    .option()
                    .ok_or_else(|| "output-tokens is not an option".to_string())?
                    .map(|inner| {
                        inner
                            .u32()
                            .ok_or_else(|| "output-tokens is not u32".to_string())
                    })
                    .transpose()?,
                total_tokens: extractor
                    .field(2)
                    .ok_or_else(|| "Missing total-tokens field".to_string())?
                    .option()
                    .ok_or_else(|| "total-tokens is not an option".to_string())?
                    .map(|inner| {
                        inner
                            .u32()
                            .ok_or_else(|| "total-tokens is not u32".to_string())
                    })
                    .transpose()?,
            })
        }
    }

    // enum finish-reason {
    //     stop,
    //     length,
    //     tool-calls,
    //     content-filter,
    //     error,
    //     other,
    //   }
    impl IntoValue for FinishReason {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                FinishReason::Stop => builder.enum_value(0),
                FinishReason::Length => builder.enum_value(1),
                FinishReason::ToolCalls => builder.enum_value(2),
                FinishReason::ContentFilter => builder.enum_value(3),
                FinishReason::Error => builder.enum_value(4),
                FinishReason::Other => builder.enum_value(5),
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            builder.r#enum(&[
                "stop",
                "length",
                "tool-calls",
                "content-filter",
                "error",
                "other",
            ])
        }
    }

    impl FromValueAndType for FinishReason {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            match extractor.enum_value() {
                Some(0) => Ok(FinishReason::Stop),
                Some(1) => Ok(FinishReason::Length),
                Some(2) => Ok(FinishReason::ToolCalls),
                Some(3) => Ok(FinishReason::ContentFilter),
                Some(4) => Ok(FinishReason::Error),
                Some(5) => Ok(FinishReason::Other),
                _ => Err("Invalid finish reason".to_string()),
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct SendInput {
        messages: Vec<Message>,
        config: Config,
    }

    impl IntoValue for SendInput {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.messages.add_to_builder(builder.item());
            builder = self.config.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = Vec::<Message>::add_to_type_builder(builder.field("messages"));
            builder = Config::add_to_type_builder(builder.field("config"));
            builder.finish()
        }
    }

    #[derive(Debug)]
    struct ContinueInput {
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    }

    impl IntoValue for ContinueInput {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.messages.add_to_builder(builder.item());
            builder = self.tool_results.add_to_builder(builder.item());
            builder = self.config.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = Vec::<Message>::add_to_type_builder(builder.field("messages"));
            builder =
                Vec::<(ToolCall, ToolResult)>::add_to_type_builder(builder.field("tool-results"));
            builder = Config::add_to_type_builder(builder.field("config"));
            builder.finish()
        }
    }

    //   record message {
    //     role: role,
    //     name: option<string>,
    //     content: list<content-part>,
    //   }
    impl IntoValue for Message {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.role.add_to_builder(builder.item());
            builder = self.name.add_to_builder(builder.item());
            builder = self.content.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = Role::add_to_type_builder(builder.field("role"));
            builder = TypeNodeBuilder::finish(builder.field("name").option().string());
            builder = Vec::<ContentPart>::add_to_type_builder(builder.field("content"));
            builder.finish()
        }
    }

    //   enum role {
    //     user,
    //     assistant,
    //     system,
    //     tool,
    //   }
    impl IntoValue for Role {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                Role::User => builder.enum_value(0),
                Role::Assistant => builder.enum_value(1),
                Role::System => builder.enum_value(2),
                Role::Tool => builder.enum_value(3),
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            builder.r#enum(&["user", "assistant", "system", "tool"])
        }
    }

    // variant content-part {
    //     text(string),
    //     image(image-reference),
    //   }
    impl IntoValue for ContentPart {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                ContentPart::Text(text) => builder.variant(0).string(&text).finish(),
                ContentPart::Image(image_reference) => {
                    image_reference.add_to_builder(builder.variant(1)).finish()
                }
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.variant();
            builder = builder.case("text").string();
            builder = ImageReference::add_to_type_builder(builder.case("image"));
            builder.finish()
        }
    }

    impl FromValueAndType for ContentPart {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            let (idx, inner) = extractor
                .variant()
                .ok_or_else(|| "ContentPart should be variant".to_string())?;
            match idx {
                0 => Ok(ContentPart::Text(
                    inner
                        .ok_or_else(|| "Missing text".to_string())?
                        .string()
                        .ok_or_else(|| "ContentPart::Text should be string".to_string())?
                        .to_string(),
                )),
                1 => Ok(ContentPart::Image(ImageReference::from_extractor(
                    &inner.ok_or_else(|| "Missing image reference".to_string())?,
                )?)),
                _ => Err(format!("Invalid ContentPart variant: {idx}")),
            }
        }
    }

    // variant image-reference {
    //   url(image-url),
    //   inline(image-source),
    // }
    impl IntoValue for ImageReference {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                ImageReference::Url(image_url) => {
                    image_url.add_to_builder(builder.variant(0)).finish()
                }
                ImageReference::Inline(image_source) => {
                    image_source.add_to_builder(builder.variant(1)).finish()
                }
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.variant();
            builder = ImageUrl::add_to_type_builder(builder.case("url"));
            builder = ImageSource::add_to_type_builder(builder.case("inline"));
            builder.finish()
        }
    }

    impl FromValueAndType for ImageReference {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            let (idx, inner) = extractor
                .variant()
                .ok_or_else(|| "ImageReference should be variant".to_string())?;
            match idx {
                0 => Ok(ImageReference::Url(ImageUrl::from_extractor(
                    &inner.ok_or_else(|| "Missing image url".to_string())?,
                )?)),
                1 => Ok(ImageReference::Inline(ImageSource::from_extractor(
                    &inner.ok_or_else(|| "Missing image source".to_string())?,
                )?)),
                _ => Err(format!("Invalid ImageReference variant: {idx}")),
            }
        }
    }

    // record image-url {
    //     url: string,
    //     detail: option<image-detail>,
    //   }
    impl IntoValue for ImageUrl {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.url.add_to_builder(builder.item());
            builder = self.detail.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("url").string();
            builder = Option::<ImageDetail>::add_to_type_builder(builder.field("detail"));
            builder.finish()
        }
    }

    impl FromValueAndType for ImageUrl {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                url: String::from_extractor(
                    &extractor
                        .field(0)
                        .ok_or_else(|| "Missing url field".to_string())?,
                )?,
                detail: Option::<ImageDetail>::from_extractor(
                    &extractor
                        .field(1)
                        .ok_or_else(|| "Missing detail field".to_string())?,
                )?,
            })
        }
    }

    //   enum image-detail {
    //     low,
    //     high,
    //     auto,
    //   }
    impl IntoValue for ImageDetail {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                ImageDetail::Low => builder.enum_value(0),
                ImageDetail::High => builder.enum_value(1),
                ImageDetail::Auto => builder.enum_value(2),
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            builder.r#enum(&["low", "high", "auto"])
        }
    }

    impl FromValueAndType for ImageDetail {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            match extractor.enum_value() {
                Some(0) => Ok(ImageDetail::Low),
                Some(1) => Ok(ImageDetail::High),
                Some(2) => Ok(ImageDetail::Auto),
                _ => Err("Invalid image detail".to_string()),
            }
        }
    }

    // record image-source {
    //   data: list<u8>,
    //   mime-type: string,
    //   detail: option<image-detail>,
    // }
    impl IntoValue for ImageSource {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.data.add_to_builder(builder.item());
            builder = self.mime_type.add_to_builder(builder.item());
            builder = self.detail.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = Vec::<u8>::add_to_type_builder(builder.field("data"));
            builder = builder.field("mime-type").string();
            builder = Option::<ImageDetail>::add_to_type_builder(builder.field("detail"));
            builder.finish()
        }
    }

    impl FromValueAndType for ImageSource {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                data: Vec::<u8>::from_extractor(
                    &extractor
                        .field(0)
                        .ok_or_else(|| "Missing data field".to_string())?,
                )?,
                mime_type: String::from_extractor(
                    &extractor
                        .field(1)
                        .ok_or_else(|| "Missing mime-type field".to_string())?,
                )?,
                detail: Option::<ImageDetail>::from_extractor(
                    &extractor
                        .field(2)
                        .ok_or_else(|| "Missing detail field".to_string())?,
                )?,
            })
        }
    }

    //   record config {
    //     model: string,
    //     temperature: option<f32>,
    //     max-tokens: option<u32>,
    //     stop-sequences: option<list<string>>,
    //     tools: list<tool-definition>,
    //     tool-choice: option<string>,
    //     provider-options: list<kv>,
    //   }
    impl IntoValue for Config {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.model.add_to_builder(builder.item());
            builder = self.temperature.add_to_builder(builder.item());
            builder = self.max_tokens.add_to_builder(builder.item());
            builder = self.stop_sequences.add_to_builder(builder.item());
            builder = self.tools.add_to_builder(builder.item());
            builder = self.tool_choice.add_to_builder(builder.item());
            builder = self.provider_options.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("model").string();
            builder = TypeNodeBuilder::finish(builder.field("temperature").option().f32());
            builder = TypeNodeBuilder::finish(builder.field("max-tokens").option().u32());
            builder = TypeNodeBuilder::finish(
                builder
                    .field("stop-sequences")
                    .option()
                    .list()
                    .string()
                    .finish(),
            );
            builder = Vec::<ToolDefinition>::add_to_type_builder(builder.field("tools"));
            builder = TypeNodeBuilder::finish(builder.field("tool-choice").option().string());
            builder = Vec::<Kv>::add_to_type_builder(builder.field("provider-options"));
            builder.finish()
        }
    }

    //   record tool-definition {
    //     name: string,
    //     description: option<string>,
    //     parameters-schema: string,
    //   }
    impl IntoValue for ToolDefinition {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.name.add_to_builder(builder.item());
            builder = self.description.add_to_builder(builder.item());
            builder = self.parameters_schema.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("name").string();
            builder = TypeNodeBuilder::finish(builder.field("description").option().string());
            builder = builder.field("parameters-schema").string();
            builder.finish()
        }
    }

    //   record kv {
    //     key: string,
    //     value: string,
    //   }
    impl IntoValue for Kv {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.key.add_to_builder(builder.item());
            builder = self.value.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("key").string();
            builder = builder.field("value").string();
            builder.finish()
        }
    }

    //   enum error-code {
    //     invalid-request,
    //     authentication-failed,
    //     rate-limit-exceeded,
    //     internal-error,
    //     unsupported,
    //     unknown,
    //   }
    impl IntoValue for ErrorCode {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                ErrorCode::InvalidRequest => builder.enum_value(0),
                ErrorCode::AuthenticationFailed => builder.enum_value(1),
                ErrorCode::RateLimitExceeded => builder.enum_value(2),
                ErrorCode::InternalError => builder.enum_value(3),
                ErrorCode::Unsupported => builder.enum_value(4),
                ErrorCode::Unknown => builder.enum_value(5),
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            builder.r#enum(&[
                "invalid-request",
                "authentication-failed",
                "rate-limit-exceeded",
                "internal-error",
                "unsupported",
                "unknown",
            ])
        }
    }

    impl FromValueAndType for ErrorCode {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            match extractor.enum_value() {
                Some(0) => Ok(ErrorCode::InvalidRequest),
                Some(1) => Ok(ErrorCode::AuthenticationFailed),
                Some(2) => Ok(ErrorCode::RateLimitExceeded),
                Some(3) => Ok(ErrorCode::InternalError),
                Some(4) => Ok(ErrorCode::Unsupported),
                Some(5) => Ok(ErrorCode::Unknown),
                _ => Err("Invalid error code".to_string()),
            }
        }
    }

    //   record error {
    //     code: error-code,
    //     message: string,
    //     provider-error-json: option<string>,
    //   }
    impl IntoValue for Error {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.code.add_to_builder(builder.item());
            builder = self.message.add_to_builder(builder.item());
            builder = self.provider_error_json.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = ErrorCode::add_to_type_builder(builder.field("code"));
            builder = builder.field("message").string();
            builder =
                TypeNodeBuilder::finish(builder.field("provider-error-json").option().string());
            builder.finish()
        }
    }

    impl FromValueAndType for Error {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                code: ErrorCode::from_extractor(
                    &extractor
                        .field(0)
                        .ok_or_else(|| "Missing code field".to_string())?,
                )?,
                message: String::from_extractor(
                    &extractor
                        .field(1)
                        .ok_or_else(|| "Missing message field".to_string())?,
                )?,
                provider_error_json: Option::<String>::from_extractor(
                    &extractor
                        .field(2)
                        .ok_or_else(|| "Missing provider-error-json field".to_string())?,
                )?,
            })
        }
    }

    #[derive(Debug)]
    struct NoInput;

    impl IntoValue for NoInput {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            builder.variant_unit(0)
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            builder.variant().unit_case("no-input").finish()
        }
    }

    #[derive(Debug, Clone)]
    struct NoOutput;

    impl IntoValue for NoOutput {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            builder.variant_unit(0)
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            builder.variant().unit_case("no-output").finish()
        }
    }

    impl FromValueAndType for NoOutput {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            let (idx, _inner) = extractor
                .variant()
                .ok_or_else(|| "NoOutput should be variant".to_string())?;
            if idx == 0 {
                Ok(NoOutput)
            } else {
                Err(format!("NoOutput should be variant 0, but got {idx}"))
            }
        }
    }

    #[derive(Debug)]
    struct UnusedError;

    impl Display for UnusedError {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "UnusedError")
        }
    }

    impl IntoValue for UnusedError {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            builder.variant_unit(0)
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            builder.variant().unit_case("unused-error").finish()
        }
    }

    impl FromValueAndType for UnusedError {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            let (idx, _inner) = extractor
                .variant()
                .ok_or_else(|| "UnusedError should be variant".to_string())?;
            if idx == 0 {
                Ok(UnusedError)
            } else {
                Err(format!("UnusedError should be variant 0, but got {idx}"))
            }
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
