use crate::golem::embed::embed::Guest;
use std::marker::PhantomData;

/// Wraps an embed implementation with custom durability
pub struct DurableEmbed<Impl> {
    phantom: PhantomData<Impl>,
}

/// Trait to be implemented in addition to the embed `Guest` trait when wrapping it with `DurableEmbed`.
pub trait ExtendedGuest: Guest + 'static {}

/// When the durability feature flag is off, wrapping with `DurableEmbed` is just a passthrough
#[cfg(not(feature = "durability"))]
mod passthrough_impl {
    use crate::durability::{DurableEmbed, ExtendedGuest};
    use crate::golem::embed::embed::{
        Config, ContentPart, EmbeddingResponse, Error, Guest, RerankResponse,
    };

    impl<Impl: ExtendedGuest> Guest for DurableEmbed<Impl> {
        fn generate(inputs: Vec<ContentPart>, config: Config) -> Result<EmbeddingResponse, Error> {
            Impl::generate(inputs, config)
        }

        fn rerank(
            query: String,
            documents: Vec<String>,
            config: Config,
        ) -> Result<RerankResponse, Error> {
            Impl::rerank(query, documents, config)
        }
    }
}

/// When the durability feature flag is on, wrapping with `DurableEmbed` adds custom durability
/// on top of the provider-specific embed implementation using Golem's special host functions and
/// the `golem-rust` helper library.
///
/// There will be custom durability entries saved in the oplog, with the full embed request and configuration
/// stored as input, and the full response stored as output. To serialize these in a way it is
/// observable by oplog consumers, each relevant data type has to be converted to/from `ValueAndType`
/// which is implemented using the type classes and builder in the `golem-rust` library.
#[cfg(feature = "durability")]
mod durable_impl {
    use crate::durability::{DurableEmbed, ExtendedGuest};
    use crate::golem::embed::embed::{
        Config, ContentPart, EmbeddingResponse, Error, Guest, RerankResponse,
    };
    use golem_rust::bindings::golem::durability::durability::DurableFunctionType;
    use golem_rust::durability::Durability;
    use golem_rust::{with_persistence_level, FromValueAndType, IntoValue, PersistenceLevel};
    use std::fmt::{Display, Formatter};

    impl<Impl: ExtendedGuest> Guest for DurableEmbed<Impl> {
        fn generate(inputs: Vec<ContentPart>, config: Config) -> Result<EmbeddingResponse, Error> {
            let durability = Durability::<Result<EmbeddingResponse, Error>, UnusedError>::new(
                "golem_embed",
                "generate",
                DurableFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    Impl::generate(inputs.clone(), config.clone())
                });
                durability.persist_infallible(GenerateInput { inputs, config }, result.clone())
            } else {
                durability.replay_infallible()
            }
        }

        fn rerank(
            query: String,
            documents: Vec<String>,
            config: Config,
        ) -> Result<RerankResponse, Error> {
            let durability = Durability::<Result<RerankResponse, Error>, UnusedError>::new(
                "golem_embed",
                "rerank",
                DurableFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    Impl::rerank(query.clone(), documents.clone(), config.clone())
                });
                durability.persist_infallible(
                    RerankInput {
                        query,
                        documents,
                        config,
                    },
                    result.clone(),
                )
            } else {
                durability.replay_infallible()
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, IntoValue, FromValueAndType)]
    struct GenerateInput {
        inputs: Vec<ContentPart>,
        config: Config,
    }

    #[derive(Debug, Clone, PartialEq, IntoValue, FromValueAndType)]
    struct RerankInput {
        query: String,
        documents: Vec<String>,
        config: Config,
    }

    #[derive(Debug, FromValueAndType, IntoValue)]
    struct UnusedError;

    impl Display for UnusedError {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "UnusedError")
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::durability::durable_impl::{GenerateInput, RerankInput};
        use crate::golem::embed::embed::{Config, ContentPart, ImageUrl, TaskType};
        use golem_rust::value_and_type::{FromValueAndType, IntoValueAndType};
        use std::fmt::Debug;

        fn roundtrip_test<T: Debug + Clone + PartialEq + IntoValueAndType + FromValueAndType>(
            value: T,
        ) {
            let vnt = value.clone().into_value_and_type();
            let extracted = T::from_value_and_type(vnt).unwrap();
            assert_eq!(value, extracted);
        }

        #[test]
        fn generate_input_encoding() {
            let input = GenerateInput {
                inputs: vec![
                    ContentPart::Text("Hello world".to_string()),
                    ContentPart::Image(ImageUrl {
                        url: "https://example.com/image.png".to_string(),
                    }),
                ],
                config: Config {
                    model: Some("text-embedding-3-small".to_string()),
                    task_type: Some(TaskType::RetrievalQuery),
                    dimensions: Some(512),
                    truncation: Some(true),
                    output_format: None,
                    output_dtype: None,
                    user: Some("test-user".to_string()),
                    provider_options: vec![],
                },
            };

            roundtrip_test(input);
        }

        #[test]
        fn rerank_input_encoding() {
            let input = RerankInput {
                query: "What is machine learning?".to_string(),
                documents: vec![
                    "Machine learning is a subset of AI".to_string(),
                    "Deep learning uses neural networks".to_string(),
                    "NLP processes human language".to_string(),
                ],
                config: Config {
                    model: Some("rerank-english-v3.0".to_string()),
                    task_type: None,
                    dimensions: None,
                    truncation: None,
                    output_format: None,
                    output_dtype: None,
                    user: None,
                    provider_options: vec![],
                },
            };

            roundtrip_test(input);
        }
    }
}
