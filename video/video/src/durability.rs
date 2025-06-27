#[allow(unused_imports)]
use crate::exports::golem::video::video::{
    GenerationConfig, Guest, MediaInput, VideoError, VideoResult,
};
use std::marker::PhantomData;

/// Wraps a Video implementation with custom durability
pub struct DurableVideo<Impl> {
    phantom: PhantomData<Impl>,
}

/// Trait to be implemented in addition to the Video `Guest` trait when wrapping it with `DurableVideo`.
pub trait ExtendedGuest: Guest + 'static {}

/// When the durability feature flag is off, wrapping with `DurableVideo` is just a passthrough
#[cfg(not(feature = "durability"))]
mod passthrough_impl {
    use crate::durability::{DurableVideo, ExtendedGuest};
    use crate::exports::golem::video::video::{
        GenerationConfig, Guest, MediaInput, VideoError, VideoResult,
    };

    impl<Impl: ExtendedGuest> Guest for DurableVideo<Impl> {
        fn generate(input: MediaInput, config: GenerationConfig) -> Result<String, VideoError> {
            Impl::generate(input, config)
        }

        fn poll(job_id: String) -> Result<VideoResult, VideoError> {
            Impl::poll(job_id)
        }

        fn cancel(job_id: String) -> Result<String, VideoError> {
            Impl::cancel(job_id)
        }
    }
}

/// When the durability feature flag is on, wrapping with `DurableVideo` adds custom durability
/// on top of the provider-specific Video implementation using Golem's special host functions and
/// the `golem-rust` helper library.
///
/// There will be custom durability entries saved in the oplog, with the full Video request and configuration
/// stored as input, and the full response stored as output. To serialize these in a way it is
/// observable by oplog consumers, each relevant data type has to be converted to/from `ValueAndType`
/// which is implemented using the type classes and builder in the `golem-rust` library.
#[cfg(feature = "durability")]
mod durable_impl {
    use crate::durability::{DurableVideo, ExtendedGuest};
    use crate::exports::golem::video::video::{
        GenerationConfig, Guest, MediaInput, VideoError, VideoResult,
    };
    use golem_rust::bindings::golem::durability::durability::DurableFunctionType;
    use golem_rust::durability::Durability;
    use golem_rust::{with_persistence_level, FromValueAndType, IntoValue, PersistenceLevel};
    use std::fmt::{Display, Formatter};

    impl<Impl: ExtendedGuest> Guest for DurableVideo<Impl> {
        fn generate(input: MediaInput, config: GenerationConfig) -> Result<String, VideoError> {
            let durability = Durability::<Result<String, VideoError>, UnusedError>::new(
                "golem_video",
                "generate",
                DurableFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    Impl::generate(input.clone(), config.clone())
                });
                durability.persist_infallible(GenerateInput { input, config }, result)
            } else {
                durability.replay_infallible()
            }
        }

        fn poll(job_id: String) -> Result<VideoResult, VideoError> {
            let durability = Durability::<Result<VideoResult, VideoError>, UnusedError>::new(
                "golem_video",
                "poll",
                DurableFunctionType::ReadRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    Impl::poll(job_id.clone())
                });
                durability.persist_infallible(PollInput { job_id }, result)
            } else {
                durability.replay_infallible()
            }
        }

        fn cancel(job_id: String) -> Result<String, VideoError> {
            let durability = Durability::<Result<String, VideoError>, UnusedError>::new(
                "golem_video",
                "cancel",
                DurableFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    Impl::cancel(job_id.clone())
                });
                durability.persist_infallible(CancelInput { job_id }, result)
            } else {
                durability.replay_infallible()
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, IntoValue, FromValueAndType)]
    struct GenerateInput {
        input: MediaInput,
        config: GenerationConfig,
    }

    #[derive(Debug, Clone, PartialEq, IntoValue, FromValueAndType)]
    struct PollInput {
        job_id: String,
    }

    #[derive(Debug, Clone, PartialEq, IntoValue, FromValueAndType)]
    struct CancelInput {
        job_id: String,
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
        use crate::durability::durable_impl::{CancelInput, GenerateInput, PollInput};
        use crate::exports::golem::video::video::{
            AspectRatio, GenerationConfig, Kv, MediaData, MediaInput, ReferenceImage, Resolution,
        };
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
        fn generate_input_roundtrip() {
            let input = GenerateInput {
                input: MediaInput::Text("Generate a video of a cat".to_string()),
                config: GenerationConfig {
                    negative_prompt: Some("blurry, low quality".to_string()),
                    seed: Some(12345),
                    scheduler: Some("ddim".to_string()),
                    guidance_scale: Some(7.5),
                    aspect_ratio: Some(AspectRatio::Landscape),
                    duration_seconds: Some(5.0),
                    resolution: Some(Resolution::Hd),
                    model: Some("runway-gen3".to_string()),
                    enable_audio: Some(true),
                    enhance_prompt: Some(false),
                    provider_options: vec![Kv {
                        key: "model".to_string(),
                        value: "runway-gen3".to_string(),
                    }],
                },
            };
            roundtrip_test(input);
        }

        #[test]
        fn generate_input_with_image_roundtrip() {
            let input = GenerateInput {
                input: MediaInput::Image(ReferenceImage {
                    data: MediaData::Bytes(vec![0, 1, 2, 3, 4, 5]),
                    prompt: Some("Animate this image".to_string()),
                }),
                config: GenerationConfig {
                    negative_prompt: None,
                    seed: None,
                    scheduler: None,
                    guidance_scale: None,
                    aspect_ratio: Some(AspectRatio::Square),
                    duration_seconds: Some(3.0),
                    resolution: Some(Resolution::Sd),
                    model: None,
                    enable_audio: Some(false),
                    enhance_prompt: None,
                    provider_options: vec![],
                },
            };
            roundtrip_test(input);
        }

        #[test]
        fn poll_input_roundtrip() {
            let input = PollInput {
                job_id: "job_12345".to_string(),
            };
            roundtrip_test(input);
        }

        #[test]
        fn cancel_input_roundtrip() {
            let input = CancelInput {
                job_id: "job_67890".to_string(),
            };
            roundtrip_test(input);
        }
    }
}
