use std::marker::PhantomData;

/// Wraps a web search implementation with custom durability
pub struct DurableWebSearch<Impl> {
    phantom: PhantomData<Impl>,
}

/// When the durability feature flag is off, wrapping with `DurableWebSearch` is just a passthrough
#[cfg(not(feature = "durability"))]
mod passthrough_impl {
    use crate::durability::DurableWebSearch;
    use crate::exports::golem::web_search::web_search::{
        Guest, SearchError, SearchMetadata, SearchParams, SearchResult, SearchSession,
    };

    impl<Impl: Guest> Guest for DurableWebSearch<Impl> {
        type SearchSession = Impl::SearchSession;

        fn start_search(params: SearchParams) -> Result<SearchSession, SearchError> {
            Impl::start_search(params)
        }

        fn search_once(
            params: SearchParams,
        ) -> Result<(Vec<SearchResult>, Option<SearchMetadata>), SearchError> {
            Impl::search_once(params)
        }
    }
}

/// When the durability feature flag is on, wrapping with `DurableWebSearch` adds custom durability
/// on top of the provider-specific web search implementation using Golem's special host functions and
/// the `golem-rust` helper library.
///
/// There will be custom durability entries saved in the oplog, with the full web search request and configuration
/// stored as input, and the full response stored as output. To serialize these in a way it is
/// observable by oplog consumers, each relevant data type has to be converted to/from `ValueAndType`
/// which is implemented using the type classes and builder in the `golem-rust` library.
#[cfg(feature = "durability")]
mod durable_impl {
    use crate::durability::DurableWebSearch;
    use crate::exports::golem::web_search::web_search::{
        Guest, SearchError, SearchMetadata, SearchParams, SearchResult, SearchSession,
    };
    use golem_rust::bindings::golem::durability::durability::DurableFunctionType;
    use golem_rust::durability::Durability;
    use golem_rust::{with_persistence_level, FromValueAndType, IntoValue, PersistenceLevel};
    use std::fmt::{Display, Formatter};

    impl<Impl: Guest> Guest for DurableWebSearch<Impl> {
        type SearchSession = Impl::SearchSession;

        fn start_search(params: SearchParams) -> Result<SearchSession, SearchError> {
            let durability = Durability::<NoOutput, UnusedError>::new(
                "golem_web_search",
                "start_search",
                DurableFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    Impl::start_search(params.clone())
                });
                match result {
                    Ok(session) => {
                        let _ = durability.persist_infallible(params, NoOutput);
                        Ok(session)
                    }
                    Err(err) => Err(err),
                }
            } else {
                let _: NoOutput = durability.replay_infallible();
                Impl::start_search(params)
            }
        }

        fn search_once(
            params: SearchParams,
        ) -> Result<(Vec<SearchResult>, Option<SearchMetadata>), SearchError> {
            let durability =
                Durability::<(Vec<SearchResult>, Option<SearchMetadata>), UnusedError>::new(
                    "golem_web_search",
                    "search_once",
                    DurableFunctionType::WriteRemote,
                );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    Impl::search_once(params.clone())
                });
                match result {
                    Ok(success) => Ok(durability.persist_infallible(params, success)),
                    Err(err) => Err(err),
                }
            } else {
                let result: (Vec<SearchResult>, Option<SearchMetadata>) =
                    durability.replay_infallible();
                Ok(result)
            }
        }
    }

    #[derive(Debug, Clone, IntoValue, FromValueAndType)]
    struct NoOutput;

    #[derive(Debug, IntoValue, FromValueAndType)]
    struct UnusedError;

    impl Display for UnusedError {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "UnusedError")
        }
    }
}
