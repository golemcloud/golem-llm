use crate::exports::golem::web_search::web_search::{Guest, GuestSearchSession, SearchSession};
use crate::golem::web_search::types::{SearchError, SearchMetadata, SearchParams, SearchResult};
use golem_rust::bindings::golem::durability::durability::DurableFunctionType;
use golem_rust::durability::Durability;
use golem_rust::{with_persistence_level, PersistenceLevel};
use std::marker::PhantomData;

pub struct DurableWebSearch<Impl> {
    phantom: PhantomData<Impl>,
}

pub trait ExtendedGuest: Guest + 'static {
    type ExtendedSearchSession: ExtendedSearchSession;
    fn unwrapped_search_session(params: SearchParams) -> Self::ExtendedSearchSession;
}

pub trait ExtendedSearchSession: GuestSearchSession + 'static {
    fn new_durable(params: SearchParams) -> Self;
    fn next_page_durable(&self) -> Result<Vec<SearchResult>, SearchError>;
    fn get_metadata_durable(&self) -> Option<SearchMetadata>;
}

#[cfg(not(feature = "durability"))]
mod passthrough_impl {
    use super::*;

    impl<Impl: ExtendedGuest> Guest for DurableWebSearch<Impl> {
        type SearchSession = Impl::ExtendedSearchSession;

        fn search_once(
            params: SearchParams,
        ) -> Result<(Vec<SearchResult>, Option<SearchMetadata>), SearchError> {
            Impl::search_once(params)
        }

        fn start_search(params: SearchParams) -> Result<Impl::ExtendedSearchSession, SearchError> {
            Ok(Impl::unwrapped_search_session(params))
        }
    }
}

#[cfg(feature = "durability")]
mod durable_impl {
    use super::*;
    use golem_rust::{FromValueAndType, IntoValue};
    use std::fmt::{Display, Formatter};

    impl<Impl: ExtendedGuest> Guest for DurableWebSearch<Impl> {
        type SearchSession = DurableSearchSession<<Impl as ExtendedGuest>::ExtendedSearchSession>;

        fn search_once(
            params: SearchParams,
        ) -> Result<(Vec<SearchResult>, Option<SearchMetadata>), SearchError> {
            let durability = Durability::<SearchOnceResult, UnusedError>::new(
                "golem_web_search",
                "search_once",
                DurableFunctionType::WriteRemote,
            );

            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    Impl::search_once(params.clone())
                });
                durability.persist_infallible(SearchOnceInput { params }, result)
            } else {
                durability.replay_infallible()
            }
        }

        fn start_search(params: SearchParams) -> Result<SearchSession, SearchError> {
            Ok(SearchSession::new(DurableSearchSession::new(
                Impl::unwrapped_search_session(params),
            )))
        }
    }

    pub struct DurableSearchSession<Impl: ExtendedSearchSession> {
        inner: Impl,
    }

    impl<Impl: ExtendedSearchSession> DurableSearchSession<Impl> {
        pub fn new(inner: Impl) -> Self {
            Self { inner }
        }
    }

    impl<Impl: ExtendedSearchSession> GuestSearchSession for DurableSearchSession<Impl> {
        fn new(params: SearchParams) -> Self {
            Self::new(Impl::new_durable(params))
        }

        fn next_page(&self) -> Result<Vec<SearchResult>, SearchError> {
            let durability = Durability::<NextPageResult, UnusedError>::new(
                "golem_web_search",
                "next_page",
                DurableFunctionType::WriteRemote,
            );

            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.next_page_durable()
                });
                durability.persist_infallible(NoInput, result)
            } else {
                durability.replay_infallible()
            }
        }

        fn get_metadata(&self) -> Option<SearchMetadata> {
            let durability = Durability::<GetMetadataResult, UnusedError>::new(
                "golem_web_search",
                "get_metadata",
                DurableFunctionType::ReadRemote,
            );

            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.get_metadata_durable()
                });
                durability.persist_infallible(NoInput, result)
            } else {
                durability.replay_infallible()
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, IntoValue)]
    struct SearchOnceInput {
        params: SearchParams,
    }

    #[derive(Debug, IntoValue)]
    struct NoInput;

    type SearchOnceResult = Result<(Vec<SearchResult>, Option<SearchMetadata>), SearchError>;
    type NextPageResult = Result<Vec<SearchResult>, SearchError>;
    type GetMetadataResult = Option<SearchMetadata>;

    #[derive(Debug, FromValueAndType, IntoValue)]
    struct UnusedError;

    impl Display for UnusedError {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "UnusedError")
        }
    }
}
