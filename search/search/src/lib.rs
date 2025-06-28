use std::cell::RefCell;

pub mod config;
pub mod durability;
pub mod error;

wit_bindgen::generate!({
    path: "../wit",
    with: {}
});

use crate::golem::search::types::{SearchError, SearchHit};

thread_local! {
    pub static LOGGING_STATE: RefCell<LoggingState> = RefCell::new(LoggingState::new());
}

struct LoggingState {
    initialized: bool,
}

impl LoggingState {
    fn new() -> Self {
        Self { initialized: false }
    }

    pub fn init(&mut self) {
        if !self.initialized {
            wasi_logger::Logger::install().unwrap();
            log::set_max_level(
                std::env::var("GOLEM_SEARCH_LOG")
                    .unwrap_or_else(|_| "warn".to_string())
                    .parse()
                    .unwrap_or(log::LevelFilter::Warn),
            );
            self.initialized = true;
        }
    }
}

pub struct SearchStream<T> {
    inner: T,
}

impl<T> SearchStream<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

pub trait SearchStreamState {
    fn failure(&self) -> &Option<SearchError>;
    fn is_finished(&self) -> bool;
    fn set_finished(&self);
    fn get_next_hits(&self) -> Result<Vec<SearchHit>, SearchError>;
}

impl<T: SearchStreamState> SearchStream<T> {
    pub fn get_next(&self) -> Option<Vec<SearchHit>> {
        if self.inner.is_finished() {
            return None;
        }

        if let Some(error) = self.inner.failure() {
            self.inner.set_finished();
            return None;
        }

        match self.inner.get_next_hits() {
            Ok(hits) => {
                if hits.is_empty() {
                    self.inner.set_finished();
                    None
                } else {
                    Some(hits)
                }
            }
            Err(_) => {
                self.inner.set_finished();
                None
            }
        }
    }

    pub fn blocking_get_next(&self) -> Vec<SearchHit> {
        self.get_next().unwrap_or_default()
    }
}

#[macro_export]
macro_rules! export_search {
    ($component:ty with_types_in $types:path) => {
        use $types::exports::golem::search::core::Guest;

        struct Component;

        impl Guest for Component {
            type SearchHitStream = <$component as Guest>::SearchHitStream;

            fn create_index(
                name: String,
                schema: Option<$types::golem::search::types::Schema>,
            ) -> Result<(), $types::golem::search::types::SearchError> {
                <$component as Guest>::create_index(name, schema)
            }

            fn delete_index(
                name: String,
            ) -> Result<(), $types::golem::search::types::SearchError> {
                <$component as Guest>::delete_index(name)
            }

            fn list_indexes() -> Result<Vec<String>, $types::golem::search::types::SearchError> {
                <$component as Guest>::list_indexes()
            }

            fn upsert(
                index: String,
                doc: $types::golem::search::types::Doc,
            ) -> Result<(), $types::golem::search::types::SearchError> {
                <$component as Guest>::upsert(index, doc)
            }

            fn upsert_many(
                index: String,
                docs: Vec<$types::golem::search::types::Doc>,
            ) -> Result<(), $types::golem::search::types::SearchError> {
                <$component as Guest>::upsert_many(index, docs)
            }

            fn delete(
                index: String,
                id: String,
            ) -> Result<(), $types::golem::search::types::SearchError> {
                <$component as Guest>::delete(index, id)
            }

            fn delete_many(
                index: String,
                ids: Vec<String>,
            ) -> Result<(), $types::golem::search::types::SearchError> {
                <$component as Guest>::delete_many(index, ids)
            }

            fn get(
                index: String,
                id: String,
            ) -> Result<Option<$types::golem::search::types::Doc>, $types::golem::search::types::SearchError> {
                <$component as Guest>::get(index, id)
            }

            fn search(
                index: String,
                query: $types::golem::search::types::SearchQuery,
            ) -> Result<$types::golem::search::types::SearchResults, $types::golem::search::types::SearchError> {
                <$component as Guest>::search(index, query)
            }

            fn stream_search(
                index: String,
                query: $types::golem::search::types::SearchQuery,
            ) -> Result<Self::SearchHitStream, $types::golem::search::types::SearchError> {
                <$component as Guest>::stream_search(index, query)
            }

            fn get_schema(
                index: String,
            ) -> Result<$types::golem::search::types::Schema, $types::golem::search::types::SearchError> {
                <$component as Guest>::get_schema(index)
            }

            fn update_schema(
                index: String,
                schema: $types::golem::search::types::Schema,
            ) -> Result<(), $types::golem::search::types::SearchError> {
                <$component as Guest>::update_schema(index, schema)
            }
        }

        $types::export!(Component with_types_in $types);
    };
}
