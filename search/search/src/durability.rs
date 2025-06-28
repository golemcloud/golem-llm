#[cfg(feature = "durability")]
use golem_rust::*;

#[cfg(feature = "durability")]
use crate::golem::search::core::Guest;
use crate::golem::search::types::{Doc, Schema, SearchError, SearchQuery, SearchResults};

pub trait ExtendedGuest: Guest {
    type SearchHitStream;
    fn unwrapped_stream_search(
        index: String,
        query: SearchQuery,
    ) -> Result<Self::SearchHitStream, SearchError>;
}

#[cfg(feature = "durability")]
pub struct DurableSearch<T: ExtendedGuest> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "durability")]
impl<T: ExtendedGuest> Guest for DurableSearch<T> {
    type SearchHitStream = T::SearchHitStream;

    fn create_index(name: String, schema: Option<Schema>) -> Result<(), SearchError> {
        durable_host_function!(
            "search-create-index",
            |name: String, schema: Option<Schema>| -> Result<(), SearchError> {
                T::create_index(name, schema)
            },
            name,
            schema
        )
    }

    fn delete_index(name: String) -> Result<(), SearchError> {
        durable_host_function!(
            "search-delete-index",
            |name: String| -> Result<(), SearchError> { T::delete_index(name) },
            name
        )
    }

    fn list_indexes() -> Result<Vec<String>, SearchError> {
        durable_host_function!(
            "search-list-indexes",
            || -> Result<Vec<String>, SearchError> { T::list_indexes() }
        )
    }

    fn upsert(index: String, doc: Doc) -> Result<(), SearchError> {
        durable_host_function!(
            "search-upsert",
            |index: String, doc: Doc| -> Result<(), SearchError> { T::upsert(index, doc) },
            index,
            doc
        )
    }

    fn upsert_many(index: String, docs: Vec<Doc>) -> Result<(), SearchError> {
        durable_host_function!(
            "search-upsert-many",
            |index: String, docs: Vec<Doc>| -> Result<(), SearchError> {
                T::upsert_many(index, docs)
            },
            index,
            docs
        )
    }

    fn delete(index: String, id: String) -> Result<(), SearchError> {
        durable_host_function!(
            "search-delete",
            |index: String, id: String| -> Result<(), SearchError> { T::delete(index, id) },
            index,
            id
        )
    }

    fn delete_many(index: String, ids: Vec<String>) -> Result<(), SearchError> {
        durable_host_function!(
            "search-delete-many",
            |index: String, ids: Vec<String>| -> Result<(), SearchError> {
                T::delete_many(index, ids)
            },
            index,
            ids
        )
    }

    fn get(index: String, id: String) -> Result<Option<Doc>, SearchError> {
        durable_host_function!(
            "search-get",
            |index: String, id: String| -> Result<Option<Doc>, SearchError> { T::get(index, id) },
            index,
            id
        )
    }

    fn search(index: String, query: SearchQuery) -> Result<SearchResults, SearchError> {
        durable_host_function!(
            "search-query",
            |index: String, query: SearchQuery| -> Result<SearchResults, SearchError> {
                T::search(index, query)
            },
            index,
            query
        )
    }

    fn stream_search(
        index: String,
        query: SearchQuery,
    ) -> Result<Self::SearchHitStream, SearchError> {
        T::unwrapped_stream_search(index, query)
    }

    fn get_schema(index: String) -> Result<Schema, SearchError> {
        durable_host_function!(
            "search-get-schema",
            |index: String| -> Result<Schema, SearchError> { T::get_schema(index) },
            index
        )
    }

    fn update_schema(index: String, schema: Schema) -> Result<(), SearchError> {
        durable_host_function!(
            "search-update-schema",
            |index: String, schema: Schema| -> Result<(), SearchError> {
                T::update_schema(index, schema)
            },
            index,
            schema
        )
    }
}

#[cfg(not(feature = "durability"))]
pub struct DurableSearch<T: ExtendedGuest> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(not(feature = "durability"))]
impl<T: ExtendedGuest> Guest for DurableSearch<T> {
    type SearchHitStream = T::SearchHitStream;

    fn create_index(name: String, schema: Option<Schema>) -> Result<(), SearchError> {
        T::create_index(name, schema)
    }

    fn delete_index(name: String) -> Result<(), SearchError> {
        T::delete_index(name)
    }

    fn list_indexes() -> Result<Vec<String>, SearchError> {
        T::list_indexes()
    }

    fn upsert(index: String, doc: Doc) -> Result<(), SearchError> {
        T::upsert(index, doc)
    }

    fn upsert_many(index: String, docs: Vec<Doc>) -> Result<(), SearchError> {
        T::upsert_many(index, docs)
    }

    fn delete(index: String, id: String) -> Result<(), SearchError> {
        T::delete(index, id)
    }

    fn delete_many(index: String, ids: Vec<String>) -> Result<(), SearchError> {
        T::delete_many(index, ids)
    }

    fn get(index: String, id: String) -> Result<Option<Doc>, SearchError> {
        T::get(index, id)
    }

    fn search(index: String, query: SearchQuery) -> Result<SearchResults, SearchError> {
        T::search(index, query)
    }

    fn stream_search(
        index: String,
        query: SearchQuery,
    ) -> Result<Self::SearchHitStream, SearchError> {
        T::unwrapped_stream_search(index, query)
    }

    fn get_schema(index: String) -> Result<Schema, SearchError> {
        T::get_schema(index)
    }

    fn update_schema(index: String, schema: Schema) -> Result<(), SearchError> {
        T::update_schema(index, schema)
    }
}
