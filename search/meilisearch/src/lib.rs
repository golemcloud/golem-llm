use crate::client::MeilisearchApi;
use crate::conversions::{
    doc_to_meilisearch_doc, meilisearch_hit_to_search_hit, meilisearch_response_to_results, 
    query_to_meilisearch_query, schema_to_meilisearch_settings, meilisearch_settings_to_schema,
};
use golem_search::config::with_config_key;
use golem_search::durability::{DurableSearch, ExtendedGuest};
use golem_search::golem::search::core::Guest;
use golem_search::golem::search::types::{
    Doc, Schema, SearchError, SearchHit, SearchQuery, SearchResults,
};
use golem_search::{SearchStream, SearchStreamState, LOGGING_STATE};
use log::trace;
use std::cell::RefCell;

mod client;
mod conversions;

struct MeilisearchStream {
    client: MeilisearchApi,
    index: String,
    query: SearchQuery,
    current_offset: RefCell<u32>,
    per_page: u32,
    finished: RefCell<bool>,
    failure: Option<SearchError>,
}

impl MeilisearchStream {
    fn new(client: MeilisearchApi, index: String, query: SearchQuery) -> Self {
        let per_page = query.per_page.unwrap_or(20);
        let initial_offset = query.offset.unwrap_or(0);
        
        Self {
            client,
            index,
            query,
            current_offset: RefCell::new(initial_offset),
            per_page,
            finished: RefCell::new(false),
            failure: None,
        }
    }

    fn failed(error: SearchError) -> Self {
        Self {
            client: MeilisearchApi::empty(),
            index: String::new(),
            query: SearchQuery {
                q: None,
                filters: vec![],
                sort: vec![],
                facets: vec![],
                page: None,
                per_page: None,
                offset: None,
                highlight: None,
                config: None,
            },
            current_offset: RefCell::new(0),
            per_page: 20,
            finished: RefCell::new(true),
            failure: Some(error),
        }
    }
}

impl SearchStreamState for MeilisearchStream {
    fn failure(&self) -> &Option<SearchError> {
        &self.failure
    }

    fn is_finished(&self) -> bool {
        *self.finished.borrow()
    }

    fn set_finished(&self) {
        *self.finished.borrow_mut() = true;
    }

    fn get_next_hits(&self) -> Result<Vec<SearchHit>, SearchError> {
        let current_offset = *self.current_offset.borrow();
        
        // Create a modified query with current pagination
        let mut paginated_query = self.query.clone();
        paginated_query.offset = Some(current_offset);
        paginated_query.per_page = Some(self.per_page);
        
        let meilisearch_query = query_to_meilisearch_query(&paginated_query)?;
        
        match self.client.search(&self.index, meilisearch_query) {
            Ok(response) => {
                let hits: Vec<SearchHit> = response
                    .hits
                    .into_iter()
                    .map(meilisearch_hit_to_search_hit)
                    .collect();
                
                if hits.len() < self.per_page as usize {
                    // This was the last page
                    self.set_finished();
                } else {
                    // Update offset for next page
                    *self.current_offset.borrow_mut() = current_offset + self.per_page;
                }
                
                Ok(hits)
            }
            Err(error) => Err(error),
        }
    }
}

struct MeilisearchComponent;

impl MeilisearchComponent {
    const HOST_VAR: &'static str = "MEILISEARCH_HOST";
    const API_KEY_VAR: &'static str = "MEILISEARCH_API_KEY";
}

impl Guest for MeilisearchComponent {
    type SearchHitStream = SearchStream<MeilisearchStream>;

    fn create_index(name: String, schema: Option<Schema>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = MeilisearchApi::new(host, api_key);
                
                if let Some(schema) = schema {
                    let settings = schema_to_meilisearch_settings(&schema);
                    client.create_index_with_settings(&name, settings)
                } else {
                    client.create_index(&name)
                }
            })
        })
    }

    fn delete_index(name: String) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = MeilisearchApi::new(host, api_key);
                client.delete_index(&name)
            })
        })
    }

    fn list_indexes() -> Result<Vec<String>, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = MeilisearchApi::new(host, api_key);
                client.list_indexes()
            })
        })
    }

    fn upsert(index: String, doc: Doc) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = MeilisearchApi::new(host, api_key);
                let meilisearch_doc = doc_to_meilisearch_doc(doc)?;
                client.index_document(&index, &meilisearch_doc.id, meilisearch_doc.content)
            })
        })
    }

    fn upsert_many(index: String, docs: Vec<Doc>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = MeilisearchApi::new(host, api_key);
                let meilisearch_docs: Result<Vec<_>, _> = docs.into_iter().map(doc_to_meilisearch_doc).collect();
                client.bulk_index(&index, meilisearch_docs?)
            })
        })
    }

    fn delete(index: String, id: String) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = MeilisearchApi::new(host, api_key);
                client.delete_document(&index, &id)
            })
        })
    }

    fn delete_many(index: String, ids: Vec<String>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = MeilisearchApi::new(host, api_key);
                client.bulk_delete(&index, ids)
            })
        })
    }

    fn get(index: String, id: String) -> Result<Option<Doc>, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = MeilisearchApi::new(host, api_key);
                client.get_document(&index, &id)
            })
        })
    }

    fn search(index: String, query: SearchQuery) -> Result<SearchResults, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = MeilisearchApi::new(host, api_key);
                let meilisearch_query = query_to_meilisearch_query(&query)?;
                trace!("Executing Meilisearch query: {:?}", meilisearch_query);
                
                match client.search(&index, meilisearch_query) {
                    Ok(response) => Ok(meilisearch_response_to_results(response, &query)),
                    Err(error) => Err(error),
                }
            })
        })
    }

    fn stream_search(
        index: String,
        query: SearchQuery,
    ) -> Result<Self::SearchHitStream, SearchError> {
        Self::unwrapped_stream_search(index, query)
    }

    fn get_schema(index: String) -> Result<Schema, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = MeilisearchApi::new(host, api_key);
                let settings = client.get_settings(&index)?;
                Ok(meilisearch_settings_to_schema(&settings))
            })
        })
    }

    fn update_schema(index: String, schema: Schema) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = MeilisearchApi::new(host, api_key);
                let settings = schema_to_meilisearch_settings(&schema);
                client.update_settings(&index, settings)
            })
        })
    }
}

impl ExtendedGuest for MeilisearchComponent {
    type SearchHitStream = SearchStream<MeilisearchStream>;

    fn unwrapped_stream_search(
        index: String,
        query: SearchQuery,
    ) -> Result<Self::SearchHitStream, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, |error| Ok(SearchStream::new(MeilisearchStream::failed(error))), |host| {
            with_config_key(Self::API_KEY_VAR, |error| Ok(SearchStream::new(MeilisearchStream::failed(error))), |api_key| {
                let client = MeilisearchApi::new(host, api_key);
                Ok(SearchStream::new(MeilisearchStream::new(client, index, query)))
            })
        })
    }
}

type DurableMeilisearchComponent = DurableSearch<MeilisearchComponent>;

golem_search::export_search!(DurableMeilisearchComponent with_types_in golem_search);
