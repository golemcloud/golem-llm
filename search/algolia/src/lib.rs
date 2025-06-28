use crate::client::AlgoliaApi;
use crate::conversions::{
    doc_to_algolia_doc, algolia_hit_to_search_hit, algolia_response_to_results, query_to_algolia_query,
    schema_to_algolia_settings, algolia_settings_to_schema,
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

struct AlgoliaStream {
    client: AlgoliaApi,
    index: String,
    query: SearchQuery,
    cursor: RefCell<Option<String>>,
    finished: RefCell<bool>,
    failure: Option<SearchError>,
}

impl AlgoliaStream {
    fn new(client: AlgoliaApi, index: String, query: SearchQuery) -> Self {
        Self {
            client,
            index,
            query,
            cursor: RefCell::new(None),
            finished: RefCell::new(false),
            failure: None,
        }
    }

    fn failed(error: SearchError) -> Self {
        Self {
            client: AlgoliaApi::empty(),
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
            cursor: RefCell::new(None),
            finished: RefCell::new(true),
            failure: Some(error),
        }
    }
}

impl SearchStreamState for AlgoliaStream {
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
        let cursor = self.cursor.borrow().clone();
        
        // Use browse API for streaming
        match self.client.browse(&self.index, cursor.as_deref()) {
            Ok(response) => {
                if response.hits.is_empty() {
                    self.set_finished();
                    Ok(vec![])
                } else {
                    *self.cursor.borrow_mut() = response.cursor;
                    if response.cursor.is_none() {
                        self.set_finished();
                    }
                    Ok(response.hits.into_iter().map(algolia_hit_to_search_hit).collect())
                }
            }
            Err(error) => Err(error),
        }
    }
}

struct AlgoliaComponent;

impl AlgoliaComponent {
    const APP_ID_VAR: &'static str = "ALGOLIA_APPLICATION_ID";
    const API_KEY_VAR: &'static str = "ALGOLIA_API_KEY";
}

impl Guest for AlgoliaComponent {
    type SearchHitStream = SearchStream<AlgoliaStream>;

    fn create_index(name: String, schema: Option<Schema>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::APP_ID_VAR, Err, |app_id| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = AlgoliaApi::new(app_id, api_key);
                
                if let Some(schema) = schema {
                    let settings = schema_to_algolia_settings(&schema);
                    client.create_index_with_schema(&name, settings)
                } else {
                    client.create_index(&name)
                }
            })
        })
    }

    fn delete_index(name: String) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::APP_ID_VAR, Err, |app_id| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = AlgoliaApi::new(app_id, api_key);
                client.delete_index(&name)
            })
        })
    }

    fn list_indexes() -> Result<Vec<String>, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::APP_ID_VAR, Err, |app_id| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = AlgoliaApi::new(app_id, api_key);
                client.list_indexes()
            })
        })
    }

    fn upsert(index: String, doc: Doc) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::APP_ID_VAR, Err, |app_id| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = AlgoliaApi::new(app_id, api_key);
                let algolia_doc = doc_to_algolia_doc(doc)?;
                client.index_document(&index, algolia_doc)
            })
        })
    }

    fn upsert_many(index: String, docs: Vec<Doc>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::APP_ID_VAR, Err, |app_id| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = AlgoliaApi::new(app_id, api_key);
                let algolia_docs: Result<Vec<_>, _> = docs.into_iter().map(doc_to_algolia_doc).collect();
                client.bulk_index(&index, algolia_docs?)
            })
        })
    }

    fn delete(index: String, id: String) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::APP_ID_VAR, Err, |app_id| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = AlgoliaApi::new(app_id, api_key);
                client.delete_document(&index, &id)
            })
        })
    }

    fn delete_many(index: String, ids: Vec<String>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::APP_ID_VAR, Err, |app_id| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = AlgoliaApi::new(app_id, api_key);
                client.bulk_delete(&index, ids)
            })
        })
    }

    fn get(index: String, id: String) -> Result<Option<Doc>, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::APP_ID_VAR, Err, |app_id| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = AlgoliaApi::new(app_id, api_key);
                client.get_document(&index, &id)
            })
        })
    }

    fn search(index: String, query: SearchQuery) -> Result<SearchResults, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::APP_ID_VAR, Err, |app_id| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = AlgoliaApi::new(app_id, api_key);
                let algolia_query = query_to_algolia_query(&query)?;
                trace!("Executing Algolia search query: {:?}", algolia_query);
                
                match client.search(&index, algolia_query) {
                    Ok(response) => Ok(algolia_response_to_results(response, &query)),
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

        with_config_key(Self::APP_ID_VAR, Err, |app_id| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = AlgoliaApi::new(app_id, api_key);
                let settings = client.get_settings(&index)?;
                algolia_settings_to_schema(&settings)
            })
        })
    }

    fn update_schema(index: String, schema: Schema) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::APP_ID_VAR, Err, |app_id| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = AlgoliaApi::new(app_id, api_key);
                let settings = schema_to_algolia_settings(&schema);
                client.update_settings(&index, settings)
            })
        })
    }
}

impl ExtendedGuest for AlgoliaComponent {
    type SearchHitStream = SearchStream<AlgoliaStream>;

    fn unwrapped_stream_search(
        index: String,
        query: SearchQuery,
    ) -> Result<Self::SearchHitStream, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::APP_ID_VAR, |error| Ok(SearchStream::new(AlgoliaStream::failed(error))), |app_id| {
            with_config_key(Self::API_KEY_VAR, |error| Ok(SearchStream::new(AlgoliaStream::failed(error))), |api_key| {
                let client = AlgoliaApi::new(app_id, api_key);
                Ok(SearchStream::new(AlgoliaStream::new(client, index, query)))
            })
        })
    }
}

type DurableAlgoliaComponent = DurableSearch<AlgoliaComponent>;

golem_search::export_search!(DurableAlgoliaComponent with_types_in golem_search);
