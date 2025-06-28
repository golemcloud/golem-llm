use crate::client::ElasticSearchApi;
use crate::conversions::{
    doc_to_es_doc, es_hit_to_search_hit, es_response_to_results, query_to_es_query,
    schema_to_mapping,
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

struct ElasticSearchStream {
    client: ElasticSearchApi,
    index: String,
    query: SearchQuery,
    scroll_id: RefCell<Option<String>>,
    finished: RefCell<bool>,
    failure: Option<SearchError>,
}

impl ElasticSearchStream {
    fn new(client: ElasticSearchApi, index: String, query: SearchQuery) -> Self {
        Self {
            client,
            index,
            query,
            scroll_id: RefCell::new(None),
            finished: RefCell::new(false),
            failure: None,
        }
    }

    fn failed(error: SearchError) -> Self {
        Self {
            client: ElasticSearchApi::empty(),
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
            scroll_id: RefCell::new(None),
            finished: RefCell::new(true),
            failure: Some(error),
        }
    }
}

impl SearchStreamState for ElasticSearchStream {
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
        let scroll_id = self.scroll_id.borrow().clone();
        
        if let Some(scroll_id) = scroll_id {
            match self.client.scroll(&scroll_id) {
                Ok(response) => {
                    if response.hits.hits.is_empty() {
                        self.set_finished();
                        Ok(vec![])
                    } else {
                        *self.scroll_id.borrow_mut() = response.scroll_id;
                        Ok(response.hits.hits.into_iter().map(es_hit_to_search_hit).collect())
                    }
                }
                Err(error) => Err(error),
            }
        } else {
            let es_query = query_to_es_query(&self.query)?;
            match self.client.search_with_scroll(&self.index, es_query) {
                Ok(response) => {
                    if response.hits.hits.is_empty() {
                        self.set_finished();
                        Ok(vec![])
                    } else {
                        *self.scroll_id.borrow_mut() = response.scroll_id;
                        Ok(response.hits.hits.into_iter().map(es_hit_to_search_hit).collect())
                    }
                }
                Err(error) => Err(error),
            }
        }
    }
}

struct ElasticSearchComponent;

impl ElasticSearchComponent {
    const ENDPOINT_VAR: &'static str = "ELASTIC_ENDPOINT";
    const PASSWORD_VAR: &'static str = "ELASTIC_PASSWORD";
    const USERNAME_VAR: &'static str = "ELASTIC_USERNAME";
}

impl Guest for ElasticSearchComponent {
    type SearchHitStream = SearchStream<ElasticSearchStream>;

    fn create_index(name: String, schema: Option<Schema>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::USERNAME_VAR, Err, |username| {
                with_config_key(Self::PASSWORD_VAR, Err, |password| {
                    let client = ElasticSearchApi::new(endpoint, username, password);
                    
                    if let Some(schema) = schema {
                        let mapping = schema_to_mapping(&schema);
                        client.create_index_with_mapping(&name, mapping)
                    } else {
                        client.create_index(&name)
                    }
                })
            })
        })
    }

    fn delete_index(name: String) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::USERNAME_VAR, Err, |username| {
                with_config_key(Self::PASSWORD_VAR, Err, |password| {
                    let client = ElasticSearchApi::new(endpoint, username, password);
                    client.delete_index(&name)
                })
            })
        })
    }

    fn list_indexes() -> Result<Vec<String>, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::USERNAME_VAR, Err, |username| {
                with_config_key(Self::PASSWORD_VAR, Err, |password| {
                    let client = ElasticSearchApi::new(endpoint, username, password);
                    client.list_indexes()
                })
            })
        })
    }

    fn upsert(index: String, doc: Doc) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::USERNAME_VAR, Err, |username| {
                with_config_key(Self::PASSWORD_VAR, Err, |password| {
                    let client = ElasticSearchApi::new(endpoint, username, password);
                    let es_doc = doc_to_es_doc(doc)?;
                    client.index_document(&index, &es_doc.id, es_doc.content)
                })
            })
        })
    }

    fn upsert_many(index: String, docs: Vec<Doc>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::USERNAME_VAR, Err, |username| {
                with_config_key(Self::PASSWORD_VAR, Err, |password| {
                    let client = ElasticSearchApi::new(endpoint, username, password);
                    let es_docs: Result<Vec<_>, _> = docs.into_iter().map(doc_to_es_doc).collect();
                    client.bulk_index(&index, es_docs?)
                })
            })
        })
    }

    fn delete(index: String, id: String) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::USERNAME_VAR, Err, |username| {
                with_config_key(Self::PASSWORD_VAR, Err, |password| {
                    let client = ElasticSearchApi::new(endpoint, username, password);
                    client.delete_document(&index, &id)
                })
            })
        })
    }

    fn delete_many(index: String, ids: Vec<String>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::USERNAME_VAR, Err, |username| {
                with_config_key(Self::PASSWORD_VAR, Err, |password| {
                    let client = ElasticSearchApi::new(endpoint, username, password);
                    client.bulk_delete(&index, ids)
                })
            })
        })
    }

    fn get(index: String, id: String) -> Result<Option<Doc>, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::USERNAME_VAR, Err, |username| {
                with_config_key(Self::PASSWORD_VAR, Err, |password| {
                    let client = ElasticSearchApi::new(endpoint, username, password);
                    client.get_document(&index, &id)
                })
            })
        })
    }

    fn search(index: String, query: SearchQuery) -> Result<SearchResults, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::USERNAME_VAR, Err, |username| {
                with_config_key(Self::PASSWORD_VAR, Err, |password| {
                    let client = ElasticSearchApi::new(endpoint, username, password);
                    let es_query = query_to_es_query(&query)?;
                    trace!("Executing search query: {:?}", es_query);
                    
                    match client.search(&index, es_query) {
                        Ok(response) => Ok(es_response_to_results(response, &query)),
                        Err(error) => Err(error),
                    }
                })
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

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::USERNAME_VAR, Err, |username| {
                with_config_key(Self::PASSWORD_VAR, Err, |password| {
                    let client = ElasticSearchApi::new(endpoint, username, password);
                    client.get_mapping(&index)
                })
            })
        })
    }

    fn update_schema(index: String, schema: Schema) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::USERNAME_VAR, Err, |username| {
                with_config_key(Self::PASSWORD_VAR, Err, |password| {
                    let client = ElasticSearchApi::new(endpoint, username, password);
                    let mapping = schema_to_mapping(&schema);
                    client.update_mapping(&index, mapping)
                })
            })
        })
    }
}

impl ExtendedGuest for ElasticSearchComponent {
    type SearchHitStream = SearchStream<ElasticSearchStream>;

    fn unwrapped_stream_search(
        index: String,
        query: SearchQuery,
    ) -> Result<Self::SearchHitStream, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, |error| Ok(SearchStream::new(ElasticSearchStream::failed(error))), |endpoint| {
            with_config_key(Self::USERNAME_VAR, |error| Ok(SearchStream::new(ElasticSearchStream::failed(error))), |username| {
                with_config_key(Self::PASSWORD_VAR, |error| Ok(SearchStream::new(ElasticSearchStream::failed(error))), |password| {
                    let client = ElasticSearchApi::new(endpoint, username, password);
                    Ok(SearchStream::new(ElasticSearchStream::new(client, index, query)))
                })
            })
        })
    }
}

type DurableElasticSearchComponent = DurableSearch<ElasticSearchComponent>;

golem_search::export_search!(DurableElasticSearchComponent with_types_in golem_search);
