use crate::client::OpenSearchApi;
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

struct OpenSearchStream {
    client: OpenSearchApi,
    index: String,
    query: SearchQuery,
    scroll_id: RefCell<Option<String>>,
    finished: RefCell<bool>,
    failure: Option<SearchError>,
}

impl OpenSearchStream {
    fn new(client: OpenSearchApi, index: String, query: SearchQuery) -> Self {
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
            client: OpenSearchApi::empty(),
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

impl SearchStreamState for OpenSearchStream {
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

struct OpenSearchComponent;

impl OpenSearchComponent {
    const ENDPOINT_VAR: &'static str = "OPENSEARCH_ENDPOINT";
    const ACCESS_KEY_VAR: &'static str = "AWS_ACCESS_KEY_ID";
    const SECRET_KEY_VAR: &'static str = "AWS_SECRET_ACCESS_KEY";
    const REGION_VAR: &'static str = "AWS_REGION";
}

impl Guest for OpenSearchComponent {
    type SearchHitStream = SearchStream<OpenSearchStream>;

    fn create_index(name: String, schema: Option<Schema>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::ACCESS_KEY_VAR, Err, |access_key| {
                with_config_key(Self::SECRET_KEY_VAR, Err, |secret_key| {
                    with_config_key(Self::REGION_VAR, Err, |region| {
                        let client = OpenSearchApi::new(endpoint, access_key, secret_key, region);
                        
                        if let Some(schema) = schema {
                            let mapping = schema_to_mapping(&schema);
                            client.create_index_with_mapping(&name, mapping)
                        } else {
                            client.create_index(&name)
                        }
                    })
                })
            })
        })
    }

    fn delete_index(name: String) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::ACCESS_KEY_VAR, Err, |access_key| {
                with_config_key(Self::SECRET_KEY_VAR, Err, |secret_key| {
                    with_config_key(Self::REGION_VAR, Err, |region| {
                        let client = OpenSearchApi::new(endpoint, access_key, secret_key, region);
                        client.delete_index(&name)
                    })
                })
            })
        })
    }

    fn list_indexes() -> Result<Vec<String>, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::ACCESS_KEY_VAR, Err, |access_key| {
                with_config_key(Self::SECRET_KEY_VAR, Err, |secret_key| {
                    with_config_key(Self::REGION_VAR, Err, |region| {
                        let client = OpenSearchApi::new(endpoint, access_key, secret_key, region);
                        client.list_indexes()
                    })
                })
            })
        })
    }

    fn upsert(index: String, doc: Doc) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::ACCESS_KEY_VAR, Err, |access_key| {
                with_config_key(Self::SECRET_KEY_VAR, Err, |secret_key| {
                    with_config_key(Self::REGION_VAR, Err, |region| {
                        let client = OpenSearchApi::new(endpoint, access_key, secret_key, region);
                        let es_doc = doc_to_es_doc(doc)?;
                        client.index_document(&index, &es_doc.id, es_doc.content)
                    })
                })
            })
        })
    }

    fn upsert_many(index: String, docs: Vec<Doc>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::ACCESS_KEY_VAR, Err, |access_key| {
                with_config_key(Self::SECRET_KEY_VAR, Err, |secret_key| {
                    with_config_key(Self::REGION_VAR, Err, |region| {
                        let client = OpenSearchApi::new(endpoint, access_key, secret_key, region);
                        let es_docs: Result<Vec<_>, _> = docs.into_iter().map(doc_to_es_doc).collect();
                        client.bulk_index(&index, es_docs?)
                    })
                })
            })
        })
    }

    fn delete(index: String, id: String) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::ACCESS_KEY_VAR, Err, |access_key| {
                with_config_key(Self::SECRET_KEY_VAR, Err, |secret_key| {
                    with_config_key(Self::REGION_VAR, Err, |region| {
                        let client = OpenSearchApi::new(endpoint, access_key, secret_key, region);
                        client.delete_document(&index, &id)
                    })
                })
            })
        })
    }

    fn delete_many(index: String, ids: Vec<String>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::ACCESS_KEY_VAR, Err, |access_key| {
                with_config_key(Self::SECRET_KEY_VAR, Err, |secret_key| {
                    with_config_key(Self::REGION_VAR, Err, |region| {
                        let client = OpenSearchApi::new(endpoint, access_key, secret_key, region);
                        client.bulk_delete(&index, ids)
                    })
                })
            })
        })
    }

    fn get(index: String, id: String) -> Result<Option<Doc>, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::ACCESS_KEY_VAR, Err, |access_key| {
                with_config_key(Self::SECRET_KEY_VAR, Err, |secret_key| {
                    with_config_key(Self::REGION_VAR, Err, |region| {
                        let client = OpenSearchApi::new(endpoint, access_key, secret_key, region);
                        client.get_document(&index, &id)
                    })
                })
            })
        })
    }

    fn search(index: String, query: SearchQuery) -> Result<SearchResults, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::ACCESS_KEY_VAR, Err, |access_key| {
                with_config_key(Self::SECRET_KEY_VAR, Err, |secret_key| {
                    with_config_key(Self::REGION_VAR, Err, |region| {
                        let client = OpenSearchApi::new(endpoint, access_key, secret_key, region);
                        let es_query = query_to_es_query(&query)?;
                        trace!("Executing search query: {:?}", es_query);
                        
                        match client.search(&index, es_query) {
                            Ok(response) => Ok(es_response_to_results(response, &query)),
                            Err(error) => Err(error),
                        }
                    })
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
            with_config_key(Self::ACCESS_KEY_VAR, Err, |access_key| {
                with_config_key(Self::SECRET_KEY_VAR, Err, |secret_key| {
                    with_config_key(Self::REGION_VAR, Err, |region| {
                        let client = OpenSearchApi::new(endpoint, access_key, secret_key, region);
                        client.get_mapping(&index)
                    })
                })
            })
        })
    }

    fn update_schema(index: String, schema: Schema) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, Err, |endpoint| {
            with_config_key(Self::ACCESS_KEY_VAR, Err, |access_key| {
                with_config_key(Self::SECRET_KEY_VAR, Err, |secret_key| {
                    with_config_key(Self::REGION_VAR, Err, |region| {
                        let client = OpenSearchApi::new(endpoint, access_key, secret_key, region);
                        let mapping = schema_to_mapping(&schema);
                        client.update_mapping(&index, mapping)
                    })
                })
            })
        })
    }
}

impl ExtendedGuest for OpenSearchComponent {
    type SearchHitStream = SearchStream<OpenSearchStream>;

    fn unwrapped_stream_search(
        index: String,
        query: SearchQuery,
    ) -> Result<Self::SearchHitStream, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENDPOINT_VAR, |error| Ok(SearchStream::new(OpenSearchStream::failed(error))), |endpoint| {
            with_config_key(Self::ACCESS_KEY_VAR, |error| Ok(SearchStream::new(OpenSearchStream::failed(error))), |access_key| {
                with_config_key(Self::SECRET_KEY_VAR, |error| Ok(SearchStream::new(OpenSearchStream::failed(error))), |secret_key| {
                    with_config_key(Self::REGION_VAR, |error| Ok(SearchStream::new(OpenSearchStream::failed(error))), |region| {
                        let client = OpenSearchApi::new(endpoint, access_key, secret_key, region);
                        Ok(SearchStream::new(OpenSearchStream::new(client, index, query)))
                    })
                })
            })
        })
    }
}

type DurableOpenSearchComponent = DurableSearch<OpenSearchComponent>;

golem_search::export_search!(DurableOpenSearchComponent with_types_in golem_search);
