use crate::client::TypesenseApi;
use crate::conversions::{
    doc_to_typesense_doc, export_params_from_query, query_to_typesense_params, 
    schema_to_typesense_fields, typesense_hit_to_search_hit, typesense_response_to_results,
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

struct TypesenseStream {
    client: TypesenseApi,
    collection: String,
    query: SearchQuery,
    current_page: RefCell<u32>,
    finished: RefCell<bool>,
    failure: Option<SearchError>,
}

impl TypesenseStream {
    fn new(client: TypesenseApi, collection: String, query: SearchQuery) -> Self {
        Self {
            client,
            collection,
            query,
            current_page: RefCell::new(1),
            finished: RefCell::new(false),
            failure: None,
        }
    }

    fn failed(error: SearchError) -> Self {
        Self {
            client: TypesenseApi::empty(),
            collection: String::new(),
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
            current_page: RefCell::new(1),
            finished: RefCell::new(true),
            failure: Some(error),
        }
    }
}

impl SearchStreamState for TypesenseStream {
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
        let current_page = *self.current_page.borrow();
        
        // For streaming, we use the export API for better performance
        if current_page == 1 {
            // First page - use regular search for better ranking
            let mut params = query_to_typesense_params(&self.query)?;
            params.insert("page".to_string(), current_page.to_string());
            
            match self.client.search(&self.collection, &params) {
                Ok(response) => {
                    if response.hits.is_empty() {
                        self.set_finished();
                        Ok(vec![])
                    } else {
                        *self.current_page.borrow_mut() = current_page + 1;
                        Ok(response.hits.into_iter().map(typesense_hit_to_search_hit).collect())
                    }
                }
                Err(error) => Err(error),
            }
        } else {
            // Subsequent pages - use export API for efficiency
            let export_params = export_params_from_query(&self.query);
            match self.client.export(&self.collection, &export_params) {
                Ok(docs) => {
                    if docs.is_empty() {
                        self.set_finished();
                        Ok(vec![])
                    } else {
                        self.set_finished(); // Export gets all remaining results
                        Ok(docs.into_iter().map(|doc| SearchHit {
                            id: doc.id,
                            score: None,
                            content: Some(doc.content),
                            highlights: None,
                        }).collect())
                    }
                }
                Err(error) => Err(error),
            }
        }
    }
}

struct TypesenseComponent;

impl TypesenseComponent {
    const HOST_VAR: &'static str = "TYPESENSE_HOST";
    const API_KEY_VAR: &'static str = "TYPESENSE_API_KEY";
}

impl Guest for TypesenseComponent {
    type SearchHitStream = SearchStream<TypesenseStream>;

    fn create_index(name: String, schema: Option<Schema>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = TypesenseApi::new(host, api_key);
                client.create_collection(&name, schema.as_ref())
            })
        })
    }

    fn delete_index(name: String) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = TypesenseApi::new(host, api_key);
                client.delete_collection(&name)
            })
        })
    }

    fn list_indexes() -> Result<Vec<String>, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = TypesenseApi::new(host, api_key);
                client.list_collections()
            })
        })
    }

    fn upsert(index: String, doc: Doc) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = TypesenseApi::new(host, api_key);
                let ts_doc = doc_to_typesense_doc(doc)?;
                client.upsert_document(&index, ts_doc)
            })
        })
    }

    fn upsert_many(index: String, docs: Vec<Doc>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = TypesenseApi::new(host, api_key);
                let ts_docs: Result<Vec<_>, _> = docs.into_iter().map(doc_to_typesense_doc).collect();
                client.bulk_upsert(&index, ts_docs?)
            })
        })
    }

    fn delete(index: String, id: String) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = TypesenseApi::new(host, api_key);
                client.delete_document(&index, &id)
            })
        })
    }

    fn delete_many(index: String, ids: Vec<String>) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = TypesenseApi::new(host, api_key);
                client.bulk_delete(&index, ids)
            })
        })
    }

    fn get(index: String, id: String) -> Result<Option<Doc>, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = TypesenseApi::new(host, api_key);
                client.get_document(&index, &id)
            })
        })
    }

    fn search(index: String, query: SearchQuery) -> Result<SearchResults, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = TypesenseApi::new(host, api_key);
                let params = query_to_typesense_params(&query)?;
                trace!("Executing search query: {:?}", params);
                
                match client.search(&index, &params) {
                    Ok(response) => Ok(typesense_response_to_results(response, &query)),
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
                let client = TypesenseApi::new(host, api_key);
                client.get_collection_schema(&index)
            })
        })
    }

    fn update_schema(index: String, schema: Schema) -> Result<(), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, Err, |host| {
            with_config_key(Self::API_KEY_VAR, Err, |api_key| {
                let client = TypesenseApi::new(host, api_key);
                client.update_collection_schema(&index, &schema)
            })
        })
    }
}

impl ExtendedGuest for TypesenseComponent {
    type SearchHitStream = SearchStream<TypesenseStream>;

    fn unwrapped_stream_search(
        index: String,
        query: SearchQuery,
    ) -> Result<Self::SearchHitStream, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::HOST_VAR, |error| Ok(SearchStream::new(TypesenseStream::failed(error))), |host| {
            with_config_key(Self::API_KEY_VAR, |error| Ok(SearchStream::new(TypesenseStream::failed(error))), |api_key| {
                let client = TypesenseApi::new(host, api_key);
                Ok(SearchStream::new(TypesenseStream::new(client, index, query)))
            })
        })
    }
}

type DurableTypesenseComponent = DurableSearch<TypesenseComponent>;

golem_search::export_search!(DurableTypesenseComponent with_types_in golem_search);
