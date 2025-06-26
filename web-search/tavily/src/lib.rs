use crate::client::TavilySearchApi;
use crate::conversions::{convert_params_to_request, convert_response_to_results};
use golem_web_search::config::with_config_key;

use golem_web_search::durability::DurableWebSearch;
use golem_web_search::golem::web_search::types::{
    SearchError, SearchMetadata, SearchParams, SearchResult,
};
use golem_web_search::golem_web_search::web_search::web_search::{
    Guest, GuestSearchSession, SearchSession,
};

use golem_web_search::LOGGING_STATE;
use std::cell::RefCell;

mod client;
mod conversions;

struct TavilyWebSearchComponent;

impl TavilyWebSearchComponent {
    const API_KEY_ENV_VAR: &'static str = "TAVILY_API_KEY";
}

pub struct TavilySearchSession {
    client: TavilySearchApi,
    params: SearchParams,
    current_offset: RefCell<u32>,
    last_metadata: RefCell<Option<SearchMetadata>>,
    has_more_results: RefCell<bool>,
}

impl TavilySearchSession {
    fn new(client: TavilySearchApi, params: SearchParams) -> Self {
        Self {
            client,
            params,
            current_offset: RefCell::new(0),
            last_metadata: RefCell::new(None),
            has_more_results: RefCell::new(true),
        }
    }
}

impl GuestSearchSession for TavilySearchSession {
    fn next_page(&self) -> Result<SearchResult, SearchError> {
        if !*self.has_more_results.borrow() {
            return Err(SearchError::BackendError(
                "No more results available".to_string(),
            ));
        }

        let page_size = self.params.max_results.unwrap_or(10);
        let current_offset = *self.current_offset.borrow();
        let new_offset = current_offset + page_size;
        *self.current_offset.borrow_mut() = new_offset;

        let request = convert_params_to_request(&self.params, Some(new_offset));
        let response = self.client.search(request)?;
        let (results, metadata) = convert_response_to_results(response, &self.params);

        *self.last_metadata.borrow_mut() = metadata.clone();

        if results.is_empty() {
            *self.has_more_results.borrow_mut() = false;
            return Err(SearchError::BackendError("No more results".to_string()));
        }

        // Tavily doesn't support pagination in the same way, so we'll mark as no more results
        *self.has_more_results.borrow_mut() = false;

        results
            .into_iter()
            .next()
            .ok_or_else(|| SearchError::BackendError("No results returned".to_string()))
    }

    fn get_metadata(&self) -> Option<SearchMetadata> {
        self.last_metadata.borrow().clone()
    }
}

impl Guest for TavilyWebSearchComponent {
    type SearchSession = TavilySearchSession;

    fn start_search(params: SearchParams) -> Result<SearchSession, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(&[Self::API_KEY_ENV_VAR], Err, |keys| {
            let api_key = keys.get(Self::API_KEY_ENV_VAR).unwrap().to_owned();
            let client = TavilySearchApi::new(api_key);
            Ok(SearchSession::new(TavilySearchSession::new(client, params)))
        })
    }

    fn search_once(
        params: SearchParams,
    ) -> Result<(Vec<SearchResult>, Option<SearchMetadata>), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(&[Self::API_KEY_ENV_VAR], Err, |keys| {
            let api_key = keys.get(Self::API_KEY_ENV_VAR).unwrap().to_owned();
            let client = TavilySearchApi::new(api_key);
            let request = convert_params_to_request(&params, None);
            let response = client.search(request)?;
            let (results, metadata) = convert_response_to_results(response, &params);
            Ok((results, metadata))
        })
    }
}

type DurableTavilyWebSearchComponent = DurableWebSearch<TavilyWebSearchComponent>;

golem_web_search::export_web_search!(DurableTavilyWebSearchComponent with_types_in golem_web_search);
