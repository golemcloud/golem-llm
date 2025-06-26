use crate::client::GoogleSearchApi;
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

struct GoogleWebSearchComponent;

impl GoogleWebSearchComponent {
    const API_KEY_ENV_VAR: &'static str = "GOOGLE_API_KEY";
    const SEARCH_ENGINE_ID_ENV_VAR: &'static str = "GOOGLE_SEARCH_ENGINE_ID";
}

pub struct GoogleSearchSession {
    client: GoogleSearchApi,
    params: SearchParams,
    current_start_index: RefCell<u32>,
    last_metadata: RefCell<Option<SearchMetadata>>,
    has_more_results: RefCell<bool>,
}

impl GoogleSearchSession {
    fn new(client: GoogleSearchApi, params: SearchParams) -> Self {
        Self {
            client,
            params,
            current_start_index: RefCell::new(0),
            last_metadata: RefCell::new(None),
            has_more_results: RefCell::new(true),
        }
    }
}

impl GuestSearchSession for GoogleSearchSession {
    fn next_page(&self) -> Result<SearchResult, SearchError> {
        if !*self.has_more_results.borrow_mut() {
            return Err(SearchError::BackendError(
                "No more results available".to_string(),
            ));
        }
        *self.current_start_index.borrow_mut() = *self.current_start_index.borrow_mut() + 1_u32;
        let request =
            convert_params_to_request(&self.params, Some(*self.current_start_index.borrow()));
        let response = self.client.search(request)?;
        let (results, metadata) = convert_response_to_results(response, &self.params);

        *self.last_metadata.borrow_mut() = metadata.clone();

        if results.is_empty() {
            *self.has_more_results.borrow_mut() = false;
            return Err(SearchError::BackendError("No more results".to_string()));
        }

        if let Some(metadata) = &metadata {
            *self.has_more_results.borrow_mut() = metadata.next_page_token.is_some();
        } else {
            *self.has_more_results.borrow_mut() = false;
        }

        results
            .into_iter()
            .next()
            .ok_or_else(|| SearchError::BackendError("No results returned".to_string()))
    }

    fn get_metadata(&self) -> Option<SearchMetadata> {
        self.last_metadata.borrow().clone()
    }
}

impl Guest for GoogleWebSearchComponent {
    type SearchSession = GoogleSearchSession;

    fn start_search(params: SearchParams) -> Result<SearchSession, SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(
            &[Self::API_KEY_ENV_VAR, Self::SEARCH_ENGINE_ID_ENV_VAR],
            Err,
            |keys| {
                let api_key = keys.get(Self::API_KEY_ENV_VAR).unwrap().to_owned();
                let search_engine_id = keys.get(Self::SEARCH_ENGINE_ID_ENV_VAR).unwrap().to_owned();
                let client = GoogleSearchApi::new(api_key, search_engine_id);
                Ok(SearchSession::new(GoogleSearchSession::new(client, params)))
            },
        )
    }

    fn search_once(
        params: SearchParams,
    ) -> Result<(Vec<SearchResult>, Option<SearchMetadata>), SearchError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(
            &[Self::API_KEY_ENV_VAR, Self::SEARCH_ENGINE_ID_ENV_VAR],
            Err,
            |keys| {
                let api_key = keys.get(Self::API_KEY_ENV_VAR).unwrap().to_owned();
                let search_engine_id = keys.get(Self::SEARCH_ENGINE_ID_ENV_VAR).unwrap().to_owned();
                let client = GoogleSearchApi::new(api_key, search_engine_id);
                let request = convert_params_to_request(&params, None);
                let response = client.search(request)?;
                let (results, metadata) = convert_response_to_results(response, &params);
                Ok((results, metadata))
            },
        )
    }
}

type DurableGoogleWebSearchComponent = DurableWebSearch<GoogleWebSearchComponent>;

golem_web_search::export_web_search!(DurableGoogleWebSearchComponent with_types_in golem_web_search);
