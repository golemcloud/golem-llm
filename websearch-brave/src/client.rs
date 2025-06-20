use crate::exports::golem::web_search::web_search::GuestSearchSession;
use crate::golem::web_search::types::{SearchError, SearchMetadata, SearchParams, SearchResult, SafeSearchLevel};
use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use std::env;
use std::sync::{Arc, Mutex};
use url::Url;

const BRAVE_SEARCH_API_URL: &str = "https://api.search.brave.com/res/v1/web/search";

#[derive(Debug, Clone)]
pub struct BraveWebSearchClient {
    params: SearchParams,
    client: Client,
    api_key: String,
    state: Arc<Mutex<ClientState>>,
}

#[derive(Debug, Clone)]
struct ClientState {
    current_offset: u32,
    total_results: Option<u64>,
    search_time_ms: Option<f64>,
}

impl BraveWebSearchClient {
    pub fn new(params: SearchParams) -> Self {
        let api_key = env::var("BRAVE_API_KEY").unwrap_or_else(|_| {
            eprintln!("Warning: BRAVE_API_KEY not set, using dummy key");
            "dummy_key".to_string()
        });
        Self {
            params,
            client: Client::new(),
            api_key,
            state: Arc::new(Mutex::new(ClientState {
                current_offset: 0,
                total_results: None,
                search_time_ms: None,
            })),
        }
    }

    pub fn search_once(params: SearchParams) -> Result<(Vec<SearchResult>, Option<SearchMetadata>), SearchError> {
        let client = Self::new(params.clone());
        let results = client.next_page_durable()?;
        let metadata = client.get_metadata_durable();
        Ok((results, metadata))
    }

    fn build_search_url(&self) -> Result<Url, SearchError> {
        let mut url = Url::parse(BRAVE_SEARCH_API_URL)
            .map_err(|e| SearchError::BackendError(format!("Invalid URL: {}", e)))?;
        {
            let mut query_pairs = url.query_pairs_mut();
            query_pairs.append_pair("q", &self.params.query);
            if let Some(max_results) = self.params.max_results {
                query_pairs.append_pair("count", &max_results.to_string());
            }
            if let Some(safe_search) = &self.params.safe_search {
                let safe = match safe_search {
                    SafeSearchLevel::Off => "off",
                    SafeSearchLevel::Medium => "moderate",
                    SafeSearchLevel::High => "strict",
                };
                query_pairs.append_pair("safesearch", safe);
            }
            if let Some(language) = &self.params.language {
                query_pairs.append_pair("search_lang", language);
            }
            if let Some(region) = &self.params.region {
                query_pairs.append_pair("country", region);
            }
            // Brave does not support time_range, include_domains, exclude_domains directly
        }
        Ok(url)
    }

    async fn perform_search(&self) -> Result<BraveSearchResponse, SearchError> {
        let url = self.build_search_url()?;
        let response = self.client
            .get(url)
            .header("X-Subscription-Token", &self.api_key)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| SearchError::BackendError(format!("Request failed: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(SearchError::BackendError(format!("HTTP {}: {}", status, error_text)));
        }
        let search_response: BraveSearchResponse = response
            .json()
            .await
            .map_err(|e| SearchError::BackendError(format!("Failed to parse response: {}", e)))?;
        Ok(search_response)
    }
}

impl GuestSearchSession for BraveWebSearchClient {
    fn new(params: SearchParams) -> Self {
        Self::new(params)
    }
    fn next_page(&self) -> Result<Vec<SearchResult>, SearchError> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| SearchError::BackendError(format!("Failed to create runtime: {}", e)))?;
        rt.block_on(async {
            let response = self.perform_search().await?;
            let mut state = self.state.lock().unwrap();
            state.total_results = response.web.as_ref().and_then(|w| w.total).map(|t| t as u64);
            // Brave does not provide search_time_ms
            let results = response.web
                .and_then(|w| Some(w.results))
                .unwrap_or_default()
                .into_iter()
                .map(super::conversions::convert_brave_result)
                .collect();
            Ok(results)
        })
    }
    fn get_metadata(&self) -> Option<SearchMetadata> {
        let state = self.state.lock().unwrap();
        Some(SearchMetadata {
            query: self.params.query.clone(),
            total_results: state.total_results,
            search_time_ms: state.search_time_ms,
            safe_search: self.params.safe_search.clone(),
            language: self.params.language.clone(),
            region: self.params.region.clone(),
            next_page_token: None,
            rate_limits: None,
        })
    }
}

pub trait ExtendedSearchSession: GuestSearchSession {
    fn new_durable(params: SearchParams) -> Self;
    fn next_page_durable(&self) -> Result<Vec<SearchResult>, SearchError>;
    fn get_metadata_durable(&self) -> Option<SearchMetadata>;
}

impl ExtendedSearchSession for BraveWebSearchClient {
    fn new_durable(params: SearchParams) -> Self {
        Self::new(params)
    }
    fn next_page_durable(&self) -> Result<Vec<SearchResult>, SearchError> {
        self.next_page()
    }
    fn get_metadata_durable(&self) -> Option<SearchMetadata> {
        self.get_metadata()
    }
}

#[derive(Debug, Deserialize)]
pub struct BraveSearchResponse {
    pub web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
pub struct BraveWebResults {
    pub results: Vec<BraveSearchItem>,
    pub total: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct BraveSearchItem {
    pub title: String,
    pub url: String,
    pub description: String,
    pub sitelinks: Option<Vec<String>>,
    pub thumbnail: Option<String>,
    pub date_published: Option<String>,
} 