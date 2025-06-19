use crate::exports::golem::web_search::web_search::GuestSearchSession;
use crate::golem::web_search::types::{SearchError, SearchMetadata, SearchParams, SearchResult, SafeSearchLevel, TimeRange};
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::{Arc, Mutex};

const SERPER_SEARCH_API_URL: &str = "https://google.serper.dev/search";

#[derive(Debug, Clone)]
pub struct SerperWebSearchClient {
    params: SearchParams,
    client: Client,
    api_key: String,
    state: Arc<Mutex<ClientState>>,
}

#[derive(Debug, Clone)]
struct ClientState {
    total_results: Option<u64>,
    search_time_ms: Option<f64>,
}

impl SerperWebSearchClient {
    pub fn new(params: SearchParams) -> Self {
        let api_key = env::var("SERPER_API_KEY").unwrap_or_else(|_| {
            eprintln!("Warning: SERPER_API_KEY not set, using dummy key");
            "dummy_key".to_string()
        });
        Self {
            params,
            client: Client::new(),
            api_key,
            state: Arc::new(Mutex::new(ClientState {
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

    fn build_search_body(&self) -> SerperSearchRequest {
        SerperSearchRequest {
            q: self.params.query.clone(),
        }
    }

    async fn perform_search(&self) -> Result<SerperSearchResponse, SearchError> {
        let body = self.build_search_body();
        let response = self.client
            .post(SERPER_SEARCH_API_URL)
            .header("X-API-KEY", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| SearchError::BackendError(format!("Request failed: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(SearchError::BackendError(format!("HTTP {}: {}", status, error_text)));
        }
        let search_response: SerperSearchResponse = response
            .json()
            .await
            .map_err(|e| SearchError::BackendError(format!("Failed to parse response: {}", e)))?;
        Ok(search_response)
    }
}

impl GuestSearchSession for SerperWebSearchClient {
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
            state.total_results = Some(response.organic.len() as u64);
            // No search_time_ms in Serper response
            let results = response.organic
                .into_iter()
                .map(super::conversions::convert_serper_result)
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

impl ExtendedSearchSession for SerperWebSearchClient {
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

#[derive(Debug, Serialize)]
struct SerperSearchRequest {
    q: String,
}

#[derive(Debug, Deserialize)]
pub struct SerperSearchResponse {
    pub organic: Vec<SerperSearchItem>,
}

#[derive(Debug, Deserialize)]
pub struct SerperSearchItem {
    pub title: String,
    pub link: String,
    pub snippet: String,
} 