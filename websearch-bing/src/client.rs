use crate::exports::golem::web_search::web_search::GuestSearchSession;
use crate::golem::web_search::types::{SearchError, SearchMetadata, SearchParams, SearchResult, SafeSearchLevel, TimeRange};
use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex};
use url::Url;

const BING_SEARCH_API_URL: &str = "https://api.bing.microsoft.com/v7.0/search";

#[derive(Debug, Clone)]
pub struct BingWebSearchClient {
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

impl BingWebSearchClient {
    pub fn new(params: SearchParams) -> Self {
        let api_key = env::var("BING_API_KEY").unwrap_or_else(|_| {
            eprintln!("Warning: BING_API_KEY not set, using dummy key");
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
        let state = self.state.lock().unwrap();
        let current_offset = state.current_offset;
        drop(state);

        let mut url = Url::parse(BING_SEARCH_API_URL)
            .map_err(|e| SearchError::BackendError(format!("Invalid URL: {}", e)))?;

        let mut query_params = HashMap::new();
        query_params.insert("q".to_string(), self.params.query.clone());

        // Add pagination
        if current_offset > 0 {
            query_params.insert("offset".to_string(), current_offset.to_string());
        }

        // Add max results (Bing API uses 10 as default, max 50)
        let max_results = self.params.max_results.unwrap_or(10).min(50);
        query_params.insert("count".to_string(), max_results.to_string());

        // Add safe search
        if let Some(safe_search) = &self.params.safe_search {
            let safe_search_value = match safe_search {
                SafeSearchLevel::Off => "Off",
                SafeSearchLevel::Medium => "Moderate",
                SafeSearchLevel::High => "Strict",
            };
            query_params.insert("safeSearch".to_string(), safe_search_value.to_string());
        }

        // Add language
        if let Some(language) = &self.params.language {
            query_params.insert("setLang".to_string(), language.clone());
        }

        // Add region
        if let Some(region) = &self.params.region {
            query_params.insert("cc".to_string(), region.clone());
        }

        // Add date range
        if let Some(time_range) = &self.params.time_range {
            let date_range = match time_range {
                TimeRange::Day => "Day",
                TimeRange::Week => "Week",
                TimeRange::Month => "Month",
                TimeRange::Year => "Year",
            };
            query_params.insert("freshness".to_string(), date_range.to_string());
        }

        // Add site restrictions
        if let Some(include_domains) = &self.params.include_domains {
            if !include_domains.is_empty() {
                let sites = include_domains.join(" OR ");
                let new_query = format!("{} site:({})", self.params.query, sites);
                query_params.insert("q".to_string(), new_query);
            }
        }

        if let Some(exclude_domains) = &self.params.exclude_domains {
            if !exclude_domains.is_empty() {
                let exclude_sites = exclude_domains.iter().map(|d| format!("-site:{}", d)).collect::<Vec<_>>().join(" ");
                let new_query = format!("{} {}", self.params.query, exclude_sites);
                query_params.insert("q".to_string(), new_query);
            }
        }

        // Add query parameters to URL
        for (key, value) in query_params {
            url.query_pairs_mut().append_pair(&key, &value);
        }

        Ok(url)
    }

    async fn perform_search(&self) -> Result<BingSearchResponse, SearchError> {
        let url = self.build_search_url()?;
        
        let response = self.client
            .get(url)
            .header("Ocp-Apim-Subscription-Key", &self.api_key)
            .send()
            .await
            .map_err(|e| SearchError::BackendError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(SearchError::BackendError(format!("HTTP {}: {}", status, error_text)));
        }

        let search_response: BingSearchResponse = response
            .json()
            .await
            .map_err(|e| SearchError::BackendError(format!("Failed to parse response: {}", e)))?;

        Ok(search_response)
    }
}

impl GuestSearchSession for BingWebSearchClient {
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
            if let Some(web_pages) = &response.web_pages {
                let mut state = self.state.lock().unwrap();
                state.total_results = Some(web_pages.total_estimated_matches);
                state.current_offset += web_pages.value.len() as u32;
            }
            let results = response.web_pages
                .map(|wp| wp.value)
                .unwrap_or_default()
                .into_iter()
                .map(super::conversions::convert_bing_result)
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
        })
    }
}

pub trait ExtendedSearchSession: GuestSearchSession {
    fn new_durable(params: SearchParams) -> Self;
    fn next_page_durable(&self) -> Result<Vec<SearchResult>, SearchError>;
    fn get_metadata_durable(&self) -> Option<SearchMetadata>;
}

impl ExtendedSearchSession for BingWebSearchClient {
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
pub struct BingSearchResponse {
    #[serde(rename = "webPages")]
    pub web_pages: Option<WebPages>,
    #[serde(rename = "error")]
    pub api_error: Option<BingApiError>,
}

#[derive(Debug, Deserialize)]
pub struct WebPages {
    #[serde(rename = "totalEstimatedMatches")]
    pub total_estimated_matches: u64,
    pub value: Vec<BingSearchItem>,
}

#[derive(Debug, Deserialize)]
pub struct BingSearchItem {
    pub name: String,
    pub url: String,
    pub snippet: String,
    #[serde(rename = "displayUrl")]
    pub display_url: Option<String>,
    #[serde(rename = "datePublished")]
    pub date_published: Option<String>,
    pub images: Option<Vec<BingImage>>,
}

#[derive(Debug, Deserialize)]
pub struct BingImage {
    pub url: String,
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BingApiError {
    pub code: String,
    pub message: String,
} 