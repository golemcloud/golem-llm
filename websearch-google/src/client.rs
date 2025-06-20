use crate::exports::golem::web_search::web_search::GuestSearchSession;
use crate::golem::web_search::types::{
    SafeSearchLevel, SearchError, SearchMetadata, SearchParams, SearchResult, TimeRange,
};
use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex};
use url::Url;

const GOOGLE_SEARCH_API_URL: &str = "https://www.googleapis.com/customsearch/v1";

#[derive(Debug, Clone)]
pub struct GoogleWebSearchClient {
    params: SearchParams,
    client: Client,
    api_key: String,
    search_engine_id: String,
    state: Arc<Mutex<ClientState>>,
}

#[derive(Debug, Clone)]
struct ClientState {
    current_page: u32,
    total_results: Option<u64>,
    search_time: Option<f64>,
}

impl GoogleWebSearchClient {
    pub fn new(params: SearchParams) -> Self {
        let api_key = env::var("GOOGLE_API_KEY").unwrap_or_else(|_| {
            eprintln!("Warning: GOOGLE_API_KEY not set, using dummy key");
            "dummy_key".to_string()
        });

        let search_engine_id = env::var("GOOGLE_SEARCH_ENGINE_ID").unwrap_or_else(|_| {
            eprintln!("Warning: GOOGLE_SEARCH_ENGINE_ID not set, using dummy ID");
            "dummy_id".to_string()
        });

        Self {
            params,
            client: Client::new(),
            api_key,
            search_engine_id,
            state: Arc::new(Mutex::new(ClientState {
                current_page: 1,
                total_results: None,
                search_time: None,
            })),
        }
    }

    pub fn search_once(
        params: SearchParams,
    ) -> Result<(Vec<SearchResult>, Option<SearchMetadata>), SearchError> {
        let client = Self::new(params.clone());
        let results = client.next_page_durable()?;
        let metadata = client.get_metadata_durable();
        Ok((results, metadata))
    }

    fn build_search_url(&self) -> Result<Url, SearchError> {
        let state = self.state.lock().unwrap();
        let current_page = state.current_page;
        drop(state);

        let mut url = Url::parse(GOOGLE_SEARCH_API_URL)
            .map_err(|e| SearchError::BackendError(format!("Invalid URL: {}", e)))?;

        let mut query_params = HashMap::new();
        query_params.insert("key".to_string(), self.api_key.clone());
        query_params.insert("cx".to_string(), self.search_engine_id.clone());
        query_params.insert("q".to_string(), self.params.query.clone());

        // Add pagination
        if current_page > 1 {
            let start_index = ((current_page - 1) * 10) + 1;
            query_params.insert("start".to_string(), start_index.to_string());
        }

        // Add max results (Google API uses 10 as default, max 10)
        let max_results = self.params.max_results.unwrap_or(10).min(10);
        query_params.insert("num".to_string(), max_results.to_string());

        // Add safe search
        if let Some(safe_search) = &self.params.safe_search {
            let safe_search_value = match safe_search {
                SafeSearchLevel::Off => "off",
                SafeSearchLevel::Medium => "medium",
                SafeSearchLevel::High => "high",
            };
            query_params.insert("safe".to_string(), safe_search_value.to_string());
        }

        // Add language
        if let Some(language) = &self.params.language {
            query_params.insert("lr".to_string(), language.clone());
        }

        // Add region
        if let Some(region) = &self.params.region {
            query_params.insert("gl".to_string(), region.clone());
        }

        // Add date range
        if let Some(time_range) = &self.params.time_range {
            let date_range = match time_range {
                TimeRange::Day => "d",
                TimeRange::Week => "w",
                TimeRange::Month => "m",
                TimeRange::Year => "y",
            };
            query_params.insert("dateRestrict".to_string(), date_range.to_string());
        }

        // Add site restrictions
        if let Some(include_domains) = &self.params.include_domains {
            if !include_domains.is_empty() {
                let sites = include_domains.join(" ");
                query_params.insert("siteSearch".to_string(), sites);
                query_params.insert("siteSearchFilter".to_string(), "i".to_string());
            }
        }

        if let Some(exclude_domains) = &self.params.exclude_domains {
            if !exclude_domains.is_empty() {
                let exclude_sites = exclude_domains
                    .iter()
                    .map(|d| format!("-site:{}", d))
                    .collect::<Vec<_>>()
                    .join(" ");
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

    async fn perform_search(&self) -> Result<GoogleSearchResponse, SearchError> {
        let url = self.build_search_url()?;

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| SearchError::BackendError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(SearchError::BackendError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let search_response: GoogleSearchResponse = response
            .json()
            .await
            .map_err(|e| SearchError::BackendError(format!("Failed to parse response: {}", e)))?;

        Ok(search_response)
    }
}

impl GuestSearchSession for GoogleWebSearchClient {
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
            if let Some(search_info) = response.search_information {
                let mut state = self.state.lock().unwrap();
                state.total_results = search_info.total_results.parse().ok();
                state.search_time = Some(search_info.search_time);
                state.current_page += 1;
            }
            let results = response
                .items
                .unwrap_or_default()
                .into_iter()
                .map(super::conversions::convert_google_result)
                .collect();
            Ok(results)
        })
    }

    fn get_metadata(&self) -> Option<SearchMetadata> {
        let state = self.state.lock().unwrap();
        Some(SearchMetadata {
            query: self.params.query.clone(),
            total_results: state.total_results,
            search_time_ms: state.search_time,
            safe_search: self.params.safe_search,
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

impl ExtendedSearchSession for GoogleWebSearchClient {
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
pub struct GoogleSearchResponse {
    pub items: Option<Vec<GoogleSearchItem>>,
    pub search_information: Option<SearchInformation>,
    #[serde(rename = "error")]
    pub api_error: Option<GoogleApiError>,
}

#[derive(Debug, Deserialize)]
pub struct GoogleSearchItem {
    pub title: String,
    pub link: String,
    pub snippet: String,
    #[serde(rename = "displayLink")]
    pub display_link: Option<String>,
    #[serde(rename = "source")]
    pub source: Option<String>,
    #[serde(rename = "htmlSnippet")]
    pub html_snippet: Option<String>,
    #[serde(rename = "pagemap")]
    pub page_map: Option<PageMap>,
}

#[derive(Debug, Deserialize)]
pub struct PageMap {
    #[serde(rename = "metatags")]
    pub meta_tags: Option<Vec<MetaTags>>,
    #[serde(rename = "cse_image")]
    pub cse_image: Option<Vec<CseImage>>,
}

#[derive(Debug, Deserialize)]
pub struct MetaTags {
    #[serde(rename = "article:published_time")]
    pub published_time: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CseImage {
    pub src: String,
    #[serde(rename = "alt")]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SearchInformation {
    #[serde(rename = "totalResults")]
    pub total_results: String,
    #[serde(rename = "searchTime")]
    pub search_time: f64,
}

#[derive(Debug, Deserialize)]
pub struct GoogleApiError {
    pub code: u32,
    pub message: String,
}
