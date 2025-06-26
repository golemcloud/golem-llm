use golem_web_search::error::from_reqwest_error;
use golem_web_search::golem::web_search::types::SearchError;
use log::trace;
use reqwest::{Client, Method, Response};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://www.googleapis.com/customsearch/v1";

pub struct GoogleSearchApi {
    api_key: String,
    search_engine_id: String,
    client: Client,
}

impl GoogleSearchApi {
    pub fn new(api_key: String, search_engine_id: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self {
            api_key,
            search_engine_id,
            client,
        }
    }

    pub fn search(
        &self,
        mut request: GoogleSearchRequest,
    ) -> Result<GoogleSearchResponse, SearchError> {
        request.key = self.api_key.clone();
        request.cx = self.search_engine_id.clone();

        trace!("Sending request to Google Custom Search API: {request:?}");

        let response: Response = self
            .client
            .request(Method::GET, BASE_URL)
            .query(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GoogleSearchRequest {
    pub q: String,
    pub cx: String,
    pub key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "dateRestrict")]
    pub date_restrict: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "siteSearch")]
    pub site_search: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "siteSearchFilter")]
    pub site_search_filter: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GoogleSearchResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<GoogleSearchItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "searchInformation")]
    pub search_information: Option<GoogleSearchInformation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queries: Option<GoogleQueries>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<GoogleError>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GoogleSearchItem {
    pub title: String,
    pub link: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "displayLink")]
    pub display_link: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "htmlSnippet")]
    pub html_snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "pagemap")]
    pub pagemap: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GoogleSearchInformation {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "totalResults")]
    pub total_results: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "searchTime")]
    pub search_time: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GoogleQueries {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<Vec<GoogleQueryInfo>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "nextPage")]
    pub next_page: Option<Vec<GoogleQueryInfo>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GoogleQueryInfo {
    pub title: String,
    #[serde(rename = "totalResults")]
    pub total_results: Option<String>,
    #[serde(rename = "searchTerms")]
    pub search_terms: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    #[serde(rename = "startIndex")]
    pub start_index: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GoogleError {
    pub code: u32,
    pub message: String,
    pub status: String,
}

fn parse_response(response: Response) -> Result<GoogleSearchResponse, SearchError> {
    let status = response.status();
    if !status.is_success() {
        return Err(SearchError::BackendError(format!(
            "HTTP {}: {}",
            status,
            status.canonical_reason().unwrap_or("Unknown error")
        )));
    }

    let search_response: GoogleSearchResponse = response
        .json()
        .map_err(|e| SearchError::BackendError(format!("JSON parsing failed: {}", e)))?;

    if let Some(error) = &search_response.error {
        return match error.code {
            429 => Err(SearchError::RateLimited(60)),
            400 => Err(SearchError::InvalidQuery),
            _ => Err(SearchError::BackendError(format!(
                "Google API error: {}",
                error.message
            ))),
        };
    }

    Ok(search_response)
}
