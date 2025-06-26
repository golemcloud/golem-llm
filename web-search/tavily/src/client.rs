use golem_web_search::error::from_reqwest_error;
use golem_web_search::golem::web_search::types::SearchError;
use log::trace;
use reqwest::{header, Client, Method, Response, StatusCode};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://api.tavily.com/search";

pub struct TavilySearchApi {
    api_key: String,
    client: Client,
}

impl TavilySearchApi {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self { api_key, client }
    }

    pub fn search(
        &self,
        request: TavilySearchRequest,
    ) -> Result<TavilySearchResponse, SearchError> {
        trace!("Sending request to Tavily Search API: {request:?}");

        let response: Response = self
            .client
            .request(Method::POST, BASE_URL)
            .header(header::ACCEPT, "application/json")
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::USER_AGENT, "golem-websearch/1.0")
            .header(header::AUTHORIZATION, format!("Bearer {}", &self.api_key))
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TavilySearchRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_depth: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_results: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_answer: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_raw_content: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_images: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_range: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub days: Option<u16>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TavilySearchResponse {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    pub results: Vec<TavilySearchResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<TavilyImage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_time: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TavilySearchResult {
    pub title: String,
    pub url: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub published_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum TavilyImage {
    Url(String),
    Object {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TavilyErrorResponse {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: TavilyError,
    pub time: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TavilyError {
    pub id: Option<String>,
    pub status: u16,
    pub code: String,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

fn parse_response(response: Response) -> Result<TavilySearchResponse, SearchError> {
    match response.status() {
        StatusCode::OK => {
            let body = response.text().map_err(|e| {
                SearchError::BackendError(format!("Failed to read response body: {}", e))
            })?;
            match serde_json::from_str::<TavilySearchResponse>(&body) {
                Ok(parsed) => Ok(parsed),
                Err(e) => Err(SearchError::BackendError(format!(
                    "Failed to parse response: {} \nRaw body: {}",
                    e, body
                ))),
            }
        }
        StatusCode::BAD_REQUEST => {
            let _body = response
                .text()
                .unwrap_or_else(|_| "<failed to read body>".into());
            Err(SearchError::InvalidQuery)
        }
        StatusCode::UNAUTHORIZED => Err(SearchError::BackendError("Invalid API key".to_string())),
        StatusCode::TOO_MANY_REQUESTS => Err(SearchError::RateLimited(60)),
        status if status.as_u16() == 432 => {
            Err(SearchError::BackendError("Plan limit exceeded".to_string()))
        }
        _ => {
            let status = response.status();
            let body = response
                .text()
                .unwrap_or_else(|_| "<failed to read body>".into());
            Err(SearchError::BackendError(format!(
                "Request failed: {} \nRaw body: {}",
                status, body
            )))
        }
    }
}
