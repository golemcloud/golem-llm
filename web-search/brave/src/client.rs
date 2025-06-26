use golem_web_search::error::from_reqwest_error;
use golem_web_search::golem::web_search::types::SearchError;
use log::trace;
use reqwest::{header, Client, Method, Response, StatusCode};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://api.search.brave.com/res/v1/web/search";

pub struct BraveSearchApi {
    api_key: String,
    client: Client,
}

impl BraveSearchApi {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self { api_key, client }
    }

    pub fn search(&self, request: BraveSearchRequest) -> Result<BraveSearchResponse, SearchError> {
        trace!("Sending request to Brave Search API: {request:?}");

        let response: Response = self
            .client
            .request(Method::GET, BASE_URL)
            .header(header::ACCEPT, "application/json")
            .header(header::ACCEPT_ENCODING, "identity")
            .header(header::USER_AGENT, "golem-websearch/1.0")
            .header("x-subscription-token", &self.api_key)
            .query(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveSearchRequest {
    pub q: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_lang: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ui_lang: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safesearch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freshness: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_decorations: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spellcheck: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_filter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub goggles_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveSearchResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<BraveQuery>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web: Option<BraveWebResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discussions: Option<BraveDiscussions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infobox: Option<BraveInfobox>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub videos: Option<BraveVideos>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mixed: Option<BraveMixed>,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveQuery {
    pub original: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_strict_warning: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_navigational: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_news_breaking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spellcheck_off: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bad_results: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub should_fallback: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postal_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header_country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub more_results_available: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveWebResults {
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<BraveWebResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family_friendly: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveWebResult {
    pub title: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_source_local: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_source_both: Option<bool>,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_age: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<BraveProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family_friendly: Option<bool>,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta_url: Option<BraveMetaUrl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<BraveThumbnail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_snippets: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveProfile {
    pub name: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub img: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveMetaUrl {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheme: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub netloc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub favicon: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveThumbnail {
    pub src: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logo: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveDiscussions {
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<BraveDiscussionResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mutated_by_goggles: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveDiscussionResult {
    pub title: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_source_local: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_source_both: Option<bool>,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family_friendly: Option<bool>,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta_url: Option<BraveMetaUrl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<BraveDiscussionData>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveDiscussionData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forum_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_answers: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub question: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_comment: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveInfobox {
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<BraveInfoboxResult>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveInfoboxResult {
    pub title: String,
    pub url: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_desc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub website_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Vec<Vec<String>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiles: Option<Vec<BraveProfile>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<BraveImage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub providers: Option<Vec<BraveProvider>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveImage {
    pub src: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logo: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveProvider {
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_type: Option<String>,
    pub name: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub img: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveVideos {
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<BraveVideoResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mutated_by_goggles: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveVideoResult {
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_type: Option<String>,
    pub url: String,
    pub title: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_age: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video: Option<BraveVideoInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta_url: Option<BraveMetaUrl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<BraveThumbnail>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveVideoInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub views: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub creator: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub publisher: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveMixed {
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main: Option<Vec<BraveMixedItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top: Option<Vec<BraveMixedItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub side: Option<Vec<BraveMixedItem>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraveMixedItem {
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub all: Option<bool>,
}

fn parse_response(response: Response) -> Result<BraveSearchResponse, SearchError> {
    match response.status() {
        StatusCode::OK => {
            let body = response.text().map_err(|e| {
                SearchError::BackendError(format!("Failed to read response body: {}", e))
            })?;
            match serde_json::from_str::<BraveSearchResponse>(&body) {
                Ok(parsed) => Ok(parsed),
                Err(e) => Err(SearchError::BackendError(format!(
                    "Failed to parse response: {} \nRaw body: {}",
                    e, body
                ))),
            }
        }
        StatusCode::TOO_MANY_REQUESTS => Err(SearchError::RateLimited(60)),
        StatusCode::BAD_REQUEST => Err(SearchError::InvalidQuery),
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
