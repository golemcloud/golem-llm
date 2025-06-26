use golem_web_search::error::from_reqwest_error;
use golem_web_search::golem::web_search::types::SearchError;
use log::trace;
use reqwest::{header, Client, Method, Response, StatusCode};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://google.serper.dev/search";

pub struct SerperSearchApi {
    api_key: String,
    client: Client,
}

impl SerperSearchApi {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self { api_key, client }
    }

    pub fn search(
        &self,
        request: SerperSearchRequest,
    ) -> Result<SerperSearchResponse, SearchError> {
        trace!("Sending request to Serper Search API: {request:?}");

        let response: Response = self
            .client
            .request(Method::POST, BASE_URL)
            .header(header::ACCEPT, "application/json")
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::USER_AGENT, "golem-websearch/1.0")
            .header("X-API-KEY", &self.api_key)
            .json(&[request]) // Serper expects an array of requests
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperSearchRequest {
    pub q: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gl: Option<String>, // Country code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hl: Option<String>, // Language code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num: Option<u32>, // Number of results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub autocorrect: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tbs: Option<String>, // Time-based search filters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperSearchResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "searchParameters")]
    pub search_parameters: Option<SerperSearchParameters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub organic: Option<Vec<SerperOrganicResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "answerBox")]
    pub answer_box: Option<SerperAnswerBox>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "knowledgeGraph")]
    pub knowledge_graph: Option<SerperKnowledgeGraph>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "peopleAlsoAsk")]
    pub people_also_ask: Option<Vec<SerperPeopleAlsoAsk>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "relatedSearches")]
    pub related_searches: Option<Vec<SerperRelatedSearch>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<SerperImage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub videos: Option<Vec<SerperVideo>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub news: Option<Vec<SerperNews>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "searchInformation")]
    pub search_information: Option<SerperSearchInformation>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperSearchParameters {
    pub q: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub autocorrect: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperOrganicResult {
    pub title: String,
    pub link: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sitelinks: Option<Vec<SerperSitelink>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperSitelink {
    pub title: String,
    pub link: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperAnswerBox {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperKnowledgeGraph {
    pub title: String,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kg_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub website: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "imageUrl")]
    pub image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "descriptionSource")]
    pub description_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "descriptionLink")]
    pub description_link: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperPeopleAlsoAsk {
    pub question: String,
    pub snippet: String,
    pub title: String,
    pub link: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperRelatedSearch {
    pub query: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperImage {
    pub title: String,
    #[serde(rename = "imageUrl")]
    pub image_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "imageWidth")]
    pub image_width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "imageHeight")]
    pub image_height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "thumbnailUrl")]
    pub thumbnail_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "thumbnailWidth")]
    pub thumbnail_width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "thumbnailHeight")]
    pub thumbnail_height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperVideo {
    pub title: String,
    pub link: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "imageUrl")]
    pub image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperNews {
    pub title: String,
    pub link: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "imageUrl")]
    pub image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SerperSearchInformation {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "totalResults")]
    pub total_results: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "timeTaken")]
    pub time_taken: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "originalQuery")]
    pub original_query: Option<String>,
}

fn parse_response(response: Response) -> Result<SerperSearchResponse, SearchError> {
    match response.status() {
        StatusCode::OK => {
            let body = response.text().map_err(|e| {
                SearchError::BackendError(format!("Failed to read response body: {}", e))
            })?;

            // Serper returns an array of responses when we send an array, we take the first one
            let parsed_array =
                serde_json::from_str::<Vec<SerperSearchResponse>>(&body).map_err(|e| {
                    SearchError::BackendError(format!(
                        "Failed to parse response as array: {} \nRaw body: {}",
                        e, body
                    ))
                })?;

            parsed_array
                .into_iter()
                .next()
                .ok_or_else(|| SearchError::BackendError("Empty response array".to_string()))
        }
        StatusCode::TOO_MANY_REQUESTS => Err(SearchError::RateLimited(60)),
        StatusCode::BAD_REQUEST => Err(SearchError::InvalidQuery),
        StatusCode::UNAUTHORIZED => Err(SearchError::BackendError("Invalid API key".to_string())),
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
