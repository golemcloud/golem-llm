use golem_llm::error::{error_code_from_status, from_event_source_error, from_reqwest_error};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::Error;
use log::trace;
use reqwest::header::HeaderValue;
use reqwest::{Client, Method, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::str::FromStr;

const BASE_URL: &str = "https://api.x.ai";

/// The Completions API client for creating model responses.
pub struct CompletionsApi {
    api_key: String,
    client: Client,
}

impl CompletionsApi {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self { api_key, client }
    }

    pub fn send_messages(&self, request: CompletionsRequest) -> Result<CompletionsResponse, Error> {
        trace!("Sending request to xAI API: {request:?}");

        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v1/chat/completions"))
            .bearer_auth(self.api_key.clone())
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn stream_send_messages(&self, request: CompletionsRequest) -> Result<EventSource, Error> {
        trace!("Sending request to xAI API: {request:?}");

        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v1/chat/completions"))
            .bearer_auth(self.api_key.clone())
            .header(
                reqwest::header::ACCEPT,
                HeaderValue::from_static("text/event-stream"),
            )
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        trace!("Initializing SSE stream");

        EventSource::new(response)
            .map_err(|err| from_event_source_error("Failed to create SSE stream", err))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionsRequest {
    pub messages: Vec<Message>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Effort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Tool {
    #[serde(rename = "function")]
    Function { function: Function },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System {
        content: Content,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "user")]
    User {
        content: Content,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "assistant")]
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<Content>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    #[serde(rename = "tool")]
    Tool {
        content: Content,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    TextInput(String),
    List(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    TextInput { text: String },
    #[serde(rename = "image_url")]
    ImageInput { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum Detail {
    #[serde(rename = "auto")]
    #[default]
    Auto,
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "high")]
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Effort {
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "high")]
    High,
}

impl FromStr for Effort {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "low" => Ok(Effort::Low),
            "high" => Ok(Effort::High),
            _ => Err(format!("Invalid effort value: {s}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Detail>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolCall {
    #[serde(rename = "function")]
    Function {
        function: FunctionCall,
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<u32>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub arguments: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionsResponse {
    pub choices: Vec<Choice>,
    pub created: u64,
    pub id: String,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub finish_reason: Option<FinishReason>,
    pub index: u32,
    pub message: ResponseMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    #[serde(rename = "stop")]
    Stop,
    #[serde(rename = "length")]
    Length,
    #[serde(rename = "end_turn")]
    EndTurn,
    #[serde(rename = "tool_calls")]
    ToolCalls,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub completion_tokens: u32,
    pub completion_tokens_details: CompletionTokenDetails,
    pub prompt_tokens: u32,
    pub prompt_tokens_details: PromptTokenDetails,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokenDetails {
    pub accepted_prediction_tokens: u32,
    pub audio_tokens: u32,
    pub reasoning_tokens: u32,
    pub rejected_prediction_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokenDetails {
    pub audio_tokens: u32,
    pub cached_tokens: u32,
    pub image_tokens: u32,
    pub text_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChoiceChunk>,
    pub usage: Option<Usage>,
    pub system_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceChunk {
    pub index: u32,
    pub delta: ChoiceDelta,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceDelta {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub role: String,
}

fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();
    if status.is_success() {
        let body = response
            .json::<T>()
            .map_err(|err| from_reqwest_error("Failed to decode response body", err))?;

        trace!("Received response from xAI API: {body:?}");

        Ok(body)
    } else {
        let error_body = response
            .text()
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

        trace!("Received {status} response from xAI API: {error_body:?}");

        Err(Error {
            code: error_code_from_status(status),
            message: format!("Request failed with {status}"),
            provider_error_json: Some(serde_json::to_string(&error_body).unwrap()),
        })
    }
}
