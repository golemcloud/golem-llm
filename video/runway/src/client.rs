use golem_video::error::{from_reqwest_error, video_error_from_status};
use golem_video::exports::golem::video::video::VideoError;
use log::trace;
use reqwest::{Client, Method, Response};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://api.dev.runwayml.com";
const API_VERSION: &str = "2024-11-06";

/// The Runway API client for image-to-video generation
pub struct RunwayApi {
    api_key: String,
    client: Client,
}

impl RunwayApi {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .default_headers(reqwest::header::HeaderMap::new())
            .build()
            .expect("Failed to initialize HTTP client");
        Self { api_key, client }
    }

    pub fn generate_video(
        &self,
        request: ImageToVideoRequest,
    ) -> Result<GenerationResponse, VideoError> {
        trace!("Sending image-to-video request to Runway API");

        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v1/image_to_video"))
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .header("X-Runway-Version", API_VERSION)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn poll_generation(&self, task_id: &str) -> Result<PollResponse, VideoError> {
        trace!("Polling generation status for ID: {}", task_id);

        let response: Response = self
            .client
            .request(Method::GET, format!("{BASE_URL}/v1/tasks/{}", task_id))
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .header("X-Runway-Version", API_VERSION)
            .send()
            .map_err(|err| from_reqwest_error("Poll request failed", err))?;

        let status = response.status();

        if status.is_success() {
            let task_response: TaskResponse = response
                .json()
                .map_err(|err| from_reqwest_error("Failed to parse task response", err))?;

            match task_response.status.as_str() {
                "PENDING" | "RUNNING" => Ok(PollResponse::Processing),
                "SUCCEEDED" => {
                    if let Some(output) = task_response.output {
                        if let Some(video_url) = output.first() {
                            // Download the video from the URL
                            let video_data = self.download_video(video_url)?;
                            Ok(PollResponse::Complete {
                                video_data,
                                mime_type: "video/mp4".to_string(),
                            })
                        } else {
                            Err(VideoError::InternalError(
                                "No output URL in successful task".to_string(),
                            ))
                        }
                    } else {
                        Err(VideoError::InternalError(
                            "No output in successful task".to_string(),
                        ))
                    }
                }
                "FAILED" | "CANCELED" => Err(VideoError::GenerationFailed(
                    "Task failed or was canceled".to_string(),
                )),
                _ => Err(VideoError::InternalError(format!(
                    "Unknown task status: {}",
                    task_response.status
                ))),
            }
        } else {
            let error_body = response
                .text()
                .map_err(|err| from_reqwest_error("Failed to read error response", err))?;

            Err(video_error_from_status(status, error_body))
        }
    }

    pub fn cancel_task(&self, task_id: &str) -> Result<(), VideoError> {
        trace!("Canceling task: {}", task_id);

        let response: Response = self
            .client
            .request(Method::DELETE, format!("{BASE_URL}/v1/tasks/{}", task_id))
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .header("X-Runway-Version", API_VERSION)
            .send()
            .map_err(|err| from_reqwest_error("Cancel request failed", err))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let error_body = response
                .text()
                .map_err(|err| from_reqwest_error("Failed to read error response", err))?;

            Err(video_error_from_status(status, error_body))
        }
    }

    fn download_video(&self, url: &str) -> Result<Vec<u8>, VideoError> {
        trace!("Downloading video from URL: {}", url);

        let response: Response = self
            .client
            .get(url)
            .send()
            .map_err(|err| from_reqwest_error("Failed to download video", err))?;

        if !response.status().is_success() {
            return Err(VideoError::InternalError(format!(
                "Failed to download video: HTTP {}",
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .map_err(|err| from_reqwest_error("Failed to read video data", err))?;

        Ok(bytes.to_vec())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageToVideoRequest {
    #[serde(rename = "promptImage")]
    pub prompt_image: Vec<PromptImage>,
    pub model: String,
    pub ratio: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(rename = "promptText", skip_serializing_if = "Option::is_none")]
    pub prompt_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u32>,
    #[serde(rename = "contentModeration", skip_serializing_if = "Option::is_none")]
    pub content_moderation: Option<ContentModeration>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PromptImage {
    pub uri: String,
    pub position: String, // "first" or "last"
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentModeration {
    #[serde(rename = "publicFigureThreshold")]
    pub public_figure_threshold: String, // "auto" or "low"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub id: String,
}

#[derive(Debug, Clone)]
pub enum PollResponse {
    Processing,
    Complete {
        video_data: Vec<u8>,
        mime_type: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResponse {
    pub id: String,
    pub status: String,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    pub output: Option<Vec<String>>,
}

fn parse_response<T: serde::de::DeserializeOwned>(response: Response) -> Result<T, VideoError> {
    let status = response.status();
    if status.is_success() {
        response
            .json::<T>()
            .map_err(|err| from_reqwest_error("Failed to decode response body", err))
    } else {
        let error_body = response
            .text()
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

        let error_message = format!("Request failed with {}: {}", status, error_body);
        Err(video_error_from_status(status, error_message))
    }
}
