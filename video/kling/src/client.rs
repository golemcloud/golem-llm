use crate::authentication::generate_jwt_token;
use golem_video::error::{from_reqwest_error, video_error_from_status};
use golem_video::exports::golem::video::video::VideoError;
use log::trace;
use reqwest::{Client, Method, Response};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://api-singapore.klingai.com";

/// The Kling API client for video generation
pub struct KlingApi {
    access_key: String,
    secret_key: String,
    client: Client,
}

impl KlingApi {
    pub fn new(access_key: String, secret_key: String) -> Self {
        let client = Client::builder()
            .default_headers(reqwest::header::HeaderMap::new())
            .build()
            .expect("Failed to initialize HTTP client");
        Self {
            access_key,
            secret_key,
            client,
        }
    }

    fn get_auth_header(&self) -> Result<String, VideoError> {
        let token = generate_jwt_token(&self.access_key, &self.secret_key).map_err(|e| {
            VideoError::InternalError(format!("JWT token generation failed: {}", e))
        })?;
        Ok(format!("Bearer {}", token))
    }

    pub fn generate_text_to_video(
        &self,
        request: TextToVideoRequest,
    ) -> Result<GenerationResponse, VideoError> {
        trace!("Sending text-to-video request to Kling API");

        let auth_header = self.get_auth_header()?;

        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v1/videos/text2video"))
            .header("Authorization", auth_header)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn generate_image_to_video(
        &self,
        request: ImageToVideoRequest,
    ) -> Result<GenerationResponse, VideoError> {
        trace!("Sending image-to-video request to Kling API");

        let auth_header = self.get_auth_header()?;

        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v1/videos/image2video"))
            .header("Authorization", auth_header)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn poll_generation(&self, task_id: &str) -> Result<PollResponse, VideoError> {
        trace!("Polling generation status for ID: {}", task_id);

        let auth_header = self.get_auth_header()?;

        let response: Response = self
            .client
            .request(
                Method::GET,
                format!("{BASE_URL}/v1/videos/text2video/{}", task_id),
            )
            .header("Authorization", auth_header)
            .send()
            .map_err(|err| from_reqwest_error("Poll request failed", err))?;

        let status = response.status();

        if status.is_success() {
            let task_response: TaskResponse = response
                .json()
                .map_err(|err| from_reqwest_error("Failed to parse task response", err))?;

            if task_response.code != 0 {
                return Err(VideoError::GenerationFailed(format!(
                    "API error {}: {}",
                    task_response.code, task_response.message
                )));
            }

            match task_response.data.task_status.as_str() {
                "submitted" | "processing" => Ok(PollResponse::Processing),
                "succeed" => {
                    if let Some(task_result) = task_response.data.task_result {
                        if let Some(videos) = task_result.videos {
                            if let Some(video) = videos.first() {
                                // Download the video from the URL
                                let video_data = self.download_video(&video.url)?;
                                Ok(PollResponse::Complete {
                                    video_data,
                                    mime_type: "video/mp4".to_string(),
                                    duration: video.duration.clone(),
                                })
                            } else {
                                Err(VideoError::InternalError(
                                    "No video in successful task".to_string(),
                                ))
                            }
                        } else {
                            Err(VideoError::InternalError(
                                "No videos in successful task".to_string(),
                            ))
                        }
                    } else {
                        Err(VideoError::InternalError(
                            "No task result in successful task".to_string(),
                        ))
                    }
                }
                "failed" => {
                    let error_msg = task_response
                        .data
                        .task_status_msg
                        .unwrap_or_else(|| "Task failed".to_string());
                    Err(VideoError::GenerationFailed(error_msg))
                }
                _ => Err(VideoError::InternalError(format!(
                    "Unknown task status: {}",
                    task_response.data.task_status
                ))),
            }
        } else {
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
pub struct TextToVideoRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_task_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageToVideoRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>, // Base64 encoded image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_task_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub code: i32,
    pub message: String,
    pub request_id: String,
    pub data: GenerationResponseData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResponseData {
    pub task_id: String,
    pub task_status: String,
    pub task_info: TaskInfo,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_task_id: Option<String>,
}

#[derive(Debug, Clone)]
pub enum PollResponse {
    Processing,
    Complete {
        video_data: Vec<u8>,
        mime_type: String,
        duration: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResponse {
    pub code: i32,
    pub message: String,
    pub request_id: String,
    pub data: TaskResponseData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResponseData {
    pub task_id: String,
    pub task_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_status_msg: Option<String>,
    pub task_info: TaskInfo,
    pub created_at: u64,
    pub updated_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_result: Option<TaskResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub videos: Option<Vec<VideoResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoResult {
    pub id: String,
    pub url: String,
    pub duration: String,
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
