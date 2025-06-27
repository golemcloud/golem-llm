use crate::authentication::generate_access_token;
use golem_video::error::{from_reqwest_error, video_error_from_status};
use golem_video::exports::golem::video::video::VideoError;
use log::trace;
use reqwest::{Client, Method, Response};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://us-central1-aiplatform.googleapis.com/v1";
const SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// The Veo API client for video generation
pub struct VeoApi {
    project_id: String,
    client_email: String,
    private_key: String,
    client: Client,
}

impl VeoApi {
    pub fn new(project_id: String, client_email: String, private_key: String) -> Self {
        let client = Client::builder()
            .default_headers(reqwest::header::HeaderMap::new())
            .build()
            .expect("Failed to initialize HTTP client");
        Self {
            project_id,
            client_email,
            private_key,
            client,
        }
    }

    fn get_auth_header(&self) -> Result<String, VideoError> {
        let token = generate_access_token(&self.client_email, &self.private_key, SCOPE)?;
        Ok(format!("Bearer {}", token))
    }

    pub fn generate_text_to_video(
        &self,
        request: TextToVideoRequest,
        model_id: Option<String>,
    ) -> Result<GenerationResponse, VideoError> {
        trace!("Sending text-to-video request to Veo API");

        let auth_header = self.get_auth_header()?;
        let model = model_id.as_deref().unwrap_or("veo-2.0-generate-001");

        let response: Response = self
            .client
            .request(
                Method::POST,
                format!(
                    "{}/projects/{}/locations/us-central1/publishers/google/models/{}:predictLongRunning",
                    BASE_URL, self.project_id, model
                )
            )
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
        model_id: Option<String>,
    ) -> Result<GenerationResponse, VideoError> {
        trace!("Sending image-to-video request to Veo API");

        let auth_header = self.get_auth_header()?;
        let model = model_id.as_deref().unwrap_or("veo-2.0-generate-001");

        let response: Response = self
            .client
            .request(
                Method::POST,
                format!(
                    "{}/projects/{}/locations/us-central1/publishers/google/models/{}:predictLongRunning",
                    BASE_URL, self.project_id, model
                )
            )
            .header("Authorization", auth_header)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn poll_generation(&self, operation_name: &str) -> Result<PollResponse, VideoError> {
        trace!("Polling Veo API for operation: {}", operation_name);

        let auth_header = self.get_auth_header()?;

        let poll_request = PollRequest {
            operation_name: operation_name.to_string(),
        };

        // The operation_name is in format: projects/.../locations/.../publishers/google/models/MODEL_ID/operations/OPERATION_ID
        // We need to extract the base path and construct the fetchPredictOperation endpoint
        // Note: We split on the original string since "/operations/" is a known delimiter
        let parts: Vec<&str> = operation_name.split("/operations/").collect();
        if parts.len() != 2 {
            trace!(
                "Invalid operation name format - parts count: {}, operation_name: {}",
                parts.len(),
                operation_name
            );
            return Err(VideoError::InternalError(format!(
                "Invalid operation name format: {}",
                operation_name
            )));
        }

        let base_path = parts[0]; // projects/.../locations/.../publishers/google/models/MODEL_ID
        let fetch_url = format!(
            "https://us-central1-aiplatform.googleapis.com/v1/{}:fetchPredictOperation",
            base_path
        );

        trace!("Constructed fetch URL: {}", fetch_url);
        trace!("Poll request payload: {:?}", poll_request);

        let response: Response = self
            .client
            .request(Method::POST, fetch_url)
            .header("Authorization", auth_header)
            .header("Content-Type", "application/json")
            .json(&poll_request)
            .send()
            .map_err(|err| from_reqwest_error("Poll request failed", err))?;

        let status = response.status();
        trace!("Received response with status: {}", status);

        if status.is_success() {
            // Get the raw response text first for debugging
            let response_text = response
                .text()
                .map_err(|err| from_reqwest_error("Failed to read response text", err))?;

            trace!("Raw response JSON: {}", response_text);

            // Try to parse the JSON
            let operation_response: OperationResponse = serde_json::from_str(&response_text)
                .map_err(|err| {
                    trace!("JSON parsing failed with error: {}", err);
                    trace!("Failed to parse this JSON: {}", response_text);
                    VideoError::InternalError(format!(
                        "Failed to parse operation response: {}",
                        err
                    ))
                })?;

            trace!(
                "Successfully parsed OperationResponse: {:?}",
                operation_response
            );

            // When the operation is still processing, the API may not include the 'done' field
            // If 'done' is missing, we treat it as false (still processing)
            let is_done = operation_response.done.unwrap_or(false);
            trace!("Operation done status: {}", is_done);

            if !is_done {
                trace!("Operation still processing, returning Processing status");
                return Ok(PollResponse::Processing);
            }

            trace!("Operation completed, checking for response data");

            if let Some(response) = operation_response.response {
                trace!("Found response data: {:?}", response);

                if let Some(videos) = response.videos {
                    trace!("Found {} videos in response", videos.len());

                    let video_results: Result<Vec<_>, VideoError> = videos
                        .into_iter()
                        .enumerate()
                        .map(|(index, video)| {
                            trace!("Processing video {}: {:?}", index, video);

                            if let Some(base64_data) = video.bytes_base64_encoded {
                                trace!(
                                    "Video {} has base64 data, length: {}",
                                    index,
                                    base64_data.len()
                                );

                                // Decode base64 video data
                                let video_data = base64::Engine::decode(
                                    &base64::engine::general_purpose::STANDARD,
                                    &base64_data,
                                )
                                .map_err(|e| {
                                    trace!("Failed to decode base64 for video {}: {}", index, e);
                                    VideoError::InternalError(format!(
                                        "Failed to decode base64 video data: {}",
                                        e
                                    ))
                                })?;

                                let mime_type =
                                    video.mime_type.unwrap_or_else(|| "video/mp4".to_string());

                                trace!(
                                    "Successfully decoded video {}, data length: {}, mime_type: {}",
                                    index,
                                    video_data.len(),
                                    mime_type
                                );

                                Ok(VideoResultData {
                                    video_data,
                                    mime_type,
                                })
                            } else {
                                trace!("Video {} has no base64 encoded data", index);
                                Err(VideoError::InternalError(
                                    "No base64 encoded video data in response".to_string(),
                                ))
                            }
                        })
                        .collect();

                    match video_results {
                        Ok(videos) => {
                            trace!("Successfully processed all {} videos", videos.len());
                            Ok(PollResponse::Complete(videos))
                        }
                        Err(e) => {
                            trace!("Failed to process videos: {:?}", e);
                            Err(e)
                        }
                    }
                } else {
                    trace!("No videos found in successful operation response");
                    Err(VideoError::InternalError(
                        "No videos in successful operation".to_string(),
                    ))
                }
            } else {
                trace!("Operation completed but no response data found");
                Err(VideoError::GenerationFailed(
                    "Operation completed but no response data".to_string(),
                ))
            }
        } else {
            let error_body = response
                .text()
                .map_err(|err| from_reqwest_error("Failed to read error response", err))?;

            trace!(
                "Request failed with status {}, error body: {}",
                status,
                error_body
            );
            Err(video_error_from_status(status, error_body))
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TextToVideoRequest {
    pub instances: Vec<TextToVideoInstance>,
    pub parameters: VideoParameters,
}

#[derive(Debug, Clone, Serialize)]
pub struct TextToVideoInstance {
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageToVideoRequest {
    pub instances: Vec<ImageToVideoInstance>,
    pub parameters: VideoParameters,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageToVideoInstance {
    pub prompt: String,
    pub image: ImageData,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageData {
    #[serde(rename = "bytesBase64Encoded")]
    pub bytes_base64_encoded: String,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct VideoParameters {
    #[serde(rename = "aspectRatio", skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(rename = "durationSeconds")]
    pub duration_seconds: u32,
    #[serde(rename = "enhancePrompt", skip_serializing_if = "Option::is_none")]
    pub enhance_prompt: Option<bool>,
    #[serde(rename = "generateAudio", skip_serializing_if = "Option::is_none")]
    pub generate_audio: Option<bool>,
    #[serde(rename = "negativePrompt", skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(rename = "personGeneration", skip_serializing_if = "Option::is_none")]
    pub person_generation: Option<String>,
    #[serde(rename = "sampleCount", skip_serializing_if = "Option::is_none")]
    pub sample_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(rename = "storageUri", skip_serializing_if = "Option::is_none")]
    pub storage_uri: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PollRequest {
    #[serde(rename = "operationName")]
    pub operation_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub name: String,
}

#[derive(Debug, Clone)]
pub enum PollResponse {
    Processing,
    Complete(Vec<VideoResultData>),
}

#[derive(Debug, Clone)]
pub struct VideoResultData {
    pub video_data: Vec<u8>,
    pub mime_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationResponse {
    pub name: String,
    /// Whether the operation is complete. When still processing, this field may be absent.
    pub done: Option<bool>,
    pub response: Option<VeoResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VeoResponse {
    #[serde(rename = "@type")]
    pub type_field: String,
    pub videos: Option<Vec<VeoVideo>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VeoVideo {
    #[serde(rename = "bytesBase64Encoded")]
    pub bytes_base64_encoded: Option<String>,
    #[serde(rename = "mimeType")]
    pub mime_type: Option<String>,
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
