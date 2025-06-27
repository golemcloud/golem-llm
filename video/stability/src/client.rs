use golem_video::error::{from_reqwest_error, video_error_from_status};
use golem_video::exports::golem::video::video::VideoError;
use log::trace;
use reqwest::{Client, Method, Response};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://api.stability.ai";

/// The Stability API client for image-to-video generation
pub struct StabilityApi {
    api_key: String,
    client: Client,
}

impl StabilityApi {
    pub fn new(api_key: String) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("accept", "video/*".parse().expect("Invalid header value"));

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .expect("Failed to initialize HTTP client");
        Self { api_key, client }
    }

    pub fn generate_video(
        &self,
        request: ImageToVideoRequest,
    ) -> Result<GenerationResponse, VideoError> {
        trace!("Sending image-to-video request to Stability API");

        // Manually construct multipart/form-data body
        let boundary = generate_boundary();
        let body = build_multipart_body(&request, &boundary);

        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v2beta/image-to-video"))
            .header("authorization", format!("Bearer {}", &self.api_key))
            .header(
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            )
            .body(body)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn poll_generation(&self, generation_id: &str) -> Result<PollResponse, VideoError> {
        trace!("Polling generation status for ID: {}", generation_id);

        let response: Response = self
            .client
            .request(
                Method::GET,
                format!("{BASE_URL}/v2beta/image-to-video/result/{}", generation_id),
            )
            .header("authorization", format!("Bearer {}", &self.api_key))
            .send()
            .map_err(|err| from_reqwest_error("Poll request failed", err))?;

        let status = response.status();

        if status == reqwest::StatusCode::ACCEPTED {
            // 202 - Still processing
            Ok(PollResponse::Processing)
        } else if status.is_success() {
            // 200 - Complete, get video data
            let video_bytes = response
                .bytes()
                .map_err(|err| from_reqwest_error("Failed to read video data", err))?;

            Ok(PollResponse::Complete {
                video_data: video_bytes.to_vec(),
                mime_type: "video/mp4".to_string(),
            })
        } else {
            // Error response
            let error_body = response
                .text()
                .map_err(|err| from_reqwest_error("Failed to read error response", err))?;

            // Try to parse as JSON error, otherwise use raw text
            let error_message =
                if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&error_body) {
                    error_response.error.message
                } else {
                    error_body
                };

            Err(video_error_from_status(status, error_message))
        }
    }
}

#[derive(Debug, Clone)]
// aspect ratio and resolution are supported directly by input image dimensions
// aspect ratio: 1:1, 16:9, 9:16
// resolution: 1024x576, 576x1024, 768x768
pub struct ImageToVideoRequest {
    pub image_data: Vec<u8>,
    pub seed: Option<u64>,
    pub cfg_scale: Option<f32>,
    pub motion_bucket_id: Option<u32>,
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
pub struct ErrorResponse {
    pub error: ErrorDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetails {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
}

fn generate_boundary() -> String {
    // Generate a simple boundary using a timestamp-based approach
    // Since we can't use rand in WASM, we'll use a deterministic approach
    format!(
        "----formdata-golem-{}",
        std::env::var("GOLEM_WORKER_NAME").unwrap_or_else(|_| "stability".to_string())
    )
}

fn build_multipart_body(request: &ImageToVideoRequest, boundary: &str) -> Vec<u8> {
    let mut body = Vec::new();

    // Add image field
    body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"image\"; filename=\"image.png\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body.extend_from_slice(&request.image_data);
    body.extend_from_slice(b"\r\n");

    // Add optional fields
    if let Some(seed) = request.seed {
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"seed\"\r\n\r\n");
        body.extend_from_slice(seed.to_string().as_bytes());
        body.extend_from_slice(b"\r\n");
    }

    if let Some(cfg_scale) = request.cfg_scale {
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"cfg_scale\"\r\n\r\n");
        body.extend_from_slice(cfg_scale.to_string().as_bytes());
        body.extend_from_slice(b"\r\n");
    }

    if let Some(motion_bucket_id) = request.motion_bucket_id {
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"motion_bucket_id\"\r\n\r\n",
        );
        body.extend_from_slice(motion_bucket_id.to_string().as_bytes());
        body.extend_from_slice(b"\r\n");
    }

    // Close boundary
    body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());

    body
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

        let error_message =
            if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&error_body) {
                error_response.error.message
            } else {
                format!("Request failed with {}: {}", status, error_body)
            };

        Err(video_error_from_status(status, error_message))
    }
}
