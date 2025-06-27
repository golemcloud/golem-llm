use crate::client::{ImageToVideoRequest, KlingApi, PollResponse, TextToVideoRequest};
use golem_video::error::{invalid_input, unsupported_feature};
use golem_video::exports::golem::video::video::{
    AspectRatio, GenerationConfig, JobStatus, MediaData, MediaInput, Resolution, Video, VideoError,
    VideoResult,
};
use std::collections::HashMap;

pub fn media_input_to_request(
    input: MediaInput,
    config: GenerationConfig,
) -> Result<(Option<TextToVideoRequest>, Option<ImageToVideoRequest>), VideoError> {
    // Parse provider options
    let options: HashMap<String, String> = config
        .provider_options
        .iter()
        .map(|kv| (kv.key.clone(), kv.value.clone()))
        .collect();

    // Determine model - default to kling-v1, can be overridden
    let model_name = config.model.clone().or_else(|| {
        options
            .get("model")
            .cloned()
            .or_else(|| Some("kling-v1".to_string()))
    });

    // Validate model if provided
    if let Some(ref model) = model_name {
        if !matches!(
            model.as_str(),
            "kling-v1" | "kling-v1-6" | "kling-v2-master" | "kling-v2-1-master"
        ) {
            return Err(invalid_input(
                "Model must be one of: kling-v1, kling-v1-6, kling-v2-master, kling-v2-1-master",
            ));
        }
    }

    // Determine aspect ratio
    let aspect_ratio = determine_aspect_ratio(config.aspect_ratio, config.resolution)?;

    // Duration support - Kling supports 5 and 10 seconds
    let duration = config.duration_seconds.map(|d| {
        if d <= 5.0 {
            "5".to_string()
        } else {
            "10".to_string()
        }
    });

    // Mode support - std or pro
    let mode = options
        .get("mode")
        .cloned()
        .or_else(|| Some("std".to_string()));
    if let Some(ref mode_val) = mode {
        if !matches!(mode_val.as_str(), "std" | "pro") {
            return Err(invalid_input("Mode must be 'std' or 'pro'"));
        }
    }

    // CFG scale support (0.0 to 1.0)
    let cfg_scale = config
        .guidance_scale
        .map(|scale| (scale / 10.0).clamp(0.0, 1.0));

    // Clone negative_prompt before moving values
    let negative_prompt = config.negative_prompt.clone();

    match input {
        MediaInput::Text(prompt) => {
            let request = TextToVideoRequest {
                model_name,
                prompt,
                negative_prompt,
                cfg_scale,
                mode,
                aspect_ratio: Some(aspect_ratio),
                duration,
                callback_url: None,
                external_task_id: None,
            };

            // Log warnings for unsupported options
            log_unsupported_options(&config, &options);

            Ok((Some(request), None))
        }
        MediaInput::Image(ref_image) => {
            let image_data = match ref_image.data {
                MediaData::Url(_url) => {
                    return Err(unsupported_feature(
                        "Image URLs are not supported by Kling API, only base64 data",
                    ));
                }
                MediaData::Bytes(bytes) => {
                    // Convert bytes to base64 string
                    use base64::Engine;
                    base64::engine::general_purpose::STANDARD.encode(&bytes)
                }
            };

            // Use prompt from the reference image, or default
            let prompt = ref_image
                .prompt
                .clone()
                .unwrap_or_else(|| "Generate a video from this image".to_string());

            let request = ImageToVideoRequest {
                model_name,
                prompt,
                negative_prompt,
                cfg_scale,
                mode,
                aspect_ratio: Some(aspect_ratio),
                duration,
                image: Some(image_data),
                callback_url: None,
                external_task_id: None,
            };

            // Log warnings for unsupported options
            log_unsupported_options(&config, &options);

            Ok((None, Some(request)))
        }
    }
}

fn determine_aspect_ratio(
    aspect_ratio: Option<AspectRatio>,
    _resolution: Option<Resolution>,
) -> Result<String, VideoError> {
    let target_aspect = aspect_ratio.unwrap_or(AspectRatio::Landscape);

    match target_aspect {
        AspectRatio::Landscape => Ok("16:9".to_string()),
        AspectRatio::Portrait => Ok("9:16".to_string()),
        AspectRatio::Square => Ok("1:1".to_string()),
        AspectRatio::Cinema => {
            log::warn!("Cinema aspect ratio not directly supported, using 16:9");
            Ok("16:9".to_string())
        }
    }
}

fn log_unsupported_options(config: &GenerationConfig, options: &HashMap<String, String>) {
    if config.scheduler.is_some() {
        log::warn!("scheduler is not supported by Kling API and will be ignored");
    }
    if config.enable_audio.is_some() {
        log::warn!("enable_audio is not supported by Kling API and will be ignored");
    }
    if config.enhance_prompt.is_some() {
        log::warn!("enhance_prompt is not supported by Kling API and will be ignored");
    }

    // Log unused provider options
    for key in options.keys() {
        if !matches!(key.as_str(), "model" | "mode") {
            log::warn!("Provider option '{key}' is not supported by Kling API");
        }
    }
}

pub fn generate_video(
    client: &KlingApi,
    input: MediaInput,
    config: GenerationConfig,
) -> Result<String, VideoError> {
    let (text_request, image_request) = media_input_to_request(input, config)?;

    if let Some(request) = text_request {
        let response = client.generate_text_to_video(request)?;
        if response.code == 0 {
            Ok(response.data.task_id)
        } else {
            Err(VideoError::GenerationFailed(format!(
                "API error {}: {}",
                response.code, response.message
            )))
        }
    } else if let Some(request) = image_request {
        let response = client.generate_image_to_video(request)?;
        if response.code == 0 {
            Ok(response.data.task_id)
        } else {
            Err(VideoError::GenerationFailed(format!(
                "API error {}: {}",
                response.code, response.message
            )))
        }
    } else {
        Err(VideoError::InternalError(
            "No valid request generated".to_string(),
        ))
    }
}

pub fn poll_video_generation(
    client: &KlingApi,
    task_id: String,
) -> Result<VideoResult, VideoError> {
    match client.poll_generation(&task_id) {
        Ok(PollResponse::Processing) => Ok(VideoResult {
            status: JobStatus::Running,
            videos: None,
            metadata: None,
        }),
        Ok(PollResponse::Complete {
            video_data,
            mime_type,
            duration,
        }) => {
            // Parse duration to extract seconds if possible
            let duration_seconds = parse_duration_string(&duration);

            let video = Video {
                uri: None,
                base64_bytes: Some(video_data),
                mime_type,
                width: None,
                height: None,
                fps: None,
                duration_seconds,
            };

            Ok(VideoResult {
                status: JobStatus::Succeeded,
                videos: Some(vec![video]),
                metadata: None,
            })
        }
        Err(error) => Err(error),
    }
}

fn parse_duration_string(duration_str: &str) -> Option<f32> {
    // Try to parse duration string like "5" or "10" to float
    duration_str.parse::<f32>().ok()
}

pub fn cancel_video_generation(_client: &KlingApi, task_id: String) -> Result<String, VideoError> {
    // Kling API does not support cancellation according to requirements
    Err(VideoError::UnsupportedFeature(format!(
        "Cancellation is not supported by Kling API for task {task_id}"
    )))
}
