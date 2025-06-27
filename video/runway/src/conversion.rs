use crate::client::{ContentModeration, ImageToVideoRequest, PollResponse, PromptImage, RunwayApi};
use golem_video::error::{invalid_input, unsupported_feature};
use golem_video::exports::golem::video::video::{
    AspectRatio, GenerationConfig, JobStatus, MediaData, MediaInput, Resolution, Video, VideoError,
    VideoResult,
};
use std::collections::HashMap;

pub fn media_input_to_request(
    input: MediaInput,
    config: GenerationConfig,
) -> Result<ImageToVideoRequest, VideoError> {
    match input {
        MediaInput::Text(_) => Err(unsupported_feature(
            "Text-to-video is not supported by Runway API",
        )),
        MediaInput::Image(ref_image) => {
            let image_data = match ref_image.data {
                MediaData::Url(url) => url,
                MediaData::Bytes(bytes) => {
                    // Convert bytes to data URI
                    use base64::Engine;
                    let base64_data = base64::engine::general_purpose::STANDARD.encode(&bytes);
                    format!("data:image/png;base64,{}", base64_data)
                }
            };

            // Parse provider options
            let options: HashMap<String, String> = config
                .provider_options
                .into_iter()
                .map(|kv| (kv.key, kv.value))
                .collect();

            // Determine model - default to gen3a_turbo, can be overridden
            let model = config.model.unwrap_or_else(|| {
                options
                    .get("model")
                    .cloned()
                    .unwrap_or_else(|| "gen3a_turbo".to_string())
            });

            // Validate model
            if !matches!(model.as_str(), "gen3a_turbo" | "gen4_turbo") {
                return Err(invalid_input("Model must be 'gen3a_turbo' or 'gen4_turbo'"));
            }

            // Determine ratio based on aspect_ratio and resolution
            let ratio = determine_ratio(&model, config.aspect_ratio, config.resolution)?;

            // Duration support
            let duration = config.duration_seconds.map(|d| d as u32);
            if let Some(dur) = duration {
                if !(5..=10).contains(&dur) {
                    return Err(invalid_input("Duration must be between 5 and 10 seconds"));
                }
            }

            // Content moderation
            let content_moderation = options.get("publicFigureThreshold").map(|threshold| {
                let threshold_value = if threshold == "low" { "low" } else { "auto" };
                ContentModeration {
                    public_figure_threshold: threshold_value.to_string(),
                }
            });

            // Create prompt image - assuming all images are first frame as requested
            let prompt_image = vec![PromptImage {
                uri: image_data,
                position: "first".to_string(),
            }];

            // Use prompt text from the image if available
            let prompt_text = ref_image.prompt;

            // Validate seed if provided
            if let Some(seed_val) = config.seed {
                if seed_val > 4294967295 {
                    return Err(invalid_input("Seed must be between 0 and 4294967295"));
                }
            }

            // Log warnings for unsupported built-in options
            if config.negative_prompt.is_some() {
                log::warn!("negative_prompt is not supported by Runway API and will be ignored");
            }
            if config.scheduler.is_some() {
                log::warn!("scheduler is not supported by Runway API and will be ignored");
            }
            if config.guidance_scale.is_some() {
                log::warn!("guidance_scale is not supported by Runway API and will be ignored");
            }
            if config.enable_audio.is_some() {
                log::warn!("enable_audio is not supported by Runway API and will be ignored");
            }
            if config.enhance_prompt.is_some() {
                log::warn!("enhance_prompt is not supported by Runway API and will be ignored");
            }

            Ok(ImageToVideoRequest {
                prompt_image,
                model,
                ratio,
                seed: config.seed,
                prompt_text,
                duration,
                content_moderation,
            })
        }
    }
}

fn determine_ratio(
    model: &str,
    aspect_ratio: Option<AspectRatio>,
    _resolution: Option<Resolution>,
) -> Result<String, VideoError> {
    // Default ratios by model
    let default_ratio = match model {
        "gen3a_turbo" => "1280:768",
        "gen4_turbo" => "1280:720",
        _ => return Err(invalid_input("Invalid model")),
    };

    // If no aspect ratio specified, use default
    let target_aspect = aspect_ratio.unwrap_or(AspectRatio::Landscape);

    match model {
        "gen3a_turbo" => match target_aspect {
            AspectRatio::Landscape => Ok("1280:768".to_string()),
            AspectRatio::Portrait => Ok("768:1280".to_string()),
            AspectRatio::Square | AspectRatio::Cinema => {
                log::warn!(
                    "Aspect ratio {:?} not supported by gen3a_turbo, using landscape",
                    target_aspect
                );
                Ok("1280:768".to_string())
            }
        },
        "gen4_turbo" => match target_aspect {
            AspectRatio::Landscape => Ok("1280:720".to_string()),
            AspectRatio::Portrait => Ok("720:1280".to_string()),
            AspectRatio::Square => Ok("960:960".to_string()),
            AspectRatio::Cinema => Ok("1584:672".to_string()),
        },
        _ => Ok(default_ratio.to_string()),
    }
}

pub fn generate_video(
    client: &RunwayApi,
    input: MediaInput,
    config: GenerationConfig,
) -> Result<String, VideoError> {
    let request = media_input_to_request(input, config)?;
    let response = client.generate_video(request)?;

    // Return the task ID directly from Runway API
    Ok(response.id)
}

pub fn poll_video_generation(
    client: &RunwayApi,
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
        }) => {
            let video = Video {
                uri: None,
                base64_bytes: Some(video_data),
                mime_type,
                width: None,
                height: None,
                fps: None,
                duration_seconds: None,
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

pub fn cancel_video_generation(client: &RunwayApi, task_id: String) -> Result<String, VideoError> {
    client.cancel_task(&task_id)?;
    Ok(format!("Task {} canceled successfully", task_id))
}
