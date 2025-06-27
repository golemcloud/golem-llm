use crate::client::{ImageToVideoRequest, PollResponse, StabilityApi};
use golem_video::error::{invalid_input, unsupported_feature};
use golem_video::exports::golem::video::video::{
    GenerationConfig, JobStatus, MediaData, MediaInput, Video, VideoError, VideoResult,
};
use golem_video::utils::download_image_from_url;
use std::collections::HashMap;

pub fn media_input_to_request(
    input: MediaInput,
    config: GenerationConfig,
) -> Result<ImageToVideoRequest, VideoError> {
    match input {
        MediaInput::Text(_) => Err(unsupported_feature(
            "Text-to-video is not supported by Stability API",
        )),
        MediaInput::Image(ref_image) => {
            let image_data = match ref_image.data {
                MediaData::Url(url) => {
                    // Download the image from the URL and convert to bytes
                    download_image_from_url(&url)?
                }
                MediaData::Bytes(bytes) => bytes,
            };

            // Note: Stability doesn't support prompts with images, so we ignore ref_image.prompt

            // Parse provider options - only for parameters not directly supported in WIT
            let options: HashMap<String, String> = config
                .provider_options
                .into_iter()
                .map(|kv| (kv.key, kv.value))
                .collect();

            // Use built-in config fields directly
            let seed = config.seed;
            let cfg_scale = config.guidance_scale;

            // motion_bucket_id is only available via provider options since it's Stability-specific
            let motion_bucket_id = options
                .get("motion_bucket_id")
                .and_then(|s| s.parse::<u32>().ok());

            // Validate parameter ranges according to Stability API
            if let Some(seed_val) = seed {
                if seed_val > 4294967294 {
                    return Err(invalid_input("Seed must be between 0 and 4294967294"));
                }
            }

            if let Some(cfg_val) = cfg_scale {
                if !(0.0..=10.0).contains(&cfg_val) {
                    return Err(invalid_input(
                        "CFG scale (guidance_scale) must be between 0.0 and 10.0",
                    ));
                }
            }

            if let Some(bucket_val) = motion_bucket_id {
                if !(1..=255).contains(&bucket_val) {
                    return Err(invalid_input("Motion bucket ID must be between 1 and 255"));
                }
            }

            // Log warnings for unsupported built-in options
            if config.model.is_some() {
                log::warn!("model is not supported by Stability API and will be ignored");
            }
            if config.negative_prompt.is_some() {
                log::warn!("negative_prompt is not supported by Stability API and will be ignored");
            }
            if config.scheduler.is_some() {
                log::warn!("scheduler is not supported by Stability API and will be ignored");
            }
            if config.aspect_ratio.is_some() {
                log::warn!(
                    "aspect_ratio is supported directly by input image dimensions, 1:1, 16:9, 9:16"
                );
            }
            if config.duration_seconds.is_some() {
                log::warn!(
                    "duration_seconds is not supported by Stability API and will be ignored"
                );
            }
            if config.resolution.is_some() {
                log::warn!("resolution is supported directly by input image dimensions, 1024x576, 576x1024, 768x768");
            }
            if config.enable_audio.is_some() {
                log::warn!("enable_audio is not supported by Stability API and will be ignored");
            }
            if config.enhance_prompt.is_some() {
                log::warn!("enhance_prompt is not supported by Stability API and will be ignored");
            }

            Ok(ImageToVideoRequest {
                image_data,
                seed,
                cfg_scale,
                motion_bucket_id,
            })
        }
    }
}

pub fn generate_video(
    client: &StabilityApi,
    input: MediaInput,
    config: GenerationConfig,
) -> Result<String, VideoError> {
    let request = media_input_to_request(input, config)?;
    let response = client.generate_video(request)?;

    // Return the task ID directly from Stability API
    Ok(response.id)
}

pub fn poll_video_generation(
    client: &StabilityApi,
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

pub fn cancel_video_generation(_task_id: String) -> Result<String, VideoError> {
    Err(unsupported_feature(
        "Video generation cancellation is not supported by Stability API",
    ))
}
