#[allow(static_mut_refs)]
mod bindings;

use crate::bindings::exports::test::video_exports::test_video_api::*;
use crate::bindings::golem::video::video;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::thread;
use std::time::Duration;

struct Component;

fn save_video_result(video_result: &video::VideoResult, job_id: &str) -> String {
    if let Some(videos) = &video_result.videos {
        for (i, video_data) in videos.iter().enumerate() {
            let filename = format!("/output/video-{}.mp4", i);
            
            // Create output directory if it doesn't exist
            if let Err(err) = fs::create_dir_all("/output") {
                return format!("Failed to create output directory: {}", err);
            }
            
            // Save the video data
            match &video_data.base64_bytes {
                Some(video_bytes) => {
                    match fs::write(&filename, video_bytes) {
                        Ok(_) => {
                            return filename;
                        }
                        Err(err) => {
                            return format!("Failed to save video to {}: {}", filename, err);
                        }
                    }
                }
                None => {
                    if let Some(uri) = &video_data.uri {
                        return format!("Video available at URI: {}", uri);
                    } else {
                        return "No video data or URI available".to_string();
                    }
                }
            }
        }
        "No videos in result".to_string()
    } else {
        "No videos in result".to_string()
    }
}

///job_id to test stability: 939104de411db613f610b6193259df171e7a5bbd555db55f2310009ad06bfae
///because stability polling fails

/// kling job_id 767103052582096949
/// incase

//// google projects/golem-test-463802/locations/us-central1/publishers/google/models/veo-2.0-generate-001/operations/6013adea-df6a-465a-ae73-21dbf73a0b1f
impl Guest for Component {
    /// test1 demonstrates a simple video generation using a binary image input.
    fn test1() -> String {
        // VEO image-to-video job_id for testing: 6013adea-df6a-465a-ae73-21dbf73a0b1f
        // let job_id = "projects/golem-test-463802/locations/us-central1/publishers/google/models/veo-2.0-generate-001/operations/6013adea-df6a-465a-ae73-21dbf73a0b1f".to_string();
        
        println!("Reading image from Initial File System...");
        let mut file = match File::open("/data/old.png") {
            Ok(file) => file,
            Err(err) => return format!("ERROR: Failed to open old.png: {}", err),
        };

        let mut buffer = Vec::new();
        match file.read_to_end(&mut buffer) {
            Ok(_) => println!("Successfully read {} bytes from old.png", buffer.len()),
            Err(err) => return format!("ERROR: Failed to read old.png: {}", err),
        }

        // Create video generation configuration
        let config = video::GenerationConfig {
            negative_prompt: None,
            seed: None,
            scheduler: None,
            guidance_scale: None,
            aspect_ratio: None, // Will be determined by input image dimensions
            model: None,
            duration_seconds: None,
            resolution: None, // Will be determined by input image dimensions  
            enable_audio: Some(false),
            enhance_prompt: Some(false),
            provider_options: vec![
                video::Kv {
                    key: "motion_bucket_id".to_string(),
                    value: "127".to_string(),
                }
            ],
        };

        // Create media input with the image data
        let media_input = video::MediaInput::Image(video::ReferenceImage {
            data: video::MediaData::Bytes(buffer),
            prompt: Some("An Old smiling man, and waving his hand".to_string()),
        });

        println!("Sending video generation request...");
        let job_id = match video::generate(&media_input, &config) {
            Ok(id) => {
                println!("Generated job ID: '{}'", id);
                println!("Job ID length: {}", id.len());
                println!("Job ID bytes: {:?}", id.as_bytes());
                id.trim().to_string() // Trim whitespace to fix stringification issues
            }
            Err(error) => {
                return format!("ERROR: Failed to generate video: {:?}", error);
            }
        };

        // Wait 5 seconds after job creation before starting polling
        println!("Waiting 5 seconds for job initialization...");
        thread::sleep(Duration::from_secs(5));

        println!("Polling for video generation results with job ID: {}", job_id);

        // Poll every 9 seconds until completion
        loop {
            match video::poll(&job_id) {
                Ok(video_result) => {
                    match video_result.status {
                        video::JobStatus::Pending => {
                            println!("Video generation is pending...");
                        }
                        video::JobStatus::Running => {
                            println!("Video generation is running...");
                        }
                        video::JobStatus::Succeeded => {
                            println!("Video generation completed successfully!");
                            let file_path = save_video_result(&video_result, &job_id);
                            return format!("Video generated successfully. Saved to: {}", file_path);
                        }
                        video::JobStatus::Failed(error_msg) => {
                            return format!("Video generation failed: {}", error_msg);
                        }
                    }
                }
                Err(error) => {
                    return format!("Error polling video generation: {:?}", error);
                }
            }
            
            // Wait 9 seconds before polling again
            thread::sleep(Duration::from_secs(9));
        }
    }

    /// test2 demonstrates text-to-video generation with a creative prompt.
    fn test2() -> String {
        println!("Starting text-to-video generation...");

        // VEO text-to-video job_id for testing: a8b50c2f-3726-48f7-9d81-6f4c6038e1e7
        // let job_id = "projects/golem-test-463802/locations/us-central1/publishers/google/models/veo-2.0-generate-001/operations/a8b50c2f-3726-48f7-9d81-6f4c6038e1e7".to_string();
        
        // Create video generation configuration
        let config = video::GenerationConfig {
            negative_prompt: Some("blurry, low quality, distorted, ugly".to_string()),
            seed: None,
            scheduler: None,
            guidance_scale: None,
            aspect_ratio: None,
            model: None,
            duration_seconds: Some(5.0),
            resolution: None,
            enable_audio: Some(false),
            enhance_prompt: Some(true),
            provider_options: vec![
                video::Kv {
                    key: "mode".to_string(),
                    value: "std".to_string(),
                }
            ],
        };

        // Create text prompt for video generation
        let creative_prompt = "Create a joyful, cartoon-style scene of a playful snow leopard cub with big, expressive eyes prancing through a whimsical winter forest. The cub leaps over snowdrifts, chases falling snowflakes, and slides playfully down a hill. Use bright, cheerful colors, rounded trees dusted with snow, and gentle sunlight filtering through the branches for a heartwarming, upbeat mood".to_string();
        
        let media_input = video::MediaInput::Text(creative_prompt.clone());

        println!("Sending text-to-video generation request with prompt: {}", creative_prompt);
        let job_id = match video::generate(&media_input, &config) {
            Ok(id) => {
                println!("Generated job ID: '{}'", id);
                println!("Job ID length: {}", id.len());
                println!("Job ID bytes: {:?}", id.as_bytes());
                id.trim().to_string() // Trim whitespace to fix stringification issues
            }
            Err(error) => {
                return format!("ERROR: Failed to generate video: {:?}", error);
            }
        };
        
        // Wait 5 seconds after job creation before starting polling
        println!("Waiting 5 seconds for job initialization...");
        thread::sleep(Duration::from_secs(5));

        // let job_id = "projects/golem-test-463802/locations/us-central1/publishers/google/models/veo-2.0-generate-001/operations/8dae1743-e1b7-4f38-b3da-af7feff2e8ca".to_string();
        // let job_id = "projects/golem-test-463802/locations/us-central1/publishers/google/models/veo-2.0-generate-001/operations/8ff4ffc3-3885-4d4c-ac53-5f8dc0672620".to_string();
        println!("Polling for video generation results with job ID: {}", job_id);

        // Poll every 9 seconds until completion
        loop {
            match video::poll(&job_id) {
                Ok(video_result) => {
                    match video_result.status {
                        video::JobStatus::Pending => {
                            println!("Text-to-video generation is pending...");
                        }
                        video::JobStatus::Running => {
                            println!("Text-to-video generation is running...");
                        }
                        video::JobStatus::Succeeded => {
                            println!("Text-to-video generation completed successfully!");
                            let file_path = save_video_result(&video_result, &job_id);
                            return format!("Text-to-video generated successfully. Saved to: {}", file_path);
                        }
                        video::JobStatus::Failed(error_msg) => {
                            return format!("Text-to-video generation failed: {}", error_msg);
                        }
                    }
                }
                Err(error) => {
                    return format!("Error polling text-to-video generation: {:?}", error);
                }
            }
            
            // Wait 9 seconds before polling again
            thread::sleep(Duration::from_secs(9));
        }
    }
}

bindings::export!(Component with_types_in bindings);