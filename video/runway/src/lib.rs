mod client;
mod conversion;

use crate::client::RunwayApi;
use crate::conversion::{cancel_video_generation, generate_video, poll_video_generation};
use golem_video::config::with_config_key;
use golem_video::durability::{DurableVideo, ExtendedGuest};
use golem_video::exports::golem::video::video::{GenerationConfig, MediaInput, VideoError};
use golem_video::exports::golem::video::video::{Guest, VideoResult};
use golem_video::LOGGING_STATE;

struct RunwayComponent;

impl RunwayComponent {
    const ENV_VAR_NAME: &'static str = "RUNWAY_API_KEY";
}

impl Guest for RunwayComponent {
    fn generate(input: MediaInput, config: GenerationConfig) -> Result<String, VideoError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        with_config_key(
            Self::ENV_VAR_NAME,
            |err| {
                // Return the error from the config lookup
                Err(err)
            },
            |api_key| {
                let client = RunwayApi::new(api_key);
                generate_video(&client, input, config)
            },
        )
    }

    fn poll(job_id: String) -> Result<VideoResult, VideoError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        with_config_key(Self::ENV_VAR_NAME, Err, |api_key| {
            let client = RunwayApi::new(api_key);
            poll_video_generation(&client, job_id)
        })
    }

    fn cancel(job_id: String) -> Result<String, VideoError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        with_config_key(Self::ENV_VAR_NAME, Err, |api_key| {
            let client = RunwayApi::new(api_key);
            cancel_video_generation(&client, job_id)
        })
    }
}

impl ExtendedGuest for RunwayComponent {}

type DurableRunwayComponent = DurableVideo<RunwayComponent>;

golem_video::export_video!(DurableRunwayComponent with_types_in golem_video);
