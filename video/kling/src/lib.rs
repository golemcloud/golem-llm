mod authentication;
mod client;
mod conversion;

use crate::client::KlingApi;
use crate::conversion::{cancel_video_generation, generate_video, poll_video_generation};
use golem_video::config::with_config_key;
use golem_video::durability::{DurableVideo, ExtendedGuest};
use golem_video::exports::golem::video::video::{GenerationConfig, MediaInput, VideoError};
use golem_video::exports::golem::video::video::{Guest, VideoResult};
use golem_video::LOGGING_STATE;

struct KlingComponent;

impl KlingComponent {
    const ACCESS_KEY_ENV_VAR: &'static str = "KLING_ACCESS_KEY";
    const SECRET_KEY_ENV_VAR: &'static str = "KLING_SECRET_KEY";
}

impl Guest for KlingComponent {
    fn generate(input: MediaInput, config: GenerationConfig) -> Result<String, VideoError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ACCESS_KEY_ENV_VAR, Err, |access_key| {
            with_config_key(Self::SECRET_KEY_ENV_VAR, Err, |secret_key| {
                let client = KlingApi::new(access_key, secret_key);
                generate_video(&client, input, config)
            })
        })
    }

    fn poll(job_id: String) -> Result<VideoResult, VideoError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ACCESS_KEY_ENV_VAR, Err, |access_key| {
            with_config_key(Self::SECRET_KEY_ENV_VAR, Err, |secret_key| {
                let client = KlingApi::new(access_key, secret_key);
                poll_video_generation(&client, job_id)
            })
        })
    }

    fn cancel(job_id: String) -> Result<String, VideoError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ACCESS_KEY_ENV_VAR, Err, |access_key| {
            with_config_key(Self::SECRET_KEY_ENV_VAR, Err, |secret_key| {
                let client = KlingApi::new(access_key, secret_key);
                cancel_video_generation(&client, job_id)
            })
        })
    }
}

impl ExtendedGuest for KlingComponent {}

type DurableKlingComponent = DurableVideo<KlingComponent>;

golem_video::export_video!(DurableKlingComponent with_types_in golem_video);
