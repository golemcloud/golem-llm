mod authentication;
mod client;
mod conversion;

use crate::client::VeoApi;
use crate::conversion::{cancel_video_generation, generate_video, poll_video_generation};
use golem_video::config::with_config_key;
use golem_video::durability::{DurableVideo, ExtendedGuest};
use golem_video::exports::golem::video::video::{GenerationConfig, MediaInput, VideoError};
use golem_video::exports::golem::video::video::{Guest, VideoResult};
use golem_video::LOGGING_STATE;

struct VeoComponent;

impl VeoComponent {
    const PROJECT_ID_ENV_VAR: &'static str = "VEO_PROJECT_ID";
    const CLIENT_EMAIL_ENV_VAR: &'static str = "VEO_CLIENT_EMAIL";
    const PRIVATE_KEY_ENV_VAR: &'static str = "VEO_PRIVATE_KEY";
}

impl Guest for VeoComponent {
    fn generate(input: MediaInput, config: GenerationConfig) -> Result<String, VideoError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::PROJECT_ID_ENV_VAR, Err, |project_id| {
            with_config_key(Self::CLIENT_EMAIL_ENV_VAR, Err, |client_email| {
                with_config_key(Self::PRIVATE_KEY_ENV_VAR, Err, |private_key| {
                    let client = VeoApi::new(project_id, client_email, private_key);
                    generate_video(&client, input, config)
                })
            })
        })
    }

    fn poll(job_id: String) -> Result<VideoResult, VideoError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::PROJECT_ID_ENV_VAR, Err, |project_id| {
            with_config_key(Self::CLIENT_EMAIL_ENV_VAR, Err, |client_email| {
                with_config_key(Self::PRIVATE_KEY_ENV_VAR, Err, |private_key| {
                    let client = VeoApi::new(project_id, client_email, private_key);
                    poll_video_generation(&client, job_id)
                })
            })
        })
    }

    fn cancel(job_id: String) -> Result<String, VideoError> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::PROJECT_ID_ENV_VAR, Err, |project_id| {
            with_config_key(Self::CLIENT_EMAIL_ENV_VAR, Err, |client_email| {
                with_config_key(Self::PRIVATE_KEY_ENV_VAR, Err, |private_key| {
                    let client = VeoApi::new(project_id, client_email, private_key);
                    cancel_video_generation(&client, job_id)
                })
            })
        })
    }
}

impl ExtendedGuest for VeoComponent {}

type DurableVeoComponent = DurableVideo<VeoComponent>;

golem_video::export_video!(DurableVeoComponent with_types_in golem_video);
