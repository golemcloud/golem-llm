use crate::golem::llm::llm::{Error, ErrorCode};
use std::ffi::OsStr;

/// Gets an expected configuration value from the environment, and fails if its is not found
/// using the `fail` function. Otherwise, it runs `succeed` with the configuration value.
pub fn with_config_key<R>(
    key: impl AsRef<OsStr>,
    fail: impl FnOnce(Error) -> R,
    succeed: impl FnOnce(String) -> R,
) -> R {
    let key_str = key.as_ref().to_string_lossy().to_string();
    match std::env::var(key) {
        Ok(value) => succeed(value),
        Err(_) => {
            let error = Error {
                code: ErrorCode::InternalError,
                message: format!("Missing config key: {key_str}"),
                provider_error_json: None,
            };
            fail(error)
        }
    }
}

pub fn get_config_key(key: impl AsRef<OsStr>) -> Result<String, Error> {
    let key_str = key.as_ref().to_string_lossy().to_string();
    std::env::var(key).map_err(|_| Error {
        code: ErrorCode::InternalError,
        message: format!("Missing config key: {key_str}"),
        provider_error_json: None,
    })
}

pub fn get_config_key_or_none(key: impl AsRef<OsStr>) -> Option<String> {
    std::env::var(key).ok()
}
