use crate::golem::embed::embed::{Error, ErrorCode};
use std::ffi::OsStr;

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
                code: ErrorCode::AuthenticationFailed,
                message: format!("Missing config key: {key_str}"),
                provider_error_json: None,
            };
            fail(error)
        }
    }
}
