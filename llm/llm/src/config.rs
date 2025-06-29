use crate::golem::llm::llm::{Error, ErrorCode};
use std::{collections::HashMap, ffi::OsStr};

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

/// Gets multiple expected configuration values from the environment, and fails if any is not found
/// using the `fail` function. Otherwise, it runs `succeed` with all configuration values.
pub fn with_config_keys<R>(
    keys: &[&str],
    fail: impl FnOnce(Error) -> R,
    succeed: impl FnOnce(HashMap<String, String>) -> R,
) -> R {
    let mut values = HashMap::new();
    
    for key in keys {
        match std::env::var(key) {
            Ok(value) => {
                values.insert(key.to_string(), value);
            }
            Err(_) => {
                let error = Error {
                    code: ErrorCode::InternalError,
                    message: format!("Missing config key: {key}"),
                    provider_error_json: None,
                };
                return fail(error);
            }
        }
    }
    
    succeed(values)
}
