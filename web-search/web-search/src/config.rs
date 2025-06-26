use crate::exports::golem::web_search::web_search::SearchError;
use std::{collections::HashMap, ffi::OsStr};

/// Gets an expected configuration value from the environment, and fails if its is not found
/// using the `fail` function. Otherwise, it runs `succeed` with the configuration value.
pub fn with_config_key<R>(
    keys: &[impl AsRef<OsStr>],
    fail: impl FnOnce(SearchError) -> R,
    succeed: impl FnOnce(HashMap<String, String>) -> R,
) -> R {
    let mut hashmap = HashMap::new();
    for key in keys {
        match std::env::var(key.as_ref()) {
            Ok(value) => {
                hashmap.insert(key.as_ref().to_string_lossy().to_string(), value);
            }
            Err(_) => {
                let error = SearchError::BackendError(format!(
                    "Missing config key: {}",
                    key.as_ref().to_string_lossy()
                ));
                return fail(error);
            }
        }
    }
    succeed(hashmap)
}
