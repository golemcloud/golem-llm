use crate::golem::search::types::SearchError;
use std::env;

pub fn with_config_key<T, F, R>(key: &str, error_fn: F, func: impl FnOnce(String) -> R) -> R
where
    F: FnOnce(SearchError) -> T,
    R: From<T>,
{
    match env::var(key) {
        Ok(value) if !value.is_empty() => func(value),
        _ => {
            let error = SearchError::Internal(format!("Missing env var: {}", key));
            R::from(error_fn(error))
        }
    }
}

pub fn get_endpoint() -> Option<String> {
    env::var("SEARCH_PROVIDER_ENDPOINT").ok()
}

pub fn get_timeout_ms() -> u32 {
    env::var("SEARCH_PROVIDER_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30000)
}

pub fn get_max_retries() -> u32 {
    env::var("SEARCH_PROVIDER_MAX_RETRIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
}
