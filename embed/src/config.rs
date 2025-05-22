use std::env;

/// Helper function to get a configuration value from an environment variable.
pub fn with_config_key<F, T>(key: &str, f: F) -> Option<T>
where
    F: FnOnce(&str) -> Option<T>,
{
    env::var(key).ok().and_then(|value| f(&value))
}

/// Helper function to get a configuration value from an environment variable with a default.
pub fn with_config_key_or<F, T>(key: &str, default: T, f: F) -> T
where
    F: FnOnce(&str) -> Option<T>,
{
    with_config_key(key, f).unwrap_or(default)
}