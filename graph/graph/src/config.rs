use crate::golem::graph::connection::ConnectionConfig;
use std::env;

/// Retrieves a configuration value from an environment variable, checking the provider_config first.
pub fn with_config_key(config: &ConnectionConfig, key: &str) -> Option<String> {
    config
        .provider_config
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.clone())
        .or_else(|| env::var(key).ok())
}
