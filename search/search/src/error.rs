use crate::golem::search::types::SearchError;

pub fn parse_http_error(status: u16, body: &str) -> SearchError {
    match status {
        400 => SearchError::InvalidQuery(body.to_string()),
        404 => SearchError::IndexNotFound,
        429 => SearchError::RateLimited,
        500..=599 => SearchError::Internal(format!("Server error: {}", body)),
        _ => SearchError::Internal(format!("HTTP error {}: {}", status, body)),
    }
}

pub fn network_error(error: &str) -> SearchError {
    if error.contains("timeout") {
        SearchError::Timeout
    } else {
        SearchError::Internal(format!("Network error: {}", error))
    }
}
