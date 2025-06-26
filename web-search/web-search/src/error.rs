use crate::exports::golem::web_search::web_search::SearchError;

/// Creates an `Error` value representing that something is unsuported
pub fn unsupported(what: impl AsRef<str>) -> SearchError {
    SearchError::UnsupportedFeature(format!("Unsupported: {}", what.as_ref()))
}

pub fn from_reqwest_error(details: impl AsRef<str>, err: reqwest::Error) -> SearchError {
    SearchError::BackendError(format!("{}: {err}", details.as_ref()))
}
