wit_bindgen::generate!({
    path: "wit",
    world: "web-search-library",
    additional_derives: [PartialEq, golem_rust::FromValueAndType, golem_rust::IntoValue],
    pub_export_macro: true,
    with: {
        "golem:web-search/types@1.0.0": generate,
        "golem:web-search/web-search@1.0.0": generate,
    },
});

pub mod client;
pub mod conversions;

// Re-export the generated types for external use
pub use crate::exports::golem::web_search::web_search;
pub use crate::golem::web_search::types;

use crate::client::GoogleWebSearchClient;
use crate::exports::golem::web_search::web_search::{Guest, SearchSession};

/// The Google web search provider for the Golem Web Search API
pub struct GoogleWebSearchProvider;

impl Guest for GoogleWebSearchProvider {
    type SearchSession = GoogleWebSearchClient;

    fn search_once(
        params: crate::golem::web_search::types::SearchParams,
    ) -> Result<
        (
            Vec<crate::golem::web_search::types::SearchResult>,
            Option<crate::golem::web_search::types::SearchMetadata>,
        ),
        crate::golem::web_search::types::SearchError,
    > {
        GoogleWebSearchClient::search_once(params)
    }

    fn start_search(
        params: crate::golem::web_search::types::SearchParams,
    ) -> Result<SearchSession, crate::golem::web_search::types::SearchError> {
        let client = GoogleWebSearchClient::new(params);
        Ok(SearchSession::new(client))
    }
}

export!(GoogleWebSearchProvider with_types_in crate);
