wit_bindgen::generate!({
    path: "../wit/golem-web-search",
    world: "web-search-library",
    generate_unused_types: true,
    additional_derives: [PartialEq, golem_rust::FromValueAndType, golem_rust::IntoValue],
    pub_export_macro: true,
});

pub mod client;
pub mod conversions;

// Re-export the generated types for external use
pub use crate::exports::golem::web_search::web_search;
pub use crate::golem::web_search::types;

#[macro_export]
macro_rules! export_web_search_google {
    () => {
        const _: () => {
            use $crate::exports::golem::web_search::web_search::{Guest as WitGuest};
            use $crate::client::GoogleWebSearchClient;

            /// The Google web search provider for the Golem Web Search API
            pub struct GoogleWebSearchProvider;

            impl WitGuest for GoogleWebSearchProvider {
                type SearchSession = GoogleWebSearchClient;

                fn search_once(
                    params: $crate::golem::web_search::types::SearchParams,
                ) -> Result<(Vec<$crate::golem::web_search::types::SearchResult>, Option<$crate::golem::web_search::types::SearchMetadata>), $crate::golem::web_search::types::SearchError> {
                    GoogleWebSearchClient::search_once(params)
                }

                fn start_search(
                    params: $crate::golem::web_search::types::SearchParams,
                ) -> Result<GoogleWebSearchClient, $crate::golem::web_search::types::SearchError> {
                    Ok(GoogleWebSearchClient::new(params))
                }
            }

            #[allow(dead_code)]
            fn __export_google_web_search_provider() {
                export!(GoogleWebSearchProvider with_types_in $crate);
            }
        };
    };
}
