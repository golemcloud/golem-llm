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
pub use crate::golem::web_search::types;
pub use crate::exports::golem::web_search::web_search;

#[macro_export]
macro_rules! export_web_search_brave {
    () => {
        const _: () => {
            use crate::exports::golem::web_search::web_search::{Guest as WitGuest};
            use crate::client::BraveWebSearchClient;

            /// The Brave web search provider for the Golem Web Search API
            pub struct BraveWebSearchProvider;

            impl WitGuest for BraveWebSearchProvider {
                type SearchSession = BraveWebSearchClient;

                fn search_once(
                    params: crate::golem::web_search::types::SearchParams,
                ) -> Result<(Vec<crate::golem::web_search::types::SearchResult>, Option<crate::golem::web_search::types::SearchMetadata>), crate::golem::web_search::types::SearchError> {
                    BraveWebSearchClient::search_once(params)
                }

                fn start_search(
                    params: crate::golem::web_search::types::SearchParams,
                ) -> Result<BraveWebSearchClient, crate::golem::web_search::types::SearchError> {
                    Ok(BraveWebSearchClient::new(params))
                }
            }

            #[allow(dead_code)]
            fn __export_brave_web_search_provider() {
                export!(BraveWebSearchProvider with_types_in crate);
            }
        };
    };
} 