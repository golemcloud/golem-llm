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
macro_rules! export_web_search_bing {
    () => {
        const _: () => {
            use crate::exports::golem::web_search::web_search::{Guest as WitGuest};
            use crate::client::BingWebSearchClient;

            /// The Bing web search provider for the Golem Web Search API
            pub struct BingWebSearchProvider;

            impl WitGuest for BingWebSearchProvider {
                type SearchSession = BingWebSearchClient;

                fn search_once(
                    params: crate::golem::web_search::types::SearchParams,
                ) -> Result<(Vec<crate::golem::web_search::types::SearchResult>, Option<crate::golem::web_search::types::SearchMetadata>), crate::golem::web_search::types::SearchError> {
                    BingWebSearchClient::search_once(params)
                }

                fn start_search(
                    params: crate::golem::web_search::types::SearchParams,
                ) -> Result<BingWebSearchClient, crate::golem::web_search::types::SearchError> {
                    Ok(BingWebSearchClient::new(params))
                }
            }

            #[allow(dead_code)]
            fn __export_bing_web_search_provider() {
                export!(BingWebSearchProvider with_types_in crate);
            }
        };
    };
} 