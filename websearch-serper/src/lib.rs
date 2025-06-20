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
macro_rules! export_web_search_serper {
    () => {
        const _: () => {
            use $crate::exports::golem::web_search::web_search::{Guest as WitGuest};
            use $crate::client::SerperWebSearchClient;

            /// The Serper web search provider for the Golem Web Search API
            pub struct SerperWebSearchProvider;

            impl WitGuest for SerperWebSearchProvider {
                type SearchSession = SerperWebSearchClient;

                fn search_once(
                    params: $crate::golem::web_search::types::SearchParams,
                ) -> Result<(Vec<$crate::golem::web_search::types::SearchResult>, Option<$crate::golem::web_search::types::SearchMetadata>), $crate::golem::web_search::types::SearchError> {
                    SerperWebSearchClient::search_once(params)
                }

                fn start_search(
                    params: $crate::golem::web_search::types::SearchParams,
                ) -> Result<SerperWebSearchClient, $crate::golem::web_search::types::SearchError> {
                    Ok(SerperWebSearchClient::new(params))
                }
            }

            #[allow(dead_code)]
            fn __export_serper_web_search_provider() {
                export!(SerperWebSearchProvider with_types_in $crate);
            }
        };
    };
}
