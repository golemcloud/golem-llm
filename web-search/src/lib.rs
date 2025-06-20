pub mod durability;
wit_bindgen::generate!({
    path: "../wit/golem-web-search",
    world: "web-search-library",
    generate_unused_types: true,
    additional_derives: [PartialEq, golem_rust::FromValueAndType, golem_rust::IntoValue],
    pub_export_macro: true,
});

#[macro_export]
macro_rules! export_web_search {
    ($provider:ty) => {
        const _: () => {
            use $crate::exports::golem::web_search::web_search::{Guest as WitGuest};

            /// The service provider for the Golem Web Search API
            pub struct GolemWebSearchProvider;

            impl WitGuest for GolemWebSearchProvider {
                type SearchSession = <$provider as WitGuest>::SearchSession;

                fn search_once(
                    params: $crate::golem::web_search::types::SearchParams,
                ) -> Result<(Vec<$crate::golem::web_search::types::SearchResult>, Option<$crate::golem::web_search::types::SearchMetadata>), $crate::golem::web_search::types::SearchError> {
                    <$provider as WitGuest>::search_once(params)
                }
            }

            #[allow(dead_code)]
            fn __export_golem_web_search_provider() {
                export!(GolemWebSearchProvider with_types_in crate);
            }
        };
    };
}
