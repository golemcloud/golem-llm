#[cfg(test)]
use crate::client::{GoogleQueries, GoogleQueryInfo, GoogleSearchInformation, GoogleSearchItem};
use crate::client::{GoogleSearchRequest, GoogleSearchResponse};
use golem_web_search::golem::web_search::types::{
    SafeSearchLevel, SearchMetadata, SearchParams, SearchResult, TimeRange,
};

pub fn convert_params_to_request(
    params: &SearchParams,
    start_index: Option<u32>,
) -> GoogleSearchRequest {
    let mut request = GoogleSearchRequest {
        q: params.query.clone(),
        cx: String::new(),
        key: String::new(),
        num: params.max_results,
        start: start_index,
        safe: params.safe_search.as_ref().map(|s| match s {
            SafeSearchLevel::Off => "off".to_string(),
            SafeSearchLevel::Medium => "medium".to_string(),
            SafeSearchLevel::High => "high".to_string(),
        }),
        lr: params
            .language
            .as_ref()
            .map(|lang| format!("lang_{}", lang)),
        gl: params.region.clone(),
        date_restrict: params.time_range.as_ref().map(|tr| match tr {
            TimeRange::Day => "d1".to_string(),
            TimeRange::Week => "w1".to_string(),
            TimeRange::Month => "m1".to_string(),
            TimeRange::Year => "y1".to_string(),
        }),
        site_search: None,
        site_search_filter: None,
    };

    if let Some(include_domains) = &params.include_domains {
        if !include_domains.is_empty() {
            request.site_search = Some(
                include_domains
                    .iter()
                    .map(|domain| format!("site:{}", domain))
                    .collect::<Vec<_>>()
                    .join(" OR "),
            );
            request.site_search_filter = Some("i".to_string());
        }
    } else if let Some(exclude_domains) = &params.exclude_domains {
        if !exclude_domains.is_empty() {
            request.site_search = Some(
                exclude_domains
                    .iter()
                    .map(|domain| format!("site:{}", domain))
                    .collect::<Vec<_>>()
                    .join(" OR "),
            );
            request.site_search_filter = Some("e".to_string());
        }
    }

    request
}

pub fn convert_response_to_results(
    response: GoogleSearchResponse,
    params: &SearchParams,
) -> (Vec<SearchResult>, Option<SearchMetadata>) {
    let results = if let Some(items) = response.items {
        items
            .into_iter()
            .map(|item| SearchResult {
                title: item.title,
                url: item.link,
                snippet: item.snippet,
                display_url: item.display_link,
                source: Some("Google".to_string()),
                score: None,
                html_snippet: if params.include_html.unwrap_or(false) {
                    item.html_snippet
                } else {
                    None
                },
                date_published: None,
                images: None,
                content_chunks: None,
            })
            .collect()
    } else {
        Vec::new()
    };

    let metadata = SearchMetadata {
        query: params.query.clone(),
        total_results: response
            .search_information
            .as_ref()
            .and_then(|info| info.total_results.as_ref())
            .and_then(|s| s.parse::<u64>().ok()),
        search_time_ms: response
            .search_information
            .as_ref()
            .and_then(|info| info.search_time)
            .map(|t| t * 1000.0),
        safe_search: params.safe_search,
        language: params.language.clone(),
        region: params.region.clone(),
        next_page_token: response
            .queries
            .as_ref()
            .and_then(|q| q.next_page.as_ref())
            .and_then(|np| np.first())
            .and_then(|np| np.start_index)
            .map(|idx| idx.to_string()),
        rate_limits: None,
    };

    (results, Some(metadata))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_params() -> SearchParams {
        SearchParams {
            query: "test query".to_string(),
            safe_search: Some(SafeSearchLevel::Medium),
            language: Some("en".to_string()),
            region: Some("us".to_string()),
            max_results: Some(10),
            time_range: Some(TimeRange::Week),
            include_domains: Some(vec!["example.com".to_string(), "test.org".to_string()]),
            exclude_domains: None,
            include_images: Some(true),
            include_html: Some(true),
            advanced_answer: Some(false),
        }
    }

    fn create_test_response() -> GoogleSearchResponse {
        GoogleSearchResponse {
            items: Some(vec![
                GoogleSearchItem {
                    title: "Test Result 1".to_string(),
                    link: "https://example.com/1".to_string(),
                    snippet: "This is a test snippet 1".to_string(),
                    display_link: Some("example.com".to_string()),
                    html_snippet: Some("<b>Test</b> snippet 1".to_string()),
                    pagemap: None,
                },
                GoogleSearchItem {
                    title: "Test Result 2".to_string(),
                    link: "https://test.org/2".to_string(),
                    snippet: "This is a test snippet 2".to_string(),
                    display_link: Some("test.org".to_string()),
                    html_snippet: Some("<b>Test</b> snippet 2".to_string()),
                    pagemap: None,
                },
            ]),
            search_information: Some(GoogleSearchInformation {
                total_results: Some("1000".to_string()),
                search_time: Some(0.15),
            }),
            queries: Some(GoogleQueries {
                request: None,
                next_page: Some(vec![GoogleQueryInfo {
                    title: "Next Page".to_string(),
                    total_results: Some("1000".to_string()),
                    search_terms: "test query".to_string(),
                    count: Some(10),
                    start_index: Some(11),
                }]),
            }),
            error: None,
        }
    }

    #[test]
    fn test_convert_params_to_request_basic() {
        let params = SearchParams {
            query: "basic test".to_string(),
            safe_search: None,
            language: None,
            region: None,
            max_results: None,
            time_range: None,
            include_domains: None,
            exclude_domains: None,
            include_images: None,
            include_html: None,
            advanced_answer: None,
        };

        let request = convert_params_to_request(&params, None);

        assert_eq!(request.q, "basic test");
        assert_eq!(request.cx, "");
        assert_eq!(request.key, "");
        assert_eq!(request.num, None);
        assert_eq!(request.start, None);
        assert_eq!(request.safe, None);
        assert_eq!(request.lr, None);
        assert_eq!(request.gl, None);
        assert_eq!(request.date_restrict, None);
        assert_eq!(request.site_search, None);
        assert_eq!(request.site_search_filter, None);
    }

    #[test]
    fn test_convert_params_to_request_full() {
        let params = create_test_params();
        let request = convert_params_to_request(&params, Some(21));

        assert_eq!(request.q, "test query");
        assert_eq!(request.num, Some(10));
        assert_eq!(request.start, Some(21));
        assert_eq!(request.safe, Some("medium".to_string()));
        assert_eq!(request.lr, Some("lang_en".to_string()));
        assert_eq!(request.gl, Some("us".to_string()));
        assert_eq!(request.date_restrict, Some("w1".to_string()));
        assert_eq!(
            request.site_search,
            Some("site:example.com OR site:test.org".to_string())
        );
        assert_eq!(request.site_search_filter, Some("i".to_string()));
    }

    #[test]
    fn test_convert_params_safe_search_levels() {
        let test_cases = vec![
            (SafeSearchLevel::Off, "off"),
            (SafeSearchLevel::Medium, "medium"),
            (SafeSearchLevel::High, "high"),
        ];

        for (level, expected) in test_cases {
            let params = SearchParams {
                query: "test".to_string(),
                safe_search: Some(level),
                language: None,
                region: None,
                max_results: None,
                time_range: None,
                include_domains: None,
                exclude_domains: None,
                include_images: None,
                include_html: None,
                advanced_answer: None,
            };

            let request = convert_params_to_request(&params, None);
            assert_eq!(request.safe, Some(expected.to_string()));
        }
    }

    #[test]
    fn test_convert_params_time_ranges() {
        let test_cases = vec![
            (TimeRange::Day, "d1"),
            (TimeRange::Week, "w1"),
            (TimeRange::Month, "m1"),
            (TimeRange::Year, "y1"),
        ];

        for (range, expected) in test_cases {
            let params = SearchParams {
                query: "test".to_string(),
                safe_search: None,
                language: None,
                region: None,
                max_results: None,
                time_range: Some(range),
                include_domains: None,
                exclude_domains: None,
                include_images: None,
                include_html: None,
                advanced_answer: None,
            };

            let request = convert_params_to_request(&params, None);
            assert_eq!(request.date_restrict, Some(expected.to_string()));
        }
    }

    #[test]
    fn test_convert_params_exclude_domains() {
        let params = SearchParams {
            query: "test".to_string(),
            safe_search: None,
            language: None,
            region: None,
            max_results: None,
            time_range: None,
            include_domains: None,
            exclude_domains: Some(vec!["spam.com".to_string(), "bad.org".to_string()]),
            include_images: None,
            include_html: None,
            advanced_answer: None,
        };

        let request = convert_params_to_request(&params, None);
        assert_eq!(
            request.site_search,
            Some("site:spam.com OR site:bad.org".to_string())
        );
        assert_eq!(request.site_search_filter, Some("e".to_string()));
    }

    #[test]
    fn test_convert_response_to_results_basic() {
        let params = create_test_params();
        let response = create_test_response();

        let (results, metadata) = convert_response_to_results(response, &params);

        assert_eq!(results.len(), 2);

        assert_eq!(results[0].title, "Test Result 1");
        assert_eq!(results[0].url, "https://example.com/1");
        assert_eq!(results[0].snippet, "This is a test snippet 1");
        assert_eq!(results[0].display_url, Some("example.com".to_string()));
        assert_eq!(results[0].source, Some("Google".to_string()));
        assert_eq!(results[0].score, None);
        assert_eq!(
            results[0].html_snippet,
            Some("<b>Test</b> snippet 1".to_string())
        );
        assert_eq!(results[0].date_published, None);
        assert_eq!(results[0].images, None);
        assert_eq!(results[0].content_chunks, None);

        assert_eq!(results[1].title, "Test Result 2");
        assert_eq!(results[1].url, "https://test.org/2");
        assert_eq!(results[1].snippet, "This is a test snippet 2");
        assert_eq!(results[1].display_url, Some("test.org".to_string()));

        assert!(metadata.is_some());
        let meta = metadata.unwrap();
        assert_eq!(meta.query, "test query");
        assert_eq!(meta.total_results, Some(1000));
        assert_eq!(meta.search_time_ms, Some(150.0)); // 0.15 * 1000
        assert_eq!(meta.safe_search, Some(SafeSearchLevel::Medium));
        assert_eq!(meta.language, Some("en".to_string()));
        assert_eq!(meta.region, Some("us".to_string()));
        assert_eq!(meta.next_page_token, Some("11".to_string()));
        assert_eq!(meta.rate_limits, None);
    }

    #[test]
    fn test_convert_response_to_results_no_html() {
        let mut params = create_test_params();
        params.include_html = Some(false);
        let response = create_test_response();

        let (results, _) = convert_response_to_results(response, &params);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].html_snippet, None);
        assert_eq!(results[1].html_snippet, None);
    }

    #[test]
    fn test_convert_response_to_results_empty() {
        let params = create_test_params();
        let response = GoogleSearchResponse {
            items: None,
            search_information: None,
            queries: None,
            error: None,
        };

        let (results, metadata) = convert_response_to_results(response, &params);

        assert_eq!(results.len(), 0);
        assert!(metadata.is_some());
        let meta = metadata.unwrap();
        assert_eq!(meta.query, "test query");
        assert_eq!(meta.total_results, None);
        assert_eq!(meta.search_time_ms, None);
        assert_eq!(meta.next_page_token, None);
    }

    #[test]
    fn test_convert_response_malformed_total_results() {
        let params = create_test_params();
        let mut response = create_test_response();
        response.search_information = Some(GoogleSearchInformation {
            total_results: Some("not_a_number".to_string()),
            search_time: Some(0.25),
        });

        let (_, metadata) = convert_response_to_results(response, &params);

        let meta = metadata.unwrap();
        assert_eq!(meta.total_results, None);
        assert_eq!(meta.search_time_ms, Some(250.0));
    }
}
