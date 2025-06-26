#[cfg(test)]
use crate::client::{BraveQuery, BraveWebResult, BraveWebResults};
use crate::client::{BraveSearchRequest, BraveSearchResponse};
use golem_web_search::golem::web_search::types::{
    ImageResult, SafeSearchLevel, SearchMetadata, SearchParams, SearchResult, TimeRange,
};

pub fn convert_params_to_request(params: &SearchParams, offset: Option<u32>) -> BraveSearchRequest {
    let mut request = BraveSearchRequest {
        q: params.query.clone(),
        country: params.region.clone(),
        search_lang: params.language.clone(),
        ui_lang: params.language.clone(),
        count: params.max_results,
        offset,
        safesearch: params.safe_search.as_ref().map(|s| match s {
            SafeSearchLevel::Off => "off".to_string(),
            SafeSearchLevel::Medium => "moderate".to_string(),
            SafeSearchLevel::High => "strict".to_string(),
        }),
        freshness: params.time_range.as_ref().map(|tr| match tr {
            TimeRange::Day => "pd".to_string(),
            TimeRange::Week => "pw".to_string(),
            TimeRange::Month => "pm".to_string(),
            TimeRange::Year => "py".to_string(),
        }),
        text_decorations: Some(false),
        spellcheck: Some(true),
        result_filter: None,
        goggles_id: None,
        units: Some("metric".to_string()),
    };

    if let Some(include_domains) = &params.include_domains {
        if !include_domains.is_empty() {
            let domain_query = include_domains
                .iter()
                .map(|domain| format!("site:{}", domain))
                .collect::<Vec<_>>()
                .join(" OR ");
            request.q = format!("{} ({})", request.q, domain_query);
        }
    } else if let Some(exclude_domains) = &params.exclude_domains {
        if !exclude_domains.is_empty() {
            let domain_query = exclude_domains
                .iter()
                .map(|domain| format!("-site:{}", domain))
                .collect::<Vec<_>>()
                .join(" ");
            request.q = format!("{} {}", request.q, domain_query);
        }
    }

    request
}

pub fn convert_response_to_results(
    response: BraveSearchResponse,
    params: &SearchParams,
) -> (Vec<SearchResult>, Option<SearchMetadata>) {
    let mut results = Vec::new();

    if let Some(web) = &response.web {
        if let Some(web_results) = &web.results {
            for item in web_results {
                results.push(SearchResult {
                    title: item.title.clone(),
                    url: item.url.clone(),
                    snippet: item.description.clone(),
                    display_url: item.meta_url.as_ref().map(|meta| {
                        meta.hostname.clone().unwrap_or_else(|| {
                            item.profile
                                .as_ref()
                                .map(|p| p.long_name.clone().unwrap_or(p.name.clone()))
                                .unwrap_or(item.url.clone())
                        })
                    }),
                    source: Some("Brave".to_string()),
                    score: None,
                    html_snippet: if params.include_html.unwrap_or(false) {
                        Some(item.description.clone())
                    } else {
                        None
                    },
                    date_published: item.age.clone().or_else(|| item.page_age.clone()),
                    images: item.thumbnail.as_ref().map(|thumb| {
                        vec![ImageResult {
                            url: thumb.src.clone(),
                            description: thumb.original.clone(),
                        }]
                    }),
                    content_chunks: item.extra_snippets.clone(),
                });
            }
        }
    }

    let total_results = response
        .query
        .as_ref()
        .and_then(|q| q.more_results_available)
        .map(|has_more| {
            if has_more {
                1000u64
            } else {
                results.len() as u64
            }
        });

    let next_page_token = response
        .query
        .as_ref()
        .and_then(|q| q.more_results_available)
        .filter(|&has_more| has_more)
        .map(|_| {
            let current_offset = params.max_results.unwrap_or(10);
            (current_offset + params.max_results.unwrap_or(10)).to_string()
        });

    let metadata = SearchMetadata {
        query: params.query.clone(),
        total_results,
        search_time_ms: None,
        safe_search: params.safe_search,
        language: params.language.clone(),
        region: params.region.clone(),
        next_page_token,
        rate_limits: None,
    };

    (results, Some(metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{BraveMetaUrl, BraveProfile, BraveThumbnail};

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

    fn create_test_response() -> BraveSearchResponse {
        BraveSearchResponse {
            query: Some(BraveQuery {
                original: "test query".to_string(),
                show_strict_warning: Some(false),
                is_navigational: Some(false),
                is_news_breaking: Some(false),
                spellcheck_off: Some(false),
                country: Some("us".to_string()),
                bad_results: Some(false),
                should_fallback: Some(false),
                postal_code: None,
                city: None,
                header_country: None,
                more_results_available: Some(true),
                state: None,
            }),
            web: Some(BraveWebResults {
                result_type: Some("search".to_string()),
                results: Some(vec![
                    BraveWebResult {
                        title: "Test Result 1".to_string(),
                        url: "https://example.com/1".to_string(),
                        is_source_local: Some(false),
                        is_source_both: Some(false),
                        description: "This is a test snippet 1".to_string(),
                        page_age: Some("2023-01-15T10:30:00".to_string()),
                        profile: Some(BraveProfile {
                            name: "Example".to_string(),
                            url: "https://example.com".to_string(),
                            long_name: Some("example.com".to_string()),
                            img: None,
                        }),
                        language: Some("en".to_string()),
                        family_friendly: Some(true),
                        result_type: Some("search_result".to_string()),
                        subtype: Some("generic".to_string()),
                        meta_url: Some(BraveMetaUrl {
                            scheme: Some("https".to_string()),
                            netloc: Some("example.com".to_string()),
                            hostname: Some("example.com".to_string()),
                            favicon: None,
                            path: Some("/1".to_string()),
                        }),
                        thumbnail: Some(BraveThumbnail {
                            src: "https://example.com/thumb1.jpg".to_string(),
                            original: Some("https://example.com/orig1.jpg".to_string()),
                            logo: Some(false),
                        }),
                        age: Some("2 days ago".to_string()),
                        extra_snippets: Some(vec!["Extra info 1".to_string()]),
                    },
                    BraveWebResult {
                        title: "Test Result 2".to_string(),
                        url: "https://test.org/2".to_string(),
                        is_source_local: Some(false),
                        is_source_both: Some(false),
                        description: "This is a test snippet 2".to_string(),
                        page_age: None,
                        profile: None,
                        language: Some("en".to_string()),
                        family_friendly: Some(true),
                        result_type: Some("search_result".to_string()),
                        subtype: Some("generic".to_string()),
                        meta_url: Some(BraveMetaUrl {
                            scheme: Some("https".to_string()),
                            netloc: Some("test.org".to_string()),
                            hostname: Some("test.org".to_string()),
                            favicon: None,
                            path: Some("/2".to_string()),
                        }),
                        thumbnail: None,
                        age: None,
                        extra_snippets: None,
                    },
                ]),
                family_friendly: Some(true),
            }),
            discussions: None,
            infobox: None,
            videos: None,
            mixed: None,
            response_type: Some("search".to_string()),
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
        assert_eq!(request.country, None);
        assert_eq!(request.search_lang, None);
        assert_eq!(request.ui_lang, None);
        assert_eq!(request.count, None);
        assert_eq!(request.offset, None);
        assert_eq!(request.safesearch, None);
        assert_eq!(request.freshness, None);
        assert_eq!(request.text_decorations, Some(false));
        assert_eq!(request.spellcheck, Some(true));
        assert_eq!(request.units, Some("metric".to_string()));
    }

    #[test]
    fn test_convert_params_to_request_full() {
        let params = create_test_params();
        let request = convert_params_to_request(&params, Some(20));

        assert_eq!(request.q, "test query (site:example.com OR site:test.org)");
        assert_eq!(request.country, Some("us".to_string()));
        assert_eq!(request.search_lang, Some("en".to_string()));
        assert_eq!(request.ui_lang, Some("en".to_string()));
        assert_eq!(request.count, Some(10));
        assert_eq!(request.offset, Some(20));
        assert_eq!(request.safesearch, Some("moderate".to_string()));
        assert_eq!(request.freshness, Some("pw".to_string()));
    }

    #[test]
    fn test_convert_params_safe_search_levels() {
        let test_cases = vec![
            (SafeSearchLevel::Off, "off"),
            (SafeSearchLevel::Medium, "moderate"),
            (SafeSearchLevel::High, "strict"),
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
            assert_eq!(request.safesearch, Some(expected.to_string()));
        }
    }

    #[test]
    fn test_convert_params_time_ranges() {
        let test_cases = vec![
            (TimeRange::Day, "pd"),
            (TimeRange::Week, "pw"),
            (TimeRange::Month, "pm"),
            (TimeRange::Year, "py"),
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
            assert_eq!(request.freshness, Some(expected.to_string()));
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
        assert_eq!(request.q, "test -site:spam.com -site:bad.org");
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
        assert_eq!(results[0].source, Some("Brave".to_string()));
        assert_eq!(results[0].score, None);
        assert_eq!(
            results[0].html_snippet,
            Some("This is a test snippet 1".to_string())
        );
        assert_eq!(results[0].date_published, Some("2 days ago".to_string()));
        assert!(results[0].images.is_some());
        assert_eq!(
            results[0].content_chunks,
            Some(vec!["Extra info 1".to_string()])
        );

        assert_eq!(results[1].title, "Test Result 2");
        assert_eq!(results[1].url, "https://test.org/2");
        assert_eq!(results[1].snippet, "This is a test snippet 2");
        assert_eq!(results[1].display_url, Some("test.org".to_string()));

        assert!(metadata.is_some());
        let meta = metadata.unwrap();
        assert_eq!(meta.query, "test query");
        assert_eq!(meta.total_results, Some(1000));
        assert_eq!(meta.search_time_ms, None);
        assert_eq!(meta.safe_search, Some(SafeSearchLevel::Medium));
        assert_eq!(meta.language, Some("en".to_string()));
        assert_eq!(meta.region, Some("us".to_string()));
        assert_eq!(meta.next_page_token, Some("20".to_string()));
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
        let response = BraveSearchResponse {
            query: Some(BraveQuery {
                original: "test query".to_string(),
                show_strict_warning: None,
                is_navigational: None,
                is_news_breaking: None,
                spellcheck_off: None,
                country: None,
                bad_results: None,
                should_fallback: None,
                postal_code: None,
                city: None,
                header_country: None,
                more_results_available: Some(false),
                state: None,
            }),
            web: None,
            discussions: None,
            infobox: None,
            videos: None,
            mixed: None,
            response_type: None,
        };

        let (results, metadata) = convert_response_to_results(response, &params);

        assert_eq!(results.len(), 0);
        assert!(metadata.is_some());
        let meta = metadata.unwrap();
        assert_eq!(meta.query, "test query");
        assert_eq!(meta.total_results, Some(0));
        assert_eq!(meta.search_time_ms, None);
        assert_eq!(meta.next_page_token, None);
    }
}
