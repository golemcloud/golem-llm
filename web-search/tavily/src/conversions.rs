use crate::client::{TavilyImage, TavilySearchRequest, TavilySearchResponse};
use golem_web_search::golem::web_search::types::{
    ImageResult, SearchMetadata, SearchParams, SearchResult, TimeRange,
};

pub fn convert_params_to_request(
    params: &SearchParams,
    _offset: Option<u32>,
) -> TavilySearchRequest {
    let max_results = params.max_results.unwrap_or(10).min(20);

    TavilySearchRequest {
        query: params.query.clone(),
        search_depth: Some(
            if params.advanced_answer.unwrap_or(false) {
                "advanced"
            } else {
                "basic"
            }
            .to_string(),
        ),
        topic: Some("news".to_string()), // using news as default for realtime updates
        max_results: Some(max_results),
        include_answer: params.advanced_answer,
        include_raw_content: Some(false),
        include_images: Some(true),
        include_domains: params.include_domains.clone(),
        exclude_domains: params.exclude_domains.clone(),
        time_range: params.time_range.as_ref().map(|tr| match tr {
            TimeRange::Day => "day".to_string(),
            TimeRange::Week => "week".to_string(),
            TimeRange::Month => "month".to_string(),
            TimeRange::Year => "year".to_string(),
        }),
        country: params.region.clone(),
        days: params.time_range.as_ref().map(|tr| match tr {
            TimeRange::Day => 1,
            TimeRange::Week => 7,
            TimeRange::Month => 30,
            TimeRange::Year => 365,
        }),
    }
}

pub fn convert_response_to_results(
    response: TavilySearchResponse,
    params: &SearchParams,
) -> (Vec<SearchResult>, Option<SearchMetadata>) {
    let search_results: Vec<SearchResult> = response
        .results
        .into_iter()
        .map(|result| {
            let images = response.images.as_ref().and_then(|imgs| {
                if !imgs.is_empty() {
                    Some(
                        imgs.iter()
                            .map(|img| match img {
                                TavilyImage::Url(url) => ImageResult {
                                    url: url.clone(),
                                    description: None,
                                },
                                TavilyImage::Object { url, description } => ImageResult {
                                    url: url.clone(),
                                    description: description.clone(),
                                },
                            })
                            .collect(),
                    )
                } else {
                    None
                }
            });

            SearchResult {
                title: result.title,
                url: result.url,
                snippet: result.content,
                display_url: None,
                source: Some("Tavily".to_string()),
                score: result.score,
                html_snippet: None,
                date_published: result.published_date,
                images,
                content_chunks: None,
            }
        })
        .collect();

    let metadata = Some(SearchMetadata {
        query: response.query,
        total_results: None,
        search_time_ms: response.response_time,
        safe_search: None,
        language: params.language.clone(),
        region: params.region.clone(),
        next_page_token: None,
        rate_limits: None,
    });

    (search_results, metadata)
}
