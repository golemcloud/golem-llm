use crate::client::{SerperSearchRequest, SerperSearchResponse};
use golem_web_search::golem::web_search::types::{
    ImageResult, SearchMetadata, SearchParams, SearchResult, TimeRange,
};

pub fn convert_params_to_request(params: &SearchParams, page: Option<u32>) -> SerperSearchRequest {
    let num = params.max_results.unwrap_or(10).min(100); // Serper max is 100

    SerperSearchRequest {
        q: params.query.clone(),
        location: params.region.clone(),
        gl: params.region.clone(),   // Country code
        hl: params.language.clone(), // Language code
        num: Some(num),
        autocorrect: Some(true),
        tbs: params.time_range.as_ref().map(|tr| match tr {
            TimeRange::Day => "d".to_string(),
            TimeRange::Week => "w".to_string(),
            TimeRange::Month => "m".to_string(),
            TimeRange::Year => "y".to_string(),
        }),
        page,
    }
}

pub fn convert_response_to_results(
    response: SerperSearchResponse,
    params: &SearchParams,
) -> (Vec<SearchResult>, Option<SearchMetadata>) {
    let mut search_results = Vec::new();

    // Convert organic results
    if let Some(organic_results) = response.organic {
        for result in organic_results {
            let images = response.images.as_ref().and_then(|imgs| {
                if !imgs.is_empty() {
                    Some(
                        imgs.iter()
                            .take(3) // Limit to first 3 images
                            .map(|img| ImageResult {
                                url: img.image_url.clone(),
                                description: Some(img.title.clone()),
                            })
                            .collect(),
                    )
                } else {
                    None
                }
            });

            search_results.push(SearchResult {
                title: result.title,
                url: result.link,
                snippet: result.snippet,
                display_url: None, // Serper doesn't provide display URL
                source: Some("Serper".to_string()),
                score: result.position.map(|p| 1.0 / (p as f64 + 1.0)), // Convert position to score
                html_snippet: None,
                date_published: result.date,
                images,
                content_chunks: None,
            });
        }
    }

    // Convert answer box to a special result if present
    if let Some(answer_box) = response.answer_box {
        search_results.insert(
            0,
            SearchResult {
                title: answer_box.title,
                url: answer_box.link.unwrap_or_default(),
                snippet: answer_box.answer.or(answer_box.snippet).unwrap_or_default(),
                display_url: None,
                source: Some("Serper Answer Box".to_string()),
                score: Some(1.0), // Highest score for answer box
                html_snippet: None,
                date_published: None,
                images: None,
                content_chunks: None,
            },
        );
    }

    // Convert knowledge graph to a special result if present
    if let Some(kg) = response.knowledge_graph {
        let kg_images = kg.image_url.map(|url| {
            vec![ImageResult {
                url,
                description: Some(kg.title.clone()),
            }]
        });

        search_results.insert(
            0,
            SearchResult {
                title: kg.title,
                url: kg.website.unwrap_or_default(),
                snippet: kg.description.unwrap_or_default(),
                display_url: None,
                source: Some("Serper Knowledge Graph".to_string()),
                score: Some(1.0), // Highest score for knowledge graph
                html_snippet: None,
                date_published: None,
                images: kg_images,
                content_chunks: None,
            },
        );
    }

    let total_results = response
        .search_information
        .as_ref()
        .and_then(|info| info.total_results.as_ref())
        .and_then(|total| total.parse::<u64>().ok());

    let search_time_ms = response
        .search_information
        .as_ref()
        .and_then(|info| info.time_taken.map(|t| t * 1000.0)); // Convert to milliseconds

    let metadata = Some(SearchMetadata {
        query: response
            .search_parameters
            .as_ref()
            .map(|sp| sp.q.clone())
            .unwrap_or_else(|| params.query.clone()),
        total_results,
        search_time_ms,
        safe_search: None,
        language: params.language.clone(),
        region: params.region.clone(),
        next_page_token: None, // Serper uses page numbers
        rate_limits: None,
    });

    (search_results, metadata)
}
