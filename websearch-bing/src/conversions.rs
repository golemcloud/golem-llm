use crate::client::BingSearchItem;
use crate::golem::web_search::types::{ImageResult, SearchResult};

pub fn convert_bing_result(item: BingSearchItem) -> SearchResult {
    let mut images = None;
    let mut date_published = None;

    if let Some(images_data) = item.images {
        images = Some(
            images_data
                .into_iter()
                .map(|img| ImageResult {
                    url: img.url,
                    description: img.description,
                })
                .collect(),
        );
    }

    if let Some(date) = item.date_published {
        date_published = Some(date);
    }

    SearchResult {
        title: item.name,
        url: item.url,
        snippet: item.snippet,
        display_url: item.display_url,
        source: None,
        score: None,
        html_snippet: None,
        date_published,
        images,
        content_chunks: None,
    }
}
