use crate::golem::web_search::types::{ImageResult, SearchResult};
use crate::client::BraveSearchItem;

pub fn convert_brave_result(item: BraveSearchItem) -> SearchResult {
    let images = item.thumbnail.map(|url| vec![ImageResult { url, description: None }]);
    SearchResult {
        title: item.title,
        url: item.url,
        snippet: item.description,
        display_url: None,
        source: None,
        score: None,
        html_snippet: None,
        date_published: item.date_published,
        images,
        content_chunks: None,
    }
} 