use crate::golem::web_search::types::{ImageResult, SearchResult};
use crate::client::TavilySearchItem;

pub fn convert_tavily_result(item: TavilySearchItem) -> SearchResult {
    SearchResult {
        title: item.title,
        url: item.url,
        snippet: item.content,
        display_url: None,
        source: None,
        score: item.score,
        html_snippet: None,
        date_published: None,
        images: None,
        content_chunks: None,
    }
} 