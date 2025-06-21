use crate::client::SerperSearchItem;
use crate::golem::web_search::types::SearchResult;

pub fn convert_serper_result(item: SerperSearchItem) -> SearchResult {
    SearchResult {
        title: item.title,
        url: item.link,
        snippet: item.snippet,
        display_url: None,
        source: None,
        score: None,
        html_snippet: None,
        date_published: None,
        images: None,
        content_chunks: None,
    }
}
