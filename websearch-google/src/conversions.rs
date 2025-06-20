use crate::client::GoogleSearchItem;
use crate::golem::web_search::types::{ImageResult, SearchResult};

pub fn convert_google_result(item: GoogleSearchItem) -> SearchResult {
    let mut images = None;
    let mut date_published = None;

    if let Some(page_map) = item.page_map {
        if let Some(cse_images) = page_map.cse_image {
            images = Some(
                cse_images
                    .into_iter()
                    .map(|img| ImageResult {
                        url: img.src,
                        description: img.description,
                    })
                    .collect(),
            );
        }
        if let Some(meta_tags) = page_map.meta_tags {
            for meta in meta_tags {
                if let Some(published) = meta.published_time {
                    date_published = Some(published);
                    break;
                }
            }
        }
    }

    SearchResult {
        title: item.title,
        url: item.link,
        snippet: item.snippet,
        display_url: item.display_link,
        source: item.source,
        score: None,
        html_snippet: item.html_snippet,
        date_published,
        images,
        content_chunks: None,
    }
}
