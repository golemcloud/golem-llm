#[allow(static_mut_refs)]
mod bindings;

use crate::bindings::exports::test::websearch_exports::test_websearch_api::*;
use crate::bindings::golem::web_search::web_search;
use crate::bindings::golem::web_search::types::SearchParams;

struct Component;

impl Guest for Component {
    fn test() -> Result<(), String> {
        println!("Testing web search functionality...");
        
        let query = "latest Rust programming language news";
        println!("Searching for: {}", query);
        
        let search_params = SearchParams {
            query: query.to_string(),
            safe_search: None,
            language: None,
            region: None,
            max_results: Some(5),
            time_range: None,
            include_domains: None,
            exclude_domains: None,
            include_images: None,
            include_html: None,
            advanced_answer: None,
        };
        
        let results = web_search::search_once(&search_params);
        println!("Search results: {:?}", results);
        
        match results {
            Ok((search_results, _metadata)) => {
                println!("Found {} results", search_results.len());
                for (i, result) in search_results.iter().enumerate().take(3) {
                    println!("Result {}: {}", i + 1, result.title);
                    println!("  URL: {}", result.url);
                    println!("  Snippet: {}", result.snippet);
                }
                Ok(())
            }
            Err(error) => {
                println!("Search error: {:?}", error);
                Err(format!("Search failed: {:?}", error))
            }
        }
    }
}

bindings::export!(Component with_types_in bindings); 