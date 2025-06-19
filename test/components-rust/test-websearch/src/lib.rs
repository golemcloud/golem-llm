#[allow(static_mut_refs)]
mod bindings;

use golem_rust::atomically;
use crate::bindings::exports::golem::it_exports::api_inline_functions::*;
use crate::bindings::golem::web_search::web_search;

struct Component;

impl Guest for Component {
    fn test() -> Result<(), String> {
        println!("Testing web search functionality...");
        
        let query = "latest Rust programming language news";
        println!("Searching for: {}", query);
        
        let results = web_search::search(query);
        println!("Search results: {:?}", results);
        
        match results {
            Ok(search_results) => {
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

bindings::export!(Component); 