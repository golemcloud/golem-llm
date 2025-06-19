#[allow(static_mut_refs)]
mod bindings;

use golem_rust::atomically;
use crate::bindings::exports::test::web_search_exports::test_web_search_api::*;
use crate::bindings::golem::web_search::web_search;
use crate::bindings::test::helper_client::test_helper_client::TestHelperApi;

struct Component;

impl Guest for Component {
    /// test1 demonstrates a simple web search query
    fn test1() -> String {
        let config = web_search::Config {
            provider: "google".to_string(),
            provider_options: vec![],
        };

        println!("Sending web search request...");
        let response = web_search::search(
            "Rust programming language",
            &config,
        );
        println!("Response: {:?}", response);

        match response {
            Ok(results) => {
                format!(
                    "Found {} results. First result: {}",
                    results.len(),
                    results.first()
                        .map(|r| format!("{} - {}", r.title, r.url))
                        .unwrap_or("No results".to_string())
                )
            }
            Err(error) => {
                format!(
                    "ERROR: {}",
                    error
                )
            }
        }
    }

    /// test2 demonstrates a more complex web search query with multiple terms
    fn test2() -> String {
        let config = web_search::Config {
            provider: "google".to_string(),
            provider_options: vec![],
        };

        println!("Sending complex web search request...");
        let response = web_search::search(
            "WebAssembly WASI components tutorial",
            &config,
        );
        println!("Response: {:?}", response);

        match response {
            Ok(results) => {
                let mut output = format!("Found {} results:\n", results.len());
                for (i, result) in results.iter().take(3).enumerate() {
                    output.push_str(&format!(
                        "{}. {} - {} ({})\n",
                        i + 1,
                        result.title,
                        result.url,
                        result.snippet.as_deref().unwrap_or("No snippet")
                    ));
                }
                output
            }
            Err(error) => {
                format!("ERROR: {}", error)
            }
        }
    }

    /// test3 demonstrates error handling with an invalid search query
    fn test3() -> String {
        let config = web_search::Config {
            provider: "google".to_string(),
            provider_options: vec![],
        };

        println!("Sending empty web search request...");
        let response = web_search::search("", &config);
        println!("Response: {:?}", response);

        match response {
            Ok(results) => {
                format!("Unexpected success with {} results", results.len())
            }
            Err(error) => {
                format!("Expected error: {}", error)
            }
        }
    }

    /// test4 simulates a crash during a web search, but only first time.
    /// after the automatic recovery it will continue and finish the request successfully.
    fn test4() -> String {
        let config = web_search::Config {
            provider: "google".to_string(),
            provider_options: vec![],
        };

        let name = std::env::var("GOLEM_WORKER_NAME").unwrap();
        
        atomically(|| {
            let client = TestHelperApi::new(&name);
            let counter = client.blocking_inc_and_get();
            if counter == 1 {
                panic!("Simulating crash during web search")
            }
        });

        println!("Sending web search request after recovery...");
        let response = web_search::search(
            "Golem cloud WebAssembly components",
            &config,
        );
        println!("Response: {:?}", response);

        match response {
            Ok(results) => {
                format!(
                    "Recovery successful! Found {} results. First: {}",
                    results.len(),
                    results.first()
                        .map(|r| format!("{} - {}", r.title, r.url))
                        .unwrap_or("No results".to_string())
                )
            }
            Err(error) => {
                format!("ERROR after recovery: {}", error)
            }
        }
    }
}

bindings::export!(Component with_types_in bindings);