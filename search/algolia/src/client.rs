use golem_search::golem::search::types::{Doc, Schema, SearchError};
use log::{debug, trace};
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct AlgoliaApi {
    client: Client,
    app_id: String,
    api_key: String,
    base_url: String,
}

#[derive(Debug, Deserialize)]
pub struct AlgoliaError {
    pub message: String,
    pub status: u16,
}

#[derive(Debug, Deserialize)]
pub struct AlgoliaSearchResponse {
    pub hits: Vec<AlgoliaHit>,
    #[serde(rename = "nbHits")]
    pub nb_hits: u64,
    pub page: u32,
    #[serde(rename = "nbPages")]
    pub nb_pages: u32,
    #[serde(rename = "hitsPerPage")]
    pub hits_per_page: u32,
    #[serde(rename = "processingTimeMS")]
    pub processing_time_ms: u32,
    pub cursor: Option<String>,
    pub facets: Option<HashMap<String, HashMap<String, u32>>>,
}

#[derive(Debug, Deserialize)]
pub struct AlgoliaHit {
    #[serde(rename = "objectID")]
    pub object_id: String,
    #[serde(rename = "_highlightResult")]
    pub highlight_result: Option<HashMap<String, AlgoliaHighlight>>,
    #[serde(flatten)]
    pub attributes: Value,
}

#[derive(Debug, Deserialize)]
pub struct AlgoliaHighlight {
    pub value: String,
    #[serde(rename = "matchLevel")]
    pub match_level: String,
}

#[derive(Debug, Serialize)]
pub struct AlgoliaDoc {
    #[serde(rename = "objectID")]
    pub object_id: String,
    #[serde(flatten)]
    pub content: Value,
}

#[derive(Debug, Deserialize)]
pub struct AlgoliaBrowseResponse {
    pub hits: Vec<AlgoliaHit>,
    pub cursor: Option<String>,
    #[serde(rename = "processingTimeMS")]
    pub processing_time_ms: u32,
}

#[derive(Debug, Deserialize)]
pub struct IndexListResponse {
    pub items: Vec<IndexInfo>,
}

#[derive(Debug, Deserialize)]
pub struct IndexInfo {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct TaskResponse {
    #[serde(rename = "taskID")]
    pub task_id: u64,
}

impl AlgoliaApi {
    pub fn new(app_id: &str, api_key: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            client,
            app_id: app_id.to_string(),
            api_key: api_key.to_string(),
            base_url: format!("https://{}-dsn.algolia.net/1", app_id),
        }
    }

    pub fn empty() -> Self {
        Self {
            client: Client::new(),
            app_id: String::new(),
            api_key: String::new(),
            base_url: String::new(),
        }
    }

    async fn request(&self, method: reqwest::Method, path: &str, body: Option<Value>) -> Result<Response, SearchError> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        let mut request = self.client.request(method, &url);

        request = request
            .header("X-Algolia-Application-Id", &self.app_id)
            .header("X-Algolia-API-Key", &self.api_key)
            .header("Content-Type", "application/json");

        if let Some(body) = body {
            request = request.json(&body);
        }

        trace!("Making Algolia request to: {}", url);
        
        let response = request.send().await
            .map_err(|e| SearchError::Internal(format!("Request failed: {}", e)))?;

        if response.status().is_success() {
            Ok(response)
        } else {
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            
            if let Ok(algolia_error) = serde_json::from_str::<AlgoliaError>(&error_text) {
                match status {
                    404 => Err(SearchError::IndexNotFound),
                    400 => Err(SearchError::InvalidQuery(algolia_error.message)),
                    _ => Err(SearchError::Internal(format!("Algolia error: {}", algolia_error.message)))
                }
            } else {
                Err(SearchError::Internal(format!("HTTP {}: {}", status, error_text)))
            }
        }
    }

    pub fn create_index(&self, name: &str) -> Result<(), SearchError> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            // Algolia creates indexes automatically when first document is added
            // For explicit creation, we can set empty settings
            let body = json!({});
            self.request(reqwest::Method::POST, &format!("indexes/{}/settings", name), Some(body)).await?;
            debug!("Created Algolia index: {}", name);
            Ok(())
        })
    }

    pub fn create_index_with_schema(&self, name: &str, settings: Value) -> Result<(), SearchError> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::PUT, &format!("indexes/{}/settings", name), Some(settings)).await?;
            debug!("Created Algolia index with settings: {}", name);
            Ok(())
        })
    }

    pub fn delete_index(&self, name: &str) -> Result<(), SearchError> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::DELETE, &format!("indexes/{}", name), None).await?;
            debug!("Deleted Algolia index: {}", name);
            Ok(())
        })
    }

    pub fn list_indexes(&self) -> Result<Vec<String>, SearchError> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let response = self.request(reqwest::Method::GET, "indexes", None).await?;
            let list_response: IndexListResponse = response.json().await
                .map_err(|e| SearchError::Internal(format!("Failed to parse response: {}", e)))?;
            
            let names = list_response.items.into_iter()
                .map(|index| index.name)
                .collect();
            
            Ok(names)
        })
    }

    pub fn index_document(&self, index: &str, doc: AlgoliaDoc) -> Result<(), SearchError> {
        let path = format!("indexes/{}", index);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::POST, &path, Some(serde_json::to_value(&doc).unwrap())).await?;
            debug!("Indexed document: {}/{}", index, doc.object_id);
            Ok(())
        })
    }

    pub fn bulk_index(&self, index: &str, docs: Vec<AlgoliaDoc>) -> Result<(), SearchError> {
        if docs.is_empty() {
            return Ok(());
        }

        let path = format!("indexes/{}/batch", index);
        let body = json!({
            "requests": docs.iter().map(|doc| {
                json!({
                    "action": "addObject",
                    "body": doc
                })
            }).collect::<Vec<_>>()
        });

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::POST, &path, Some(body)).await?;
            debug!("Bulk indexed {} documents to {}", docs.len(), index);
            Ok(())
        })
    }

    pub fn delete_document(&self, index: &str, id: &str) -> Result<(), SearchError> {
        let path = format!("indexes/{}/{}", index, id);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::DELETE, &path, None).await?;
            debug!("Deleted document: {}/{}", index, id);
            Ok(())
        })
    }

    pub fn bulk_delete(&self, index: &str, ids: Vec<String>) -> Result<(), SearchError> {
        if ids.is_empty() {
            return Ok(());
        }

        let path = format!("indexes/{}/batch", index);
        let body = json!({
            "requests": ids.iter().map(|id| {
                json!({
                    "action": "deleteObject",
                    "body": {
                        "objectID": id
                    }
                })
            }).collect::<Vec<_>>()
        });

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::POST, &path, Some(body)).await?;
            debug!("Bulk deleted {} documents from {}", ids.len(), index);
            Ok(())
        })
    }

    pub fn get_document(&self, index: &str, id: &str) -> Result<Option<Doc>, SearchError> {
        let path = format!("indexes/{}/{}", index, id);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            match self.request(reqwest::Method::GET, &path, None).await {
                Ok(response) => {
                    let mut doc_value: Value = response.json().await
                        .map_err(|e| SearchError::Internal(format!("Failed to parse response: {}", e)))?;
                    
                    // Remove objectID from content to match Doc structure
                    let object_id = doc_value.get("objectID")
                        .and_then(|v| v.as_str())
                        .unwrap_or(id)
                        .to_string();
                    
                    if let Some(obj) = doc_value.as_object_mut() {
                        obj.remove("objectID");
                    }
                    
                    Ok(Some(Doc {
                        id: object_id,
                        content: serde_json::to_string(&doc_value)
                            .map_err(|e| SearchError::Internal(format!("JSON serialization error: {}", e)))?,
                    }))
                }
                Err(SearchError::Internal(msg)) if msg.contains("404") => Ok(None),
                Err(e) => Err(e),
            }
        })
    }

    pub fn search(&self, index: &str, query: Value) -> Result<AlgoliaSearchResponse, SearchError> {
        let path = format!("indexes/{}/query", index);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let response = self.request(reqwest::Method::POST, &path, Some(query)).await?;
            let search_response: AlgoliaSearchResponse = response.json().await
                .map_err(|e| SearchError::Internal(format!("Failed to parse search response: {}", e)))?;
            
            debug!("Algolia search completed in {}ms", search_response.processing_time_ms);
            Ok(search_response)
        })
    }

    pub fn browse(&self, index: &str, cursor: Option<&str>) -> Result<AlgoliaBrowseResponse, SearchError> {
        let mut path = format!("indexes/{}/browse", index);
        if let Some(cursor) = cursor {
            path.push_str(&format!("?cursor={}", cursor));
        }
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let response = self.request(reqwest::Method::POST, &path, Some(json!({}))).await?;
            let browse_response: AlgoliaBrowseResponse = response.json().await
                .map_err(|e| SearchError::Internal(format!("Failed to parse browse response: {}", e)))?;
            
            debug!("Algolia browse completed");
            Ok(browse_response)
        })
    }

    pub fn get_settings(&self, index: &str) -> Result<Value, SearchError> {
        let path = format!("indexes/{}/settings", index);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let response = self.request(reqwest::Method::GET, &path, None).await?;
            let settings: Value = response.json().await
                .map_err(|e| SearchError::Internal(format!("Failed to parse settings response: {}", e)))?;
            
            Ok(settings)
        })
    }

    pub fn update_settings(&self, index: &str, settings: Value) -> Result<(), SearchError> {
        let path = format!("indexes/{}/settings", index);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::PUT, &path, Some(settings)).await?;
            debug!("Updated settings for Algolia index: {}", index);
            Ok(())
        })
    }
}
