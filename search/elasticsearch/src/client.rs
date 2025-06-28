use golem_search::golem::search::types::{Doc, Schema, SearchError};
use log::{debug, trace};
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ElasticSearchApi {
    client: Client,
    base_url: String,
    auth: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ElasticSearchError {
    #[serde(default)]
    pub error: ErrorDetails,
}

#[derive(Debug, Deserialize, Default)]
pub struct ErrorDetails {
    #[serde(rename = "type")]
    pub error_type: String,
    pub reason: String,
}

#[derive(Debug, Deserialize)]
pub struct EsSearchResponse {
    pub hits: EsHits,
    #[serde(rename = "_scroll_id")]
    pub scroll_id: Option<String>,
    pub took: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct EsHits {
    pub total: EsTotal,
    pub hits: Vec<EsHit>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EsTotal {
    Simple(u64),
    Detailed { value: u64 },
}

impl EsTotal {
    pub fn value(&self) -> u64 {
        match self {
            EsTotal::Simple(v) => *v,
            EsTotal::Detailed { value } => *value,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EsHit {
    #[serde(rename = "_id")]
    pub id: String,
    #[serde(rename = "_score")]
    pub score: Option<f64>,
    #[serde(rename = "_source")]
    pub source: Option<Value>,
    pub highlight: Option<HashMap<String, Vec<String>>>,
}

#[derive(Debug, Serialize)]
pub struct EsDoc {
    pub id: String,
    pub content: Value,
}

#[derive(Debug, Deserialize)]
pub struct IndexResponse {
    #[serde(rename = "acknowledged")]
    pub acknowledged: bool,
}

#[derive(Debug, Deserialize)]
pub struct GetResponse {
    #[serde(rename = "_id")]
    pub id: String,
    #[serde(rename = "_source")]
    pub source: Option<Value>,
    pub found: bool,
}

impl ElasticSearchApi {
    pub fn new(endpoint: &str, username: &str, password: &str) -> Self {
        let auth = format!("{}:{}", username, password);
        let auth = base64::encode(auth);
        
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            client,
            base_url: endpoint.trim_end_matches('/').to_string(),
            auth: Some(auth),
        }
    }

    pub fn empty() -> Self {
        Self {
            client: Client::new(),
            base_url: String::new(),
            auth: None,
        }
    }

    fn get_auth_header(&self) -> Option<String> {
        self.auth.as_ref().map(|auth| format!("Basic {}", auth))
    }

    async fn request(&self, method: reqwest::Method, path: &str, body: Option<Value>) -> Result<Response, SearchError> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        let mut request = self.client.request(method, &url);

        if let Some(auth) = self.get_auth_header() {
            request = request.header("Authorization", auth);
        }

        request = request.header("Content-Type", "application/json");

        if let Some(body) = body {
            request = request.json(&body);
        }

        trace!("Making request to: {}", url);
        
        let response = request.send().await
            .map_err(|e| SearchError::Internal(format!("Request failed: {}", e)))?;

        if response.status().is_success() {
            Ok(response)
        } else {
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            
            if let Ok(es_error) = serde_json::from_str::<ElasticSearchError>(&error_text) {
                match es_error.error.error_type.as_str() {
                    "index_not_found_exception" => Err(SearchError::IndexNotFound),
                    "parsing_exception" | "query_parsing_exception" => {
                        Err(SearchError::InvalidQuery(es_error.error.reason))
                    }
                    _ => Err(SearchError::Internal(format!("ES error: {}", es_error.error.reason)))
                }
            } else {
                Err(SearchError::Internal(format!("HTTP {}: {}", status, error_text)))
            }
        }
    }

    pub fn create_index(&self, name: &str) -> Result<(), SearchError> {
        let body = json!({
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
        });

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::PUT, name, Some(body)).await?;
            debug!("Created index: {}", name);
            Ok(())
        })
    }

    pub fn create_index_with_mapping(&self, name: &str, mapping: Value) -> Result<(), SearchError> {
        let body = json!({
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": mapping
        });

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::PUT, name, Some(body)).await?;
            debug!("Created index with mapping: {}", name);
            Ok(())
        })
    }

    pub fn delete_index(&self, name: &str) -> Result<(), SearchError> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::DELETE, name, None).await?;
            debug!("Deleted index: {}", name);
            Ok(())
        })
    }

    pub fn list_indexes(&self) -> Result<Vec<String>, SearchError> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let response = self.request(reqwest::Method::GET, "_cat/indices?format=json", None).await?;
            let indices: Vec<Value> = response.json().await
                .map_err(|e| SearchError::Internal(format!("Failed to parse response: {}", e)))?;
            
            let names = indices.into_iter()
                .filter_map(|index| index.get("index").and_then(|v| v.as_str().map(|s| s.to_string())))
                .collect();
            
            Ok(names)
        })
    }

    pub fn index_document(&self, index: &str, id: &str, doc: Value) -> Result<(), SearchError> {
        let path = format!("{}/_doc/{}", index, id);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::PUT, &path, Some(doc)).await?;
            debug!("Indexed document: {}/{}", index, id);
            Ok(())
        })
    }

    pub fn bulk_index(&self, index: &str, docs: Vec<EsDoc>) -> Result<(), SearchError> {
        if docs.is_empty() {
            return Ok(());
        }

        let mut body = String::new();
        for doc in &docs {
            let index_action = json!({"index": {"_index": index, "_id": doc.id}});
            body.push_str(&serde_json::to_string(&index_action).unwrap());
            body.push('\n');
            body.push_str(&serde_json::to_string(&doc.content).unwrap());
            body.push('\n');
        }

        let url = format!("{}/_bulk", self.base_url);
        let mut request = self.client.post(&url);

        if let Some(auth) = self.get_auth_header() {
            request = request.header("Authorization", auth);
        }

        request = request
            .header("Content-Type", "application/x-ndjson")
            .body(body);

        let response = request.send()
            .map_err(|e| SearchError::Internal(format!("Bulk request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().unwrap_or_default();
            return Err(SearchError::Internal(format!("Bulk indexing failed: {}", error_text)));
        }

        debug!("Bulk indexed {} documents to {}", docs.len(), index);
        Ok(())
    }

    pub fn delete_document(&self, index: &str, id: &str) -> Result<(), SearchError> {
        let path = format!("{}/_doc/{}", index, id);
        
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

        let mut body = String::new();
        for id in ids {
            let delete_action = json!({"delete": {"_index": index, "_id": id}});
            body.push_str(&serde_json::to_string(&delete_action).unwrap());
            body.push('\n');
        }

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let url = format!("{}/_bulk", self.base_url);
            let mut request = self.client.post(&url);

            if let Some(auth) = self.get_auth_header() {
                request = request.header("Authorization", auth);
            }

            request = request
                .header("Content-Type", "application/x-ndjson")
                .body(body);

            let response = request.send().await
                .map_err(|e| SearchError::Internal(format!("Bulk delete failed: {}", e)))?;

            if !response.status().is_success() {
                let error_text = response.text().await.unwrap_or_default();
                return Err(SearchError::Internal(format!("Bulk delete failed: {}", error_text)));
            }

            debug!("Bulk deleted {} documents from {}", ids.len(), index);
            Ok(())
        })
    }

    pub fn get_document(&self, index: &str, id: &str) -> Result<Option<Doc>, SearchError> {
        let path = format!("{}/_doc/{}", index, id);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            match self.request(reqwest::Method::GET, &path, None).await {
                Ok(response) => {
                    let get_response: GetResponse = response.json().await
                        .map_err(|e| SearchError::Internal(format!("Failed to parse response: {}", e)))?;
                    
                    if get_response.found {
                        if let Some(source) = get_response.source {
                            Ok(Some(Doc {
                                id: get_response.id,
                                content: serde_json::to_string(&source)
                                    .map_err(|e| SearchError::Internal(format!("JSON serialization error: {}", e)))?,
                            }))
                        } else {
                            Ok(None)
                        }
                    } else {
                        Ok(None)
                    }
                }
                Err(SearchError::Internal(msg)) if msg.contains("404") => Ok(None),
                Err(e) => Err(e),
            }
        })
    }

    pub fn search(&self, index: &str, query: Value) -> Result<EsSearchResponse, SearchError> {
        let path = format!("{}/_search", index);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let response = self.request(reqwest::Method::POST, &path, Some(query)).await?;
            let search_response: EsSearchResponse = response.json().await
                .map_err(|e| SearchError::Internal(format!("Failed to parse search response: {}", e)))?;
            
            debug!("Search completed in {}ms", search_response.took.unwrap_or(0));
            Ok(search_response)
        })
    }

    pub fn search_with_scroll(&self, index: &str, query: Value) -> Result<EsSearchResponse, SearchError> {
        let path = format!("{}/_search?scroll=1m", index);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let response = self.request(reqwest::Method::POST, &path, Some(query)).await?;
            let search_response: EsSearchResponse = response.json().await
                .map_err(|e| SearchError::Internal(format!("Failed to parse scroll search response: {}", e)))?;
            
            debug!("Scroll search initiated");
            Ok(search_response)
        })
    }

    pub fn scroll(&self, scroll_id: &str) -> Result<EsSearchResponse, SearchError> {
        let body = json!({
            "scroll": "1m",
            "scroll_id": scroll_id
        });
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let response = self.request(reqwest::Method::POST, "_search/scroll", Some(body)).await?;
            let search_response: EsSearchResponse = response.json().await
                .map_err(|e| SearchError::Internal(format!("Failed to parse scroll response: {}", e)))?;
            
            Ok(search_response)
        })
    }

    pub fn get_mapping(&self, index: &str) -> Result<Schema, SearchError> {
        let path = format!("{}/_mapping", index);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let response = self.request(reqwest::Method::GET, &path, None).await?;
            let mapping: Value = response.json().await
                .map_err(|e| SearchError::Internal(format!("Failed to parse mapping response: {}", e)))?;
            
            crate::conversions::mapping_to_schema(&mapping)
        })
    }

    pub fn update_mapping(&self, index: &str, mapping: Value) -> Result<(), SearchError> {
        let path = format!("{}/_mapping", index);
        
        self.request(reqwest::Method::PUT, &path, Some(mapping))?;
        debug!("Updated mapping for index: {}", index);
        Ok(())
    }
}
