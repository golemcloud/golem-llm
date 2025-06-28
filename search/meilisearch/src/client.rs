use golem_search::golem::search::types::{Doc, Schema, SearchError};
use log::{debug, trace};
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct MeilisearchApi {
    client: Client,
    base_url: String,
    api_key: String,
}

#[derive(Debug, Deserialize)]
pub struct MeilisearchError {
    pub message: String,
    pub code: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub link: String,
}

#[derive(Debug, Deserialize)]
pub struct MeilisearchSearchResponse {
    pub hits: Vec<MeilisearchHit>,
    #[serde(rename = "processingTimeMs")]
    pub processing_time_ms: u64,
    pub query: String,
    pub limit: u32,
    pub offset: u32,
    #[serde(rename = "estimatedTotalHits")]
    pub estimated_total_hits: Option<u64>,
    #[serde(rename = "facetDistribution")]
    pub facet_distribution: Option<HashMap<String, HashMap<String, u64>>>,
}

#[derive(Debug, Deserialize)]
pub struct MeilisearchHit {
    #[serde(flatten)]
    pub source: Value,
    #[serde(rename = "_formatted")]
    pub formatted: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct MeilisearchSearchQuery {
    pub q: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub facets: Option<Vec<String>>,
    #[serde(rename = "attributesToHighlight", skip_serializing_if = "Option::is_none")]
    pub attributes_to_highlight: Option<Vec<String>>,
    #[serde(rename = "highlightPreTag", skip_serializing_if = "Option::is_none")]
    pub highlight_pre_tag: Option<String>,
    #[serde(rename = "highlightPostTag", skip_serializing_if = "Option::is_none")]
    pub highlight_post_tag: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MeilisearchDoc {
    pub id: String,
    #[serde(flatten)]
    pub content: Value,
}

#[derive(Debug, Deserialize)]
pub struct MeilisearchTask {
    #[serde(rename = "taskUid")]
    pub task_uid: u64,
    #[serde(rename = "indexUid")]
    pub index_uid: String,
    pub status: String,
    #[serde(rename = "type")]
    pub task_type: String,
    #[serde(rename = "enqueuedAt")]
    pub enqueued_at: String,
}

#[derive(Debug, Deserialize)]
pub struct MeilisearchIndexes {
    pub results: Vec<MeilisearchIndex>,
}

#[derive(Debug, Deserialize)]
pub struct MeilisearchIndex {
    pub uid: String,
    #[serde(rename = "primaryKey")]
    pub primary_key: Option<String>,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
}

#[derive(Debug, Serialize)]
pub struct MeilisearchIndexSettings {
    #[serde(rename = "searchableAttributes", skip_serializing_if = "Option::is_none")]
    pub searchable_attributes: Option<Vec<String>>,
    #[serde(rename = "filterableAttributes", skip_serializing_if = "Option::is_none")]
    pub filterable_attributes: Option<Vec<String>>,
    #[serde(rename = "sortableAttributes", skip_serializing_if = "Option::is_none")]
    pub sortable_attributes: Option<Vec<String>>,
    #[serde(rename = "rankingRules", skip_serializing_if = "Option::is_none")]
    pub ranking_rules: Option<Vec<String>>,
}

impl MeilisearchApi {
    pub fn new(host: String, api_key: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: host.trim_end_matches('/').to_string(),
            api_key,
        }
    }

    pub fn empty() -> Self {
        Self {
            client: Client::new(),
            base_url: String::new(),
            api_key: String::new(),
        }
    }

    fn handle_error(&self, response: Response) -> SearchError {
        let status = response.status();
        let status_code = status.as_u16();
        
        match response.text() {
            Ok(body) => {
                if let Ok(error) = serde_json::from_str::<MeilisearchError>(&body) {
                    SearchError::ProviderError {
                        code: Some(status_code as u32),
                        message: format!("{}: {}", error.code, error.message),
                    }
                } else {
                    SearchError::ProviderError {
                        code: Some(status_code as u32),
                        message: format!("HTTP {}: {}", status_code, body),
                    }
                }
            }
            Err(_) => SearchError::ProviderError {
                code: Some(status_code as u32),
                message: format!("HTTP {}: {}", status_code, status.canonical_reason().unwrap_or("Unknown error")),
            },
        }
    }

    pub fn create_index(&self, name: &str) -> Result<(), SearchError> {
        debug!("Creating Meilisearch index: {}", name);
        
        let url = format!("{}/indexes", self.base_url);
        let payload = json!({
            "uid": name,
            "primaryKey": "id"
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .map_err(|e| SearchError::ConnectionError {
                message: e.to_string(),
            })?;

        if response.status().is_success() {
            trace!("Successfully created index: {}", name);
            Ok(())
        } else {
            Err(self.handle_error(response))
        }
    }

    pub fn create_index_with_settings(&self, name: &str, settings: MeilisearchIndexSettings) -> Result<(), SearchError> {
        self.create_index(name)?;
        self.update_settings(name, settings)?;
        Ok(())
    }

    pub fn delete_index(&self, name: &str) -> Result<(), SearchError> {
        debug!("Deleting Meilisearch index: {}", name);
        
        let url = format!("{}/indexes/{}", self.base_url, name);

        let response = self
            .client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .map_err(|e| SearchError::ConnectionError {
                message: e.to_string(),
            })?;

        if response.status().is_success() {
            trace!("Successfully deleted index: {}", name);
            Ok(())
        } else {
            Err(self.handle_error(response))
        }
    }

    pub fn list_indexes(&self) -> Result<Vec<String>, SearchError> {
        debug!("Listing Meilisearch indexes");
        
        let url = format!("{}/indexes", self.base_url);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .map_err(|e| SearchError::ConnectionError {
                message: e.to_string(),
            })?;

        if response.status().is_success() {
            let indexes: MeilisearchIndexes = response.json().map_err(|e| SearchError::ParseError {
                message: e.to_string(),
            })?;
            
            Ok(indexes.results.into_iter().map(|idx| idx.uid).collect())
        } else {
            Err(self.handle_error(response))
        }
    }

    pub fn index_document(&self, index: &str, doc_id: &str, content: Value) -> Result<(), SearchError> {
        debug!("Indexing document {} in index {}", doc_id, index);
        
        let url = format!("{}/indexes/{}/documents", self.base_url, index);
        let mut doc = content;
        if let Value::Object(ref mut map) = doc {
            map.insert("id".to_string(), Value::String(doc_id.to_string()));
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&[doc])
            .send()
            .map_err(|e| SearchError::ConnectionError {
                message: e.to_string(),
            })?;

        if response.status().is_success() {
            trace!("Successfully indexed document: {}", doc_id);
            Ok(())
        } else {
            Err(self.handle_error(response))
        }
    }

    pub fn bulk_index(&self, index: &str, docs: Vec<MeilisearchDoc>) -> Result<(), SearchError> {
        debug!("Bulk indexing {} documents in index {}", docs.len(), index);
        
        let url = format!("{}/indexes/{}/documents", self.base_url, index);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&docs)
            .send()
            .map_err(|e| SearchError::ConnectionError {
                message: e.to_string(),
            })?;

        if response.status().is_success() {
            trace!("Successfully bulk indexed {} documents", docs.len());
            Ok(())
        } else {
            Err(self.handle_error(response))
        }
    }

    pub fn delete_document(&self, index: &str, doc_id: &str) -> Result<(), SearchError> {
        debug!("Deleting document {} from index {}", doc_id, index);
        
        let url = format!("{}/indexes/{}/documents/{}", self.base_url, index, doc_id);

        let response = self
            .client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .map_err(|e| SearchError::ConnectionError {
                message: e.to_string(),
            })?;

        if response.status().is_success() {
            trace!("Successfully deleted document: {}", doc_id);
            Ok(())
        } else {
            Err(self.handle_error(response))
        }
    }

    pub fn bulk_delete(&self, index: &str, ids: Vec<String>) -> Result<(), SearchError> {
        debug!("Bulk deleting {} documents from index {}", ids.len(), index);
        
        let url = format!("{}/indexes/{}/documents/delete-batch", self.base_url, index);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&ids)
            .send()
            .map_err(|e| SearchError::ConnectionError {
                message: e.to_string(),
            })?;

        if response.status().is_success() {
            trace!("Successfully bulk deleted {} documents", ids.len());
            Ok(())
        } else {
            Err(self.handle_error(response))
        }
    }

    pub fn get_document(&self, index: &str, doc_id: &str) -> Result<Option<Doc>, SearchError> {
        debug!("Getting document {} from index {}", doc_id, index);
        
        let url = format!("{}/indexes/{}/documents/{}", self.base_url, index, doc_id);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .map_err(|e| SearchError::ConnectionError {
                message: e.to_string(),
            })?;

        if response.status().is_success() {
            let doc: Value = response.json().map_err(|e| SearchError::ParseError {
                message: e.to_string(),
            })?;
            
            if let Some(id) = doc.get("id").and_then(|v| v.as_str()) {
                Ok(Some(Doc {
                    id: id.to_string(),
                    content: doc,
                }))
            } else {
                Err(SearchError::ParseError {
                    message: "Document missing id field".to_string(),
                })
            }
        } else if response.status().as_u16() == 404 {
            Ok(None)
        } else {
            Err(self.handle_error(response))
        }
    }

    pub fn search(&self, index: &str, query: MeilisearchSearchQuery) -> Result<MeilisearchSearchResponse, SearchError> {
        debug!("Searching index {} with query: {:?}", index, query);
        
        let url = format!("{}/indexes/{}/search", self.base_url, index);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&query)
            .send()
            .map_err(|e| SearchError::ConnectionError {
                message: e.to_string(),
            })?;

        if response.status().is_success() {
            let search_response: MeilisearchSearchResponse = response.json().map_err(|e| SearchError::ParseError {
                message: e.to_string(),
            })?;
            
            trace!("Search completed in {}ms, found {} hits", 
                   search_response.processing_time_ms, 
                   search_response.hits.len());
            
            Ok(search_response)
        } else {
            Err(self.handle_error(response))
        }
    }

    pub fn get_settings(&self, index: &str) -> Result<MeilisearchIndexSettings, SearchError> {
        debug!("Getting settings for index {}", index);
        
        let url = format!("{}/indexes/{}/settings", self.base_url, index);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .map_err(|e| SearchError::ConnectionError {
                message: e.to_string(),
            })?;

        if response.status().is_success() {
            let settings: MeilisearchIndexSettings = response.json().map_err(|e| SearchError::ParseError {
                message: e.to_string(),
            })?;
            
            Ok(settings)
        } else {
            Err(self.handle_error(response))
        }
    }

    pub fn update_settings(&self, index: &str, settings: MeilisearchIndexSettings) -> Result<(), SearchError> {
        debug!("Updating settings for index {}", index);
        
        let url = format!("{}/indexes/{}/settings", self.base_url, index);

        let response = self
            .client
            .patch(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&settings)
            .send()
            .map_err(|e| SearchError::ConnectionError {
                message: e.to_string(),
            })?;

        if response.status().is_success() {
            trace!("Successfully updated settings for index: {}", index);
            Ok(())
        } else {
            Err(self.handle_error(response))
        }
    }
}
