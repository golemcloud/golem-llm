use golem_search::golem::search::types::{Doc, Schema, SearchError};
use log::{debug, trace};
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct TypesenseApi {
    client: Client,
    base_url: String,
    api_key: String,
}

#[derive(Debug, Deserialize)]
pub struct TypesenseError {
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct TypesenseSearchResponse {
    pub hits: Vec<TypesenseHit>,
    pub found: u64,
    pub search_time_ms: u64,
    pub page: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct TypesenseHit {
    pub document: Value,
    #[serde(default)]
    pub highlights: Vec<TypesenseHighlight>,
    pub text_match: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct TypesenseHighlight {
    pub field: String,
    pub snippet: String,
    pub value: String,
}

#[derive(Debug, Serialize)]
pub struct TypesenseDoc {
    pub id: String,
    #[serde(flatten)]
    pub fields: Value,
}

#[derive(Debug, Deserialize)]
pub struct CollectionResponse {
    pub name: String,
    pub num_documents: u64,
    pub fields: Vec<TypesenseField>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TypesenseField {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: String,
    #[serde(default)]
    pub facet: bool,
    #[serde(default)]
    pub index: bool,
    #[serde(default)]
    pub sort: bool,
    #[serde(default)]
    pub optional: bool,
}

#[derive(Debug, Serialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub fields: Vec<TypesenseField>,
}

#[derive(Debug, Deserialize)]
pub struct GetDocumentResponse {
    pub id: String,
    #[serde(flatten)]
    pub fields: Value,
}

#[derive(Debug, Deserialize)]
pub struct ExportResponse {
    #[serde(flatten)]
    pub document: Value,
}

impl TypesenseApi {
    pub fn new(host: &str, api_key: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            client,
            base_url: host.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
        }
    }

    pub fn empty() -> Self {
        Self {
            client: Client::new(),
            base_url: String::new(),
            api_key: String::new(),
        }
    }

    async fn request(&self, method: reqwest::Method, path: &str, body: Option<Value>) -> Result<Response, SearchError> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        
        let mut request = self.client.request(method, &url);
        request = request.header("X-TYPESENSE-API-KEY", &self.api_key);
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
            
            if let Ok(ts_error) = serde_json::from_str::<TypesenseError>(&error_text) {
                match status {
                    404 => Err(SearchError::IndexNotFound),
                    400 => Err(SearchError::InvalidQuery(ts_error.message)),
                    _ => Err(SearchError::Internal(format!("Typesense error: {}", ts_error.message)))
                }
            } else {
                Err(SearchError::Internal(format!("HTTP {}: {}", status, error_text)))
            }
        }
    }

    pub fn create_collection(&self, name: &str, schema: Option<&Schema>) -> Result<(), SearchError> {
        let fields = if let Some(schema) = schema {
            crate::conversions::schema_to_typesense_fields(schema)
        } else {
            vec![
                TypesenseField {
                    name: "id".to_string(),
                    field_type: "string".to_string(),
                    facet: false,
                    index: true,
                    sort: false,
                    optional: false,
                },
                TypesenseField {
                    name: ".*".to_string(),
                    field_type: "auto".to_string(),
                    facet: false,
                    index: true,
                    sort: false,
                    optional: true,
                },
            ]
        };

        let collection_request = CreateCollectionRequest {
            name: name.to_string(),
            fields,
        };

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::POST, "collections", Some(serde_json::to_value(collection_request).unwrap())).await?;
            debug!("Created collection: {}", name);
            Ok(())
        })
    }

    pub fn delete_collection(&self, name: &str) -> Result<(), SearchError> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let path = format!("collections/{}", name);
            self.request(reqwest::Method::DELETE, &path, None).await?;
            debug!("Deleted collection: {}", name);
            Ok(())
        })
    }

    pub fn list_collections(&self) -> Result<Vec<String>, SearchError> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let response = self.request(reqwest::Method::GET, "collections", None).await?;
            let collections: Vec<CollectionResponse> = response.json().await
                .map_err(|e| SearchError::Internal(format!("Failed to parse response: {}", e)))?;
            
            let names = collections.into_iter().map(|c| c.name).collect();
            Ok(names)
        })
    }

    pub fn index_document(&self, collection: &str, doc: TypesenseDoc) -> Result<(), SearchError> {
        let path = format!("collections/{}/documents", collection);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::POST, &path, Some(serde_json::to_value(doc).unwrap())).await?;
            debug!("Indexed document to collection: {}", collection);
            Ok(())
        })
    }

    pub fn upsert_document(&self, collection: &str, doc: TypesenseDoc) -> Result<(), SearchError> {
        let path = format!("collections/{}/documents?action=upsert", collection);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::POST, &path, Some(serde_json::to_value(doc).unwrap())).await?;
            debug!("Upserted document to collection: {}", collection);
            Ok(())
        })
    }

    pub fn bulk_upsert(&self, collection: &str, docs: Vec<TypesenseDoc>) -> Result<(), SearchError> {
        if docs.is_empty() {
            return Ok(());
        }

        let path = format!("collections/{}/documents/import?action=upsert", collection);
        
        // Convert to JSONL format
        let mut body = String::new();
        for doc in &docs {
            let doc_json = serde_json::to_string(&doc)
                .map_err(|e| SearchError::Internal(format!("JSON serialization error: {}", e)))?;
            body.push_str(&doc_json);
            body.push('\n');
        }

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let url = format!("{}/{}", self.base_url, path);
            let request = self.client.post(&url)
                .header("X-TYPESENSE-API-KEY", &self.api_key)
                .header("Content-Type", "application/octet-stream")
                .body(body);

            let response = request.send().await
                .map_err(|e| SearchError::Internal(format!("Bulk request failed: {}", e)))?;

            if !response.status().is_success() {
                let error_text = response.text().await.unwrap_or_default();
                return Err(SearchError::Internal(format!("Bulk upsert failed: {}", error_text)));
            }

            debug!("Bulk upserted {} documents to {}", docs.len(), collection);
            Ok(())
        })
    }

    pub fn delete_document(&self, collection: &str, id: &str) -> Result<(), SearchError> {
        let path = format!("collections/{}/documents/{}", collection, id);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            self.request(reqwest::Method::DELETE, &path, None).await?;
            debug!("Deleted document: {}/{}", collection, id);
            Ok(())
        })
    }

    pub fn bulk_delete(&self, collection: &str, ids: Vec<String>) -> Result<(), SearchError> {
        if ids.is_empty() {
            return Ok(());
        }

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            for id in ids {
                let path = format!("collections/{}/documents/{}", collection, id);
                self.request(reqwest::Method::DELETE, &path, None).await?;
            }
            debug!("Bulk deleted documents from {}", collection);
            Ok(())
        })
    }

    pub fn get_document(&self, collection: &str, id: &str) -> Result<Option<Doc>, SearchError> {
        let path = format!("collections/{}/documents/{}", collection, id);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            match self.request(reqwest::Method::GET, &path, None).await {
                Ok(response) => {
                    let doc_response: GetDocumentResponse = response.json().await
                        .map_err(|e| SearchError::Internal(format!("Failed to parse response: {}", e)))?;
                    
                    Ok(Some(Doc {
                        id: doc_response.id,
                        content: serde_json::to_string(&doc_response.fields)
                            .map_err(|e| SearchError::Internal(format!("JSON serialization error: {}", e)))?,
                    }))
                }
                Err(SearchError::IndexNotFound) => Ok(None),
                Err(SearchError::Internal(msg)) if msg.contains("404") => Ok(None),
                Err(e) => Err(e),
            }
        })
    }

    pub fn search(&self, collection: &str, params: &HashMap<String, String>) -> Result<TypesenseSearchResponse, SearchError> {
        let path = format!("collections/{}/documents/search", collection);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let url = format!("{}/{}?{}", self.base_url, path, 
                params.iter().map(|(k, v)| format!("{}={}", k, urlencoding::encode(v))).collect::<Vec<_>>().join("&"));

            let request = self.client.get(&url)
                .header("X-TYPESENSE-API-KEY", &self.api_key);

            let response = request.send().await
                .map_err(|e| SearchError::Internal(format!("Search request failed: {}", e)))?;

            if response.status().is_success() {
                let search_response: TypesenseSearchResponse = response.json().await
                    .map_err(|e| SearchError::Internal(format!("Failed to parse search response: {}", e)))?;
                
                debug!("Search completed in {}ms", search_response.search_time_ms);
                Ok(search_response)
            } else {
                let status = response.status().as_u16();
                let error_text = response.text().await.unwrap_or_default();
                Err(SearchError::Internal(format!("Search failed HTTP {}: {}", status, error_text)))
            }
        })
    }

    pub fn export(&self, collection: &str, params: &HashMap<String, String>) -> Result<Vec<Doc>, SearchError> {
        let path = format!("collections/{}/documents/export", collection);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let url = format!("{}/{}?{}", self.base_url, path, 
                params.iter().map(|(k, v)| format!("{}={}", k, urlencoding::encode(v))).collect::<Vec<_>>().join("&"));

            let request = self.client.get(&url)
                .header("X-TYPESENSE-API-KEY", &self.api_key);

            let response = request.send().await
                .map_err(|e| SearchError::Internal(format!("Export request failed: {}", e)))?;

            if response.status().is_success() {
                let response_text = response.text().await
                    .map_err(|e| SearchError::Internal(format!("Failed to get response text: {}", e)))?;
                
                let mut docs = Vec::new();
                for line in response_text.lines() {
                    if !line.trim().is_empty() {
                        let doc_data: Value = serde_json::from_str(line)
                            .map_err(|e| SearchError::Internal(format!("Failed to parse export line: {}", e)))?;
                        
                        if let Some(id) = doc_data.get("id").and_then(|v| v.as_str()) {
                            docs.push(Doc {
                                id: id.to_string(),
                                content: serde_json::to_string(&doc_data)
                                    .map_err(|e| SearchError::Internal(format!("JSON serialization error: {}", e)))?,
                            });
                        }
                    }
                }
                
                Ok(docs)
            } else {
                let status = response.status().as_u16();
                let error_text = response.text().await.unwrap_or_default();
                Err(SearchError::Internal(format!("Export failed HTTP {}: {}", status, error_text)))
            }
        })
    }

    pub fn get_collection_schema(&self, collection: &str) -> Result<Schema, SearchError> {
        let path = format!("collections/{}", collection);
        
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| SearchError::Internal(format!("Runtime error: {}", e)))?;
        
        rt.block_on(async {
            let response = self.request(reqwest::Method::GET, &path, None).await?;
            let collection_info: CollectionResponse = response.json().await
                .map_err(|e| SearchError::Internal(format!("Failed to parse collection response: {}", e)))?;
            
            Ok(crate::conversions::typesense_fields_to_schema(&collection_info.fields))
        })
    }

    pub fn update_collection_schema(&self, collection: &str, schema: &Schema) -> Result<(), SearchError> {
        // Typesense doesn't support direct schema updates, would need to recreate collection
        // For now, return an error suggesting recreation
        Err(SearchError::Internal(
            "Typesense doesn't support schema updates. Please recreate the collection with the new schema.".to_string()
        ))
    }
}
