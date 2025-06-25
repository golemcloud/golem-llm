use base64::{engine::general_purpose, Engine as _};
use golem_graph::golem::graph::errors::GraphError;
use golem_graph::golem::graph::schema::{
    ContainerInfo, ContainerType, EdgeTypeDefinition, IndexDefinition, IndexType,
};
use reqwest::{Client, Method, Response};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};

pub struct ArangoDbApi {
    base_url: String,
    client: Client,
    auth_header: String,
}

impl ArangoDbApi {
    pub fn new(host: &str, port: u16, username: &str, password: &str, database_name: &str) -> Self {
        let base_url = format!("http://{}:{}/_db/{}", host, port, database_name);
        let auth_header = format!(
            "Basic {}",
            general_purpose::STANDARD.encode(format!("{}:{}", username, password))
        );

        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");

        Self {
            base_url,
            client,
            auth_header,
        }
    }

    fn execute<T: DeserializeOwned>(
        &self,
        method: Method,
        endpoint: &str,
        body: Option<&Value>,
    ) -> Result<T, GraphError> {
        let url = format!("{}{}", self.base_url, endpoint);

        let mut request_builder = self
            .client
            .request(method, url)
            .header("authorization", &self.auth_header);

        if let Some(body_value) = body {
            let body_string = serde_json::to_string(body_value).map_err(|e| {
                GraphError::InternalError(format!("Failed to serialize request body: {}", e))
            })?;

            request_builder = request_builder
                .header("content-type", "application/json")
                .header("content-length", body_string.len().to_string())
                .body(body_string);
        }

        let response = request_builder.send().map_err(|e| {
            GraphError::ConnectionFailed(e.to_string() + " - Failed to send request")
        })?;

        self.handle_response(response)
    }

    fn handle_response<T: DeserializeOwned>(&self, response: Response) -> Result<T, GraphError> {
        let status = response.status();
        let status_code = status.as_u16();

        if status.is_success() {
            let response_body: Value = response.json().map_err(|e| {
                GraphError::InternalError(format!("Failed to parse response body: {}", e))
            })?;

            if let Some(result) = response_body.get("result") {
                serde_json::from_value(result.clone()).map_err(|e| {
                    GraphError::InternalError(format!(
                        "Failed to deserialize successful response: {}",
                        e
                    ))
                })
            } else {
                serde_json::from_value(response_body).map_err(|e| {
                    GraphError::InternalError(format!(
                        "Failed to deserialize successful response: {}",
                        e
                    ))
                })
            }
        } else {
            let error_body: Value = response.json().map_err(|e| {
                GraphError::InternalError(format!("Failed to read error response: {}", e))
            })?;

            let error_msg = error_body
                .get("errorMessage")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");
            Err(self.map_error(status_code, error_msg))
        }
    }

    fn map_error(&self, status: u16, message: &str) -> GraphError {
        match status {
            401 => GraphError::AuthenticationFailed(message.to_string()),
            403 => GraphError::AuthorizationFailed(message.to_string()),
            404 => GraphError::InternalError(format!("Endpoint not found: {}", message)),
            409 => GraphError::TransactionConflict,
            _ => GraphError::InternalError(format!("ArangoDB error: {} - {}", status, message)),
        }
    }

    #[allow(dead_code)]
    pub fn begin_transaction(&self, read_only: bool) -> Result<String, GraphError> {
        let existing_collections = self.list_collections().unwrap_or_default();
        let collection_names: Vec<String> = existing_collections
            .iter()
            .map(|c| c.name.clone())
            .collect();

        let collections = if read_only {
            json!({ "read": collection_names })
        } else {
            json!({ "write": collection_names })
        };

        let body = json!({ "collections": collections });
        let result: Value = self.execute(Method::POST, "/_api/transaction/begin", Some(&body))?;

        result
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                GraphError::InternalError("Missing transaction ID in response".to_string())
            })
    }

    #[allow(dead_code)]
    pub fn begin_transaction_with_collections(
        &self,
        read_only: bool,
        collections: Vec<String>,
    ) -> Result<String, GraphError> {
        let collections_spec = if read_only {
            json!({ "read": collections })
        } else {
            json!({ "write": collections })
        };

        let body = json!({ "collections": collections_spec });
        let result: Value = self.execute(Method::POST, "/_api/transaction/begin", Some(&body))?;

        result
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                GraphError::InternalError("Missing transaction ID in response".to_string())
            })
    }

    pub fn commit_transaction(&self, transaction_id: &str) -> Result<(), GraphError> {
        let endpoint = format!("/_api/transaction/{}", transaction_id);
        let _: Value = self.execute(Method::PUT, &endpoint, None)?;
        Ok(())
    }

    pub fn rollback_transaction(&self, transaction_id: &str) -> Result<(), GraphError> {
        let endpoint = format!("/_api/transaction/{}", transaction_id);
        let _: Value = self.execute(Method::DELETE, &endpoint, None)?;
        Ok(())
    }

    pub fn execute_in_transaction(
        &self,
        transaction_id: &str,
        query: Value,
    ) -> Result<Value, GraphError> {
        let url = format!("{}/_api/cursor", self.base_url);

        let body_string = serde_json::to_string(&query)
            .map_err(|e| GraphError::InternalError(format!("Failed to serialize query: {}", e)))?;

        let response = self
            .client
            .request(Method::POST, url)
            .header("authorization", &self.auth_header)
            .header("content-type", "application/json")
            .header("content-length", body_string.len().to_string())
            .header("x-arango-trx-id", transaction_id)
            .body(body_string)
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;

        self.handle_response(response)
    }

    pub fn ping(&self) -> Result<(), GraphError> {
        let _: Value = self.execute(Method::GET, "/_api/version", None)?;
        Ok(())
    }

    // Schema operations
    pub fn create_collection(
        &self,
        name: &str,
        container_type: ContainerType,
    ) -> Result<(), GraphError> {
        let collection_type = match container_type {
            ContainerType::VertexContainer => 2,
            ContainerType::EdgeContainer => 3,
        };
        let body = json!({ "name": name, "type": collection_type });
        let _: Value = self.execute(Method::POST, "/_api/collection", Some(&body))?;
        Ok(())
    }

    pub fn list_collections(&self) -> Result<Vec<ContainerInfo>, GraphError> {
        let response: Value = self.execute(Method::GET, "/_api/collection", None)?;

        let collections_array = if let Some(result) = response.get("result") {
            result.as_array().ok_or_else(|| {
                GraphError::InternalError(
                    "Invalid response for list_collections - result is not array".to_string(),
                )
            })?
        } else {
            response.as_array().ok_or_else(|| {
                GraphError::InternalError("Invalid response for list_collections - no result field and response is not array".to_string())
            })?
        };

        let collections = collections_array
            .iter()
            .filter(|v| !v["isSystem"].as_bool().unwrap_or(false)) // Filter out system collections
            .map(|v| {
                let name = v["name"].as_str().unwrap_or_default().to_string();
                let coll_type = v["type"].as_u64().unwrap_or(2);
                let container_type = if coll_type == 3 {
                    ContainerType::EdgeContainer
                } else {
                    ContainerType::VertexContainer
                };
                ContainerInfo {
                    name,
                    container_type,
                    element_count: None,
                }
            })
            .collect();
        Ok(collections)
    }

    pub fn create_index(
        &self,
        collection: String,
        fields: Vec<String>,
        unique: bool,
        index_type: IndexType,
        name: Option<String>,
    ) -> Result<(), GraphError> {
        let type_str = match index_type {
            IndexType::Exact => "persistent",
            IndexType::Range => "persistent", // ArangoDB's persistent index supports range queries
            IndexType::Text => "inverted", // Full-text requires enterprise edition or arangosearch
            IndexType::Geospatial => "geo",
        };

        let mut body = json!({
            "type": type_str,
            "fields": fields,
            "unique": unique,
        });

        // Add name if provided
        if let Some(index_name) = name {
            body["name"] = json!(index_name);
        }

        let endpoint = format!("/_api/index?collection={}", collection);
        let _: Value = self.execute(Method::POST, &endpoint, Some(&body))?;
        Ok(())
    }

    pub fn drop_index(&self, name: &str) -> Result<(), GraphError> {
        // First, find the index by name to get its ID
        let collections = self.list_collections()?;

        for collection in collections {
            let endpoint = format!("/_api/index?collection={}", collection.name);

            if let Ok(response) = self.execute::<Value>(Method::GET, &endpoint, None) {
                if let Some(indexes) = response["indexes"].as_array() {
                    for idx in indexes {
                        if let Some(idx_name) = idx["name"].as_str() {
                            if idx_name == name {
                                if let Some(idx_id) = idx["id"].as_str() {
                                    let delete_endpoint = format!("/_api/index/{}", idx_id);
                                    let _: Value =
                                        self.execute(Method::DELETE, &delete_endpoint, None)?;
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
            }
        }

        Err(GraphError::InternalError(format!(
            "Index '{}' not found",
            name
        )))
    }

    pub fn list_indexes(&self) -> Result<Vec<IndexDefinition>, GraphError> {
        // Get all collections first
        let collections = self.list_collections()?;
        let mut all_indexes = Vec::new();

        for collection in collections {
            let endpoint = format!("/_api/index?collection={}", collection.name);

            match self.execute::<Value>(Method::GET, &endpoint, None) {
                Ok(response) => {
                    if let Some(indexes) = response["indexes"].as_array() {
                        for index in indexes {
                            // Skip primary and edge indexes
                            if let Some(index_type) = index["type"].as_str() {
                                if index_type == "primary" || index_type == "edge" {
                                    continue;
                                }
                            }

                            let name = index["name"].as_str().unwrap_or("").to_string();
                            let id = index["id"].as_str().unwrap_or("").to_string();

                            let fields: Vec<String> = index["fields"]
                                .as_array()
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|f| f.as_str())
                                        .map(String::from)
                                        .collect()
                                })
                                .unwrap_or_default();

                            if fields.is_empty() {
                                continue;
                            }

                            let unique = index["unique"].as_bool().unwrap_or(false);
                            let index_type_str = index["type"].as_str().unwrap_or("persistent");
                            let index_type = match index_type_str {
                                "geo" => golem_graph::golem::graph::schema::IndexType::Geospatial,
                                "inverted" => golem_graph::golem::graph::schema::IndexType::Text,
                                _ => golem_graph::golem::graph::schema::IndexType::Exact,
                            };

                            // Use a combination of collection and fields as logical name for matching
                            let logical_name = if fields.len() == 1 {
                                format!("idx_{}_{}", collection.name, fields[0])
                            } else {
                                format!("idx_{}_{}", collection.name, fields.join("_"))
                            };

                            // Prefer the ArangoDB generated name, but fall back to our logical name
                            let final_name = if !name.is_empty() {
                                name
                            } else if !id.is_empty() {
                                id
                            } else {
                                logical_name
                            };

                            all_indexes.push(IndexDefinition {
                                name: final_name,
                                label: collection.name.clone(),
                                container: Some(collection.name.clone()),
                                properties: fields,
                                unique,
                                index_type,
                            });
                        }
                    }
                }
                Err(_) => {
                    continue;
                }
            }
        }

        Ok(all_indexes)
    }

    pub fn get_index(&self, name: &str) -> Result<Option<IndexDefinition>, GraphError> {
        let all_indexes = self.list_indexes()?;

        if let Some(index) = all_indexes.iter().find(|idx| idx.name == name) {
            return Ok(Some(index.clone()));
        }

        // If the requested name follows our pattern (idx_collection_field)
        if name.starts_with("idx_") {
            let parts: Vec<&str> = name.split('_').collect();
            if parts.len() >= 3 {
                let collection_part = parts[1];
                let field_part = parts[2..].join("_");

                if let Some(index) = all_indexes.iter().find(|idx| {
                    idx.label == collection_part
                        && idx.properties.len() == 1
                        && idx.properties[0] == field_part
                }) {
                    return Ok(Some(index.clone()));
                }
            }
        }

        Ok(None)
    }

    pub fn define_edge_type(&self, definition: EdgeTypeDefinition) -> Result<(), GraphError> {
        self.create_collection(&definition.collection, ContainerType::EdgeContainer)?;
        // Note: ArangoDB doesn't enforce from/to collection constraints like some other graph databases
        // The constraints in EdgeTypeDefinition are mainly for application-level validation
        Ok(())
    }

    pub fn list_edge_types(&self) -> Result<Vec<EdgeTypeDefinition>, GraphError> {
        // In ArangoDB, we return edge collections as edge types
        // Since ArangoDB doesn't enforce from/to constraints at the DB level,
        // we return edge collections with empty from/to collections
        let collections = self.list_collections()?;
        let edge_types = collections
            .into_iter()
            .filter(|c| matches!(c.container_type, ContainerType::EdgeContainer))
            .map(|c| EdgeTypeDefinition {
                collection: c.name,
                from_collections: vec![], // ArangoDB doesn't store these constraints
                to_collections: vec![],   // ArangoDB doesn't store these constraints
            })
            .collect();
        Ok(edge_types)
    }

    pub fn get_transaction_status(&self, transaction_id: &str) -> Result<String, GraphError> {
        let endpoint = format!("/_api/transaction/{}", transaction_id);
        let response: TransactionStatusResponse = self.execute(Method::GET, &endpoint, None)?;
        Ok(response.status)
    }

    pub fn get_database_statistics(&self) -> Result<DatabaseStatistics, GraphError> {
        let collections: ListCollectionsResponse =
            self.execute(Method::GET, "/_api/collection?excludeSystem=true", None)?;

        let mut total_vertex_count = 0;
        let mut total_edge_count = 0;

        for collection_info in collections.result {
            let properties_endpoint =
                format!("/_api/collection/{}/properties", collection_info.name);
            let properties: CollectionPropertiesResponse =
                self.execute(Method::GET, &properties_endpoint, None)?;

            if properties.collection_type == ArangoCollectionType::Edge {
                total_edge_count += properties.count;
            } else {
                total_vertex_count += properties.count;
            }
        }

        Ok(DatabaseStatistics {
            vertex_count: total_vertex_count,
            edge_count: total_edge_count,
        })
    }

    #[allow(dead_code)]
    pub fn execute_query(&self, query: Value) -> Result<Value, GraphError> {
        self.execute(Method::POST, "/_api/cursor", Some(&query))
    }

    #[allow(dead_code)]
    pub fn ensure_collection_exists(
        &self,
        name: &str,
        container_type: ContainerType,
    ) -> Result<(), GraphError> {
        match self.create_collection(name, container_type) {
            Ok(_) => Ok(()),
            Err(GraphError::InternalError(msg)) if msg.contains("duplicate name") => Ok(()),
            Err(e) => Err(e),
        }
    }

    pub fn begin_dynamic_transaction(&self, read_only: bool) -> Result<String, GraphError> {
        let common_collections = vec![
            "Person".to_string(),
            "TempUser".to_string(),
            "Company".to_string(),
            "Employee".to_string(),
            "Node".to_string(),
            "Product".to_string(),
            "User".to_string(),
            "KNOWS".to_string(),
            "WORKS_FOR".to_string(),
            "CONNECTS".to_string(),
            "FOLLOWS".to_string(),
        ];

        let existing_collections = self.list_collections().unwrap_or_default();
        let mut all_collections: Vec<String> = existing_collections
            .iter()
            .map(|c| c.name.clone())
            .collect();

        // Add common collections that might not exist yet
        for common in common_collections {
            if !all_collections.contains(&common) {
                all_collections.push(common);
            }
        }

        let collections = if read_only {
            json!({ "read": all_collections })
        } else {
            json!({ "write": all_collections })
        };

        let body = json!({ "collections": collections });
        let result: Value = self.execute(Method::POST, "/_api/transaction/begin", Some(&body))?;

        result
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                GraphError::InternalError("Missing transaction ID in response".to_string())
            })
    }
}

#[derive(serde::Deserialize, Debug)]
struct TransactionStatusResponse {
    #[serde(rename = "id")]
    _id: String,
    status: String,
}

#[derive(serde::Deserialize, Debug)]
pub struct DatabaseStatistics {
    pub vertex_count: u64,
    pub edge_count: u64,
}

#[derive(serde::Deserialize, Debug)]
struct ListCollectionsResponse {
    result: Vec<CollectionInfoShort>,
}

#[derive(serde::Deserialize, Debug)]
struct CollectionInfoShort {
    name: String,
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct CollectionPropertiesResponse {
    count: u64,
    #[serde(rename = "type")]
    collection_type: ArangoCollectionType,
}

#[derive(Debug, PartialEq)]
enum ArangoCollectionType {
    Document,
    Edge,
    Unknown(u8),
}

impl From<u8> for ArangoCollectionType {
    fn from(value: u8) -> Self {
        match value {
            2 => ArangoCollectionType::Document,
            3 => ArangoCollectionType::Edge,
            _ => ArangoCollectionType::Unknown(value),
        }
    }
}

impl<'de> serde::Deserialize<'de> for ArangoCollectionType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        Ok(ArangoCollectionType::from(value))
    }
}
