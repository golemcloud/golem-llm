use base64::{engine::general_purpose, Engine as _};
use futures::executor::block_on;
use golem_graph::golem::graph::errors::GraphError;
use golem_graph::golem::graph::schema::{
    ContainerInfo, ContainerType, EdgeTypeDefinition, IndexDefinition, IndexType,
};
use reqwest::header::{HeaderValue, AUTHORIZATION};
use reqwest::{Client, RequestBuilder, Response, StatusCode};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};

pub struct ArangoDbApi {
    base_url: String,
    client: Client,
}

impl ArangoDbApi {
    pub fn new(host: &str, port: u16, username: &str, password: &str, database_name: &str) -> Self {
        let base_url = format!("http://{}:{}/_db/{}", host, port, database_name);
        let mut headers = reqwest::header::HeaderMap::new();
        let auth_header = format!(
            "Basic {}",
            general_purpose::STANDARD.encode(format!("{}:{}", username, password))
        );
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_header).unwrap());

        let client = Client::builder().default_headers(headers).build().unwrap();

        Self { base_url, client }
    }

    fn post(&self, endpoint: &str) -> RequestBuilder {
        self.client.post(format!("{}{}", self.base_url, endpoint))
    }

    fn get(&self, endpoint: &str) -> RequestBuilder {
        self.client.get(format!("{}{}", self.base_url, endpoint))
    }

    fn put(&self, endpoint: &str) -> RequestBuilder {
        self.client.put(format!("{}{}", self.base_url, endpoint))
    }

    fn delete(&self, endpoint: &str) -> RequestBuilder {
        self.client.delete(format!("{}{}", self.base_url, endpoint))
    }

    async fn execute_async<T: DeserializeOwned>(
        &self,
        request: RequestBuilder,
    ) -> Result<T, GraphError> {
        let response = request
            .send()
            .map_err(|e| GraphError::ConnectionFailed(format!("Failed to send request: {}", e)))?;
        self.handle_response_async(response).await
    }

    async fn handle_response_async<T: DeserializeOwned>(
        &self,
        response: Response,
    ) -> Result<T, GraphError> {
        let status = response.status();
        let response_body: Value = response.json().map_err(|e| {
            GraphError::InternalError(format!("Failed to parse response body: {}", e))
        })?;

        if status.is_success() {
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
            let error_msg = response_body
                .get("errorMessage")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");
            Err(self.map_error(status, error_msg))
        }
    }

    fn map_error(&self, status: StatusCode, message: &str) -> GraphError {
        match status {
            StatusCode::UNAUTHORIZED => GraphError::AuthenticationFailed(message.to_string()),
            StatusCode::FORBIDDEN => GraphError::AuthorizationFailed(message.to_string()),
            StatusCode::NOT_FOUND => {
                GraphError::InternalError(format!("Endpoint not found: {}", message))
            } // This might need more specific handling
            StatusCode::CONFLICT => GraphError::TransactionConflict,
            _ => GraphError::InternalError(format!("ArangoDB error: {} - {}", status, message)),
        }
    }

    pub fn begin_transaction(&self, read_only: bool) -> Result<String, GraphError> {
        block_on(async {
            let collections = if read_only {
                json!({ "read": [] })
            } else {
                json!({ "write": [] })
            };

            let body = json!({ "collections": collections });
            let request = self.post("/_api/transaction/begin").json(&body);
            let result: Value = self.execute_async(request).await?;

            result
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| {
                    GraphError::InternalError("Missing transaction ID in response".to_string())
                })
        })
    }

    pub fn commit_transaction(&self, transaction_id: &str) -> Result<(), GraphError> {
        block_on(async {
            let endpoint = format!("/_api/transaction/{}", transaction_id);
            let request = self.put(&endpoint);
            let _: Value = self.execute_async(request).await?;
            Ok(())
        })
    }

    pub fn rollback_transaction(&self, transaction_id: &str) -> Result<(), GraphError> {
        block_on(async {
            let endpoint = format!("/_api/transaction/{}", transaction_id);
            let request = self.delete(&endpoint);
            let _: Value = self.execute_async(request).await?;
            Ok(())
        })
    }

    pub fn execute_in_transaction(
        &self,
        transaction_id: &str,
        query: Value,
    ) -> Result<Value, GraphError> {
        block_on(async {
            let request = self
                .post("/_api/cursor")
                .header("x-arango-trx-id", transaction_id)
                .json(&query);
            self.execute_async(request).await
        })
    }

    pub fn ping(&self) -> Result<(), GraphError> {
        block_on(async {
            let request = self.get("/_api/version");
            let _: Value = self.execute_async(request).await?;
            Ok(())
        })
    }

    // Schema operations
    pub fn create_collection(
        &self,
        name: &str,
        container_type: ContainerType,
    ) -> Result<(), GraphError> {
        block_on(async {
            let collection_type = match container_type {
                ContainerType::VertexContainer => 2,
                ContainerType::EdgeContainer => 3,
            };
            let body = json!({ "name": name, "type": collection_type });
            let request = self.post("/_api/collection").json(&body);
            let _: Value = self.execute_async(request).await?;
            Ok(())
        })
    }

    pub fn list_collections(&self) -> Result<Vec<ContainerInfo>, GraphError> {
        block_on(async {
            let request = self.get("/_api/collection");
            let response: Value = self.execute_async(request).await?;
            let collections = response["collections"]
                .as_array()
                .ok_or_else(|| {
                    GraphError::InternalError("Invalid response for list_collections".to_string())
                })?
                .iter()
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
        })
    }

    pub fn create_index(
        &self,
        collection: String,
        fields: Vec<String>,
        unique: bool,
        index_type: IndexType,
    ) -> Result<(), GraphError> {
        block_on(async {
            let type_str = match index_type {
                IndexType::Exact => "persistent",
                IndexType::Range => "persistent", // ArangoDB's persistent index supports range queries
                IndexType::Text => "inverted", // Full-text requires enterprise edition or arangosearch
                IndexType::Geospatial => "geo",
            };

            let body = json!({
                "type": type_str,
                "fields": fields,
                "unique": unique,
            });

            let request = self
                .post(&format!("/_api/index?collection={}", collection))
                .json(&body);
            let _: Value = self.execute_async(request).await?;
            Ok(())
        })
    }

    pub fn drop_index(&self, name: &str) -> Result<(), GraphError> {
        block_on(async {
            let request = self.delete(&format!("/_api/index/{}", name));
            let _: Value = self.execute_async(request).await?;
            Ok(())
        })
    }

    pub fn list_indexes(&self) -> Result<Vec<IndexDefinition>, GraphError> {
        Err(GraphError::UnsupportedOperation(
            "list_indexes is not yet supported".to_string(),
        ))
    }

    pub fn get_index(&self, _name: &str) -> Result<Option<IndexDefinition>, GraphError> {
        Err(GraphError::UnsupportedOperation(
            "get_index is not yet supported".to_string(),
        ))
    }

    pub fn define_edge_type(&self, _definition: EdgeTypeDefinition) -> Result<(), GraphError> {
        Err(GraphError::UnsupportedOperation(
            "define_edge_type is not yet fully supported".to_string(),
        ))
    }

    pub fn list_edge_types(&self) -> Result<Vec<EdgeTypeDefinition>, GraphError> {
        Err(GraphError::UnsupportedOperation("ArangoDB does not have explicit edge type definitions in the same way as some other graph DBs.".to_string()))
    }

    pub fn get_transaction_status(&self, transaction_id: &str) -> Result<String, GraphError> {
        block_on(async {
            let endpoint = format!("/_api/transaction/{}", transaction_id);
            let request = self.get(&endpoint);

            let response: TransactionStatusResponse = self.execute_async(request).await?;
            Ok(response.status)
        })
    }

    pub fn get_database_statistics(&self) -> Result<DatabaseStatistics, GraphError> {
        block_on(async {
            let collections: ListCollectionsResponse = self
                .execute_async(self.get("/_api/collection?excludeSystem=true"))
                .await?;

            let mut total_vertex_count = 0;
            let mut total_edge_count = 0;

            for collection_info in collections.result {
                let properties_endpoint =
                    format!("/_api/collection/{}/properties", collection_info.name);
                let properties: CollectionPropertiesResponse =
                    self.execute_async(self.get(&properties_endpoint)).await?;

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
