use golem_graph::golem::graph::errors::GraphError;
use reqwest::Client;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::Value;

const NEO4J_TRANSACTION_ENDPOINT: &str = "/db/data/transaction";

#[derive(Clone)]
pub(crate) struct Neo4jApi {
    base_url: String,
    auth_header: String,
    client: Client,
}

impl Neo4jApi {
    pub(crate) fn new(host: &str, port: u16, username: &str, password: &str) -> Self {
        let base_url = format!("http://{}:{}", host, port);
        let auth = format!("{}:{}", username, password);
        let auth_header = format!("Basic {}", STANDARD.encode(auth.as_bytes()));
        let client = Client::new();
        Neo4jApi {
            base_url,
            auth_header,
            client,
        }
    }

    pub(crate) fn begin_transaction(&self) -> Result<String, GraphError> {
        let url = format!("{}{}", self.base_url, NEO4J_TRANSACTION_ENDPOINT);
        let response = self
            .client
            .post(&url)
            .header("Authorization", &self.auth_header)
            .header("Content-Type", "application/json")
            .body("{}")
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;

        if response.status().is_success() {
            let location = response
                .headers()
                .get("Location")
                .and_then(|h| h.to_str().ok())
                .map(|s| s.to_string())
                .ok_or_else(|| {
                    GraphError::InternalError(
                        "No location header in begin transaction response".to_string(),
                    )
                })?;
            Ok(location)
        } else {
            let error: Value = response
                .json()
                .map_err(|e| GraphError::InternalError(e.to_string()))?;
            Err(GraphError::TransactionFailed(error.to_string()))
        }
    }

    pub(crate) fn execute_in_transaction(
        &self,
        transaction_url: &str,
        statements: Value,
    ) -> Result<Value, GraphError> {
        let response = self
            .client
            .post(transaction_url)
            .header("Authorization", &self.auth_header)
            .header("Content-Type", "application/json")
            .body(statements.to_string())
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;

        if response.status().is_success() {
            response
                .json()
                .map_err(|e| GraphError::InternalError(e.to_string()))
        } else {
            let error: Value = response
                .json()
                .map_err(|e| GraphError::InternalError(e.to_string()))?;
            Err(GraphError::TransactionFailed(error.to_string()))
        }
    }

    pub(crate) fn commit_transaction(&self, transaction_url: &str) -> Result<(), GraphError> {
        let commit_url = format!("{}/commit", transaction_url);
        let response = self
            .client
            .post(&commit_url)
            .header("Authorization", &self.auth_header)
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let error: Value = response
                .json()
                .map_err(|e| GraphError::InternalError(e.to_string()))?;
            Err(GraphError::TransactionFailed(error.to_string()))
        }
    }

    pub(crate) fn rollback_transaction(&self, transaction_url: &str) -> Result<(), GraphError> {
        let response = self
            .client
            .delete(transaction_url)
            .header("Authorization", &self.auth_header)
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let error: Value = response
                .json()
                .map_err(|e| GraphError::InternalError(e.to_string()))?;
            Err(GraphError::TransactionFailed(error.to_string()))
        }
    }
}
