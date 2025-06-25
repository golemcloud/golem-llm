use base64::{engine::general_purpose::STANDARD, Engine as _};
use golem_graph::golem::graph::errors::GraphError;
use reqwest::{Client, Response};
use serde_json::Value;

#[derive(Clone)]
pub(crate) struct Neo4jApi {
    base_url: String,
    database: String,
    auth_header: String,
    client: Client,
}

impl Neo4jApi {
    pub(crate) fn new(
        host: &str,
        port: u16,
        database: &str,
        username: &str,
        password: &str,
    ) -> Self {
        let base_url = format!("http://{}:{}", host, port);
        let auth = format!("{}:{}", username, password);
        let auth_header = format!("Basic {}", STANDARD.encode(auth.as_bytes()));
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");

        Neo4jApi {
            base_url,
            database: database.to_string(),
            auth_header,
            client,
        }
    }

    fn tx_endpoint(&self) -> String {
        format!("/db/{}/tx", self.database)
    }

    pub(crate) fn begin_transaction(&self) -> Result<String, GraphError> {
        let url = format!("{}{}", self.base_url, self.tx_endpoint());
        let resp = self
            .client
            .post(&url)
            .header("Authorization", &self.auth_header)
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;
        Self::ensure_success_and_get_location(resp)
    }

    pub(crate) fn execute_in_transaction(
        &self,
        tx_url: &str,
        statements: Value,
    ) -> Result<Value, GraphError> {
        println!("[Neo4jApi] Cypher request: {}", statements);
        let resp = self
            .client
            .post(tx_url)
            .header("Authorization", &self.auth_header)
            .header("Content-Type", "application/json")
            .body(statements.to_string())
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;
        let json = Self::ensure_success_and_json(resp)?;
        println!("[Neo4jApi] Cypher response: {}", json);
        Ok(json)
    }

    pub(crate) fn commit_transaction(&self, tx_url: &str) -> Result<(), GraphError> {
        let commit_url = format!("{}/commit", tx_url);
        let resp = self
            .client
            .post(&commit_url)
            .header("Authorization", &self.auth_header)
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;
        Self::ensure_success(resp).map(|_| ())
    }

    pub(crate) fn rollback_transaction(&self, tx_url: &str) -> Result<(), GraphError> {
        let resp = self
            .client
            .delete(tx_url)
            .header("Authorization", &self.auth_header)
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;
        Self::ensure_success(resp).map(|_| ())
    }

    pub(crate) fn get_transaction_status(&self, tx_url: &str) -> Result<String, GraphError> {
        let resp = self
            .client
            .get(tx_url)
            .header("Authorization", &self.auth_header)
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;

        if resp.status().is_success() {
            Ok("running".to_string())
        } else {
            Ok("closed".to_string())
        }
    }

    // Helpers

    fn ensure_success(response: Response) -> Result<Response, GraphError> {
        if response.status().is_success() {
            Ok(response)
        } else {
            let text = response
                .text()
                .map_err(|e| GraphError::InternalError(e.to_string()))?;
            let err: Value = serde_json::from_str(&text)
                .map_err(|e| GraphError::InternalError(e.to_string()))?;
            Err(GraphError::TransactionFailed(err.to_string()))
        }
    }

    fn ensure_success_and_json(response: Response) -> Result<Value, GraphError> {
        if response.status().is_success() {
            response
                .json()
                .map_err(|e| GraphError::InternalError(e.to_string()))
        } else {
            let text = response
                .text()
                .map_err(|e| GraphError::InternalError(e.to_string()))?;
            let err: Value = serde_json::from_str(&text)
                .map_err(|e| GraphError::InternalError(e.to_string()))?;
            Err(GraphError::TransactionFailed(err.to_string()))
        }
    }

    fn ensure_success_and_get_location(response: Response) -> Result<String, GraphError> {
        if response.status().is_success() {
            response
                .headers()
                .get("Location")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
                .ok_or_else(|| GraphError::InternalError("Missing Location header".into()))
        } else {
            let text = response
                .text()
                .map_err(|e| GraphError::InternalError(e.to_string()))?;
            let err: Value = serde_json::from_str(&text)
                .map_err(|e| GraphError::InternalError(e.to_string()))?;
            Err(GraphError::TransactionFailed(err.to_string()))
        }
    }
}
