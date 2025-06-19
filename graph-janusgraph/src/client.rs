use golem_graph::golem::graph::errors::GraphError;
use reqwest::Client;
use serde_json::{json, Value};

#[derive(Clone)]
pub struct JanusGraphApi {
    endpoint: String,
    client: Client,
}

impl JanusGraphApi {
    pub fn new(
        host: &str,
        port: u16,
        _username: Option<&str>,
        _password: Option<&str>,
    ) -> Result<Self, GraphError> {
        let endpoint = format!("http://{}:{}/gremlin", host, port);
        let client = Client::new();
        Ok(JanusGraphApi { endpoint, client })
    }

    pub fn execute(&self, gremlin: &str, bindings: Option<Value>) -> Result<Value, GraphError> {
        let bindings = bindings.unwrap_or_else(|| json!({}));

        let request_body = json!({
            "gremlin": gremlin,
            "bindings": bindings
        });

        let response = self
            .client
            .post(&self.endpoint)
            .json(&request_body)
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;

        if response.status().is_success() {
            response
                .json()
                .map_err(|e| GraphError::InternalError(e.to_string()))
        } else {
            let status = response.status();
            let error_body = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(GraphError::InvalidQuery(format!(
                "Gremlin query failed with status {}: {}",
                status, error_body
            )))
        }
    }
}
