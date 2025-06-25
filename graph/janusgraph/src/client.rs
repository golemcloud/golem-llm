use golem_graph::golem::graph::errors::GraphError;
use reqwest::{Client, Response};
use serde_json::{json, Value};
use uuid::Uuid;

#[derive(Clone)]
pub struct JanusGraphApi {
    endpoint: String,
    client: Client,
    session_id: String,
}

impl JanusGraphApi {
    pub fn new(
        host: &str,
        port: u16,
        _username: Option<&str>,
        _password: Option<&str>,
    ) -> Result<Self, GraphError> {
        let endpoint = format!("http://{}:{}/gremlin", host, port);
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        let session_id = Uuid::new_v4().to_string();
        Ok(JanusGraphApi {
            endpoint,
            client,
            session_id,
        })
    }

    pub fn new_with_session(
        host: &str,
        port: u16,
        _username: Option<&str>,
        _password: Option<&str>,
        session_id: String,
    ) -> Result<Self, GraphError> {
        let endpoint = format!("http://{}:{}/gremlin", host, port);
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Ok(JanusGraphApi {
            endpoint,
            client,
            session_id,
        })
    }

    pub fn commit(&self) -> Result<(), GraphError> {
        self.execute("g.tx().commit()", None)?;
        self.execute("g.tx().open()", None)?;
        Ok(())
    }

    pub fn execute(&self, gremlin: &str, bindings: Option<Value>) -> Result<Value, GraphError> {
        let bindings = bindings.unwrap_or_else(|| json!({}));
        let request_body = json!({
            "gremlin": gremlin,
            "bindings": bindings,
            "session": self.session_id,
            "processor": "session",
            "op": "eval",

        });

        eprintln!("[JanusGraphApi] DEBUG - Full request details:");
        eprintln!("[JanusGraphApi] Endpoint: {}", self.endpoint);
        eprintln!("[JanusGraphApi] Session ID: {}", self.session_id);
        eprintln!("[JanusGraphApi] Gremlin Query: {}", gremlin);
        eprintln!(
            "[JanusGraphApi] Request Body: {}",
            serde_json::to_string_pretty(&request_body)
                .unwrap_or_else(|_| "Failed to serialize".to_string())
        );

        let body_string = serde_json::to_string(&request_body).map_err(|e| {
            GraphError::InternalError(format!("Failed to serialize request body: {}", e))
        })?;

        eprintln!(
            "[JanusGraphApi] Sending POST request to: {} with body length: {}",
            self.endpoint,
            body_string.len()
        );
        let response = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Content-Length", body_string.len().to_string())
            .body(body_string)
            .send()
            .map_err(|e| {
                eprintln!("[JanusGraphApi] ERROR - Request failed: {}", e);
                GraphError::ConnectionFailed(format!("reqwest error: {}", e))
            })?;

        eprintln!(
            "[JanusGraphApi] Got response with status: {}",
            response.status()
        );
        Self::handle_response(response)
    }

    fn _read(&self, gremlin: &str, bindings: Option<Value>) -> Result<Value, GraphError> {
        let bindings = bindings.unwrap_or_else(|| json!({}));
        let request_body = json!({
            "gremlin": gremlin,
            "bindings": bindings,
        });

        let body_string = serde_json::to_string(&request_body).map_err(|e| {
            GraphError::InternalError(format!("Failed to serialize request body: {}", e))
        })?;

        let response = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Content-Length", body_string.len().to_string())
            .body(body_string)
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;
        Self::handle_response(response)
    }

    pub fn close_session(&self) -> Result<(), GraphError> {
        let request_body = json!({
            "session": self.session_id,
            "op": "close",
            "processor": "session"
        });

        let body_string = serde_json::to_string(&request_body).map_err(|e| {
            GraphError::InternalError(format!("Failed to serialize request body: {}", e))
        })?;

        let response = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Content-Length", body_string.len().to_string())
            .body(body_string)
            .send()
            .map_err(|e| GraphError::ConnectionFailed(e.to_string()))?;
        Self::handle_response(response).map(|_| ())
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    fn handle_response(response: Response) -> Result<Value, GraphError> {
        let status = response.status();
        let status_code = status.as_u16();

        if status.is_success() {
            let response_body: Value = response.json().map_err(|e| {
                GraphError::InternalError(format!("Failed to parse response body: {}", e))
            })?;
            Ok(response_body)
        } else {
            let error_body: Value = response.json().map_err(|e| {
                GraphError::InternalError(format!("Failed to read error response: {}", e))
            })?;

            let error_msg = error_body
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");
            Err(GraphError::InvalidQuery(format!(
                "{}: {}",
                status_code, error_msg
            )))
        }
    }
}
