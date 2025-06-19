mod client;
mod connection;
mod conversions;
mod helpers;
mod query;
mod schema;
mod transaction;
mod traversal;

use client::Neo4jApi;
use golem_graph::durability::{DurableGraph, ExtendedGuest};
use golem_graph::golem::graph::{
    connection::ConnectionConfig, errors::GraphError, transactions::Guest as TransactionGuest,
};
use std::sync::Arc;

pub struct GraphNeo4jComponent;

pub struct Graph {
    api: Arc<Neo4jApi>,
}

pub struct Transaction {
    api: Arc<Neo4jApi>,
    transaction_url: String,
}

pub struct SchemaManager {
    graph: Arc<Graph>,
}

impl ExtendedGuest for GraphNeo4jComponent {
    type Graph = Graph;
    fn connect_internal(config: &ConnectionConfig) -> Result<Graph, GraphError> {
        let host = config
            .hosts
            .first()
            .ok_or_else(|| GraphError::ConnectionFailed("Missing host".to_string()))?;
        let port = config.port.unwrap_or(7687);
        let username = config
            .username
            .as_deref()
            .ok_or_else(|| GraphError::ConnectionFailed("Missing username".to_string()))?;
        let password = config
            .password
            .as_deref()
            .ok_or_else(|| GraphError::ConnectionFailed("Missing password".to_string()))?;

        let api = Neo4jApi::new(host, port, username, password);
        Ok(Graph::new(api))
    }
}

impl TransactionGuest for GraphNeo4jComponent {
    type Transaction = Transaction;
}

impl Graph {
    fn new(api: Neo4jApi) -> Self {
        Self { api: Arc::new(api) }
    }

    pub(crate) fn begin_transaction(&self) -> Result<Transaction, GraphError> {
        let tx_url = self.api.begin_transaction()?;
        Ok(Transaction::new(self.api.clone(), tx_url))
    }
}

impl Transaction {
    fn new(api: Arc<Neo4jApi>, transaction_url: String) -> Self {
        Self {
            api,
            transaction_url,
        }
    }
}

type DurableGraphNeo4jComponent = DurableGraph<GraphNeo4jComponent>;

golem_graph::export_graph!(DurableGraphNeo4jComponent with_types_in golem_graph);

#[cfg(test)]
mod tests {
    use super::*;
    use golem_graph::golem::graph::{
        connection::ConnectionConfig, transactions::GuestTransaction, types::PropertyValue,
    };
    use std::env;

    fn get_test_config() -> ConnectionConfig {
        let host = env::var("NEO4J_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port = env::var("NEO4J_PORT")
            .unwrap_or_else(|_| "7474".to_string())
            .parse()
            .unwrap();
        let user = env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string());
        let password = env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".to_string());
        let database = env::var("NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".to_string());

        ConnectionConfig {
            hosts: vec![host],
            port: Some(port),
            username: Some(user),
            password: Some(password),
            database_name: Some(database),
            timeout_seconds: None,
            max_connections: None,
            provider_config: vec![],
        }
    }

    #[test]
    fn test_successful_connection() {
        if env::var("NEO4J_HOST").is_err() {
            println!("Skipping test_successful_connection: NEO4J_HOST not set");
            return;
        }

        let config = get_test_config();
        let result = GraphNeo4jComponent::connect_internal(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_failed_connection_bad_credentials() {
        if env::var("NEO4J_HOST").is_err() {
            println!("Skipping test_failed_connection_bad_credentials: NEO4J_HOST not set");
            return;
        }

        let mut config = get_test_config();
        config.password = Some("wrong_password".to_string());

        let graph = GraphNeo4jComponent::connect_internal(&config).unwrap();
        let result = graph.begin_transaction();

        assert!(matches!(result, Err(GraphError::ConnectionFailed(_))));
    }

    #[test]
    fn test_durability_of_committed_data() {
        if env::var("NEO4J_HOST").is_err() {
            println!("Skipping test_durability_of_committed_data: NEO4J_HOST not set");
            return;
        }

        let config = get_test_config();
        let vertex_type = "DurabilityTestVertex".to_string();
        let unique_prop = (
            "test_id".to_string(),
            PropertyValue::StringValue("durable_test_1".to_string()),
        );

        let created_vertex_id = {
            let graph1 = GraphNeo4jComponent::connect_internal(&config).unwrap();
            let tx1 = graph1.begin_transaction().unwrap();
            let created_vertex = tx1
                .create_vertex(vertex_type.clone(), vec![unique_prop.clone()])
                .unwrap();
            tx1.commit().unwrap();
            created_vertex.id
        };

        let graph2 = GraphNeo4jComponent::connect_internal(&config).unwrap();
        let tx2 = graph2.begin_transaction().unwrap();

        let retrieved_vertex = tx2.get_vertex(created_vertex_id.clone()).unwrap();
        assert!(
            retrieved_vertex.is_some(),
            "Vertex should be durable and retrievable in a new session"
        );

        tx2.delete_vertex(created_vertex_id, true).unwrap();
        tx2.commit().unwrap();
    }
}
