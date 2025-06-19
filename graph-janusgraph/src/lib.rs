mod client;
mod connection;
mod conversions;
mod helpers;
mod query;
mod query_utils;
mod schema;
mod transaction;
mod traversal;

use client::JanusGraphApi;
use golem_graph::durability::{DurableGraph, ExtendedGuest};
use golem_graph::golem::graph::{
    connection::ConnectionConfig, errors::GraphError, transactions::Guest as TransactionGuest,
};
use std::sync::Arc;

pub struct GraphJanusGraphComponent;

pub struct Graph {
    pub api: Arc<JanusGraphApi>,
}

pub struct Transaction {
    api: Arc<JanusGraphApi>,
}

pub struct SchemaManager {
    pub graph: Arc<Graph>,
}

impl ExtendedGuest for GraphJanusGraphComponent {
    type Graph = Graph;
    fn connect_internal(config: &ConnectionConfig) -> Result<Graph, GraphError> {
        let host = config
            .hosts
            .first()
            .ok_or_else(|| GraphError::ConnectionFailed("Missing host".to_string()))?;
        let port = config.port.unwrap_or(8182); // Default Gremlin Server port
        let username = config.username.as_deref();
        let password = config.password.as_deref();

        // Create a new JanusGraphApi instance, propagating any errors.
        let api = JanusGraphApi::new(host, port, username, password)?;
        Ok(Graph::new(api))
    }
}

impl TransactionGuest for GraphJanusGraphComponent {
    type Transaction = Transaction;
}

impl Graph {
    fn new(api: JanusGraphApi) -> Self {
        Self { api: Arc::new(api) }
    }
}

impl Transaction {
    fn new(api: Arc<JanusGraphApi>) -> Self {
        Self { api }
    }
}

type DurableGraphJanusGraphComponent = DurableGraph<GraphJanusGraphComponent>;

golem_graph::export_graph!(DurableGraphJanusGraphComponent with_types_in golem_graph);

#[cfg(test)]
mod tests {
    use super::*;
    use golem_graph::golem::graph::connection::GuestGraph;
    use golem_graph::golem::graph::transactions::GuestTransaction;

    use golem_graph::golem::graph::{connection::ConnectionConfig, types::PropertyValue};
    use std::env;
    use uuid::Uuid;

    fn get_test_config() -> ConnectionConfig {
        let host = env::var("JANUSGRAPH_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port = env::var("JANUSGRAPH_PORT")
            .unwrap_or_else(|_| "8182".to_string())
            .parse()
            .unwrap();
        let username = env::var("JANUSGRAPH_USER").ok();
        let password = env::var("JANUSGRAPH_PASSWORD").ok();

        ConnectionConfig {
            hosts: vec![host],
            port: Some(port),
            username,
            password,
            database_name: None,
            timeout_seconds: None,
            max_connections: None,
            provider_config: vec![],
        }
    }

    fn create_test_transaction(cfg: &ConnectionConfig) -> Transaction {
        let host = &cfg.hosts[0];
        let port = cfg.port.unwrap();
        let api = JanusGraphApi::new(host, port, cfg.username.as_deref(), cfg.password.as_deref())
            .unwrap();
        Transaction::new(Arc::new(api))
    }

    #[test]
    fn test_successful_connection() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_successful_connection: JANUSGRAPH_HOST not set");
            return;
        }
        let cfg = get_test_config();
        let graph = GraphJanusGraphComponent::connect_internal(&cfg);
        assert!(graph.is_ok(), "connect_internal should succeed");
    }

    #[test]
    fn test_failed_connection_bad_credentials() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_failed_connection_bad_credentials: JANUSGRAPH_HOST not set");
            return;
        }
        let mut cfg = get_test_config();
        cfg.username = Some("bad_user".to_string());
        cfg.password = Some("bad_pass".to_string());

        let graph = GraphJanusGraphComponent::connect_internal(&cfg).unwrap();
        let res = graph.begin_transaction();
        assert!(
            matches!(
                res,
                Err(GraphError::ConnectionFailed(_))
                    | Err(GraphError::InternalError(_))
                    | Err(GraphError::InvalidQuery(_))
            ),
            "Bad creds should error"
        );
    }

    #[test]
    fn test_durability_of_committed_data() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_durability_of_committed_data");
            return;
        }
        let cfg = get_test_config();

        let tx1 = create_test_transaction(&cfg);
        let unique_id = Uuid::new_v4().to_string();
        let created = tx1
            .create_vertex(
                "DurTest".to_string(),
                vec![(
                    "test_id".to_string(),
                    PropertyValue::StringValue(unique_id.clone()),
                )],
            )
            .unwrap();
        tx1.commit().unwrap();

        let tx2 = create_test_transaction(&cfg);
        let fetched = tx2.get_vertex(created.id.clone()).unwrap();
        assert!(fetched.is_some(), "Vertex persisted across sessions");

        tx2.delete_vertex(created.id, true).unwrap();
        tx2.commit().unwrap();
    }
}
