mod client;
mod connection;
mod conversions;
mod helpers;
mod query;
mod schema;
mod transaction;
mod traversal;

use client::ArangoDbApi;
use golem_graph::durability::{DurableGraph, ExtendedGuest};
use golem_graph::golem::graph::{
    connection::ConnectionConfig, errors::GraphError, transactions::Guest as TransactionGuest,
};
use std::sync::Arc;

pub struct GraphArangoDbComponent;

pub struct Graph {
    api: Arc<ArangoDbApi>,
}

pub struct Transaction {
    api: Arc<ArangoDbApi>,
    transaction_id: String,
}

pub struct SchemaManager {
    graph: Arc<Graph>,
}

impl ExtendedGuest for GraphArangoDbComponent {
    type Graph = Graph;
    fn connect_internal(config: &ConnectionConfig) -> Result<Graph, GraphError> {
        let host: &String = config
            .hosts
            .first()
            .ok_or_else(|| GraphError::ConnectionFailed("Missing host".to_string()))?;

        let port = config.port.unwrap_or(8529);

        let username = config
            .username
            .as_deref()
            .ok_or_else(|| GraphError::ConnectionFailed("Missing username".to_string()))?;
        let password = config
            .password
            .as_deref()
            .ok_or_else(|| GraphError::ConnectionFailed("Missing password".to_string()))?;

        let database_name = config.database_name.as_deref().unwrap_or("_system");

        let api = ArangoDbApi::new(host, port, username, password, database_name);
        Ok(Graph::new(api))
    }
}

impl TransactionGuest for GraphArangoDbComponent {
    type Transaction = Transaction;
}

impl Graph {
    fn new(api: ArangoDbApi) -> Self {
        Self { api: Arc::new(api) }
    }
}

impl Transaction {
    fn new(api: Arc<ArangoDbApi>, transaction_id: String) -> Self {
        Self {
            api,
            transaction_id,
        }
    }
}

type DurableGraphArangoDbComponent = DurableGraph<GraphArangoDbComponent>;

golem_graph::export_graph!(DurableGraphArangoDbComponent with_types_in golem_graph);

#[cfg(test)]
mod lib_tests {
    use super::*;
    use golem_graph::durability::ExtendedGuest;
    use golem_graph::golem::graph::{
        connection::ConnectionConfig, transactions::GuestTransaction, types::PropertyValue,
    };

    use std::env;

    fn get_test_config() -> ConnectionConfig {
        let host = env::var("ARANGODB_HOST").unwrap_or_else(|_| "localhost".into());
        let port = env::var("ARANGODB_PORT")
            .unwrap_or_else(|_| "8529".into())
            .parse()
            .expect("Invalid ARANGODB_PORT");
        let username = env::var("ARANGODB_USER").unwrap_or_else(|_| "root".into());
        let password = env::var("ARANGODB_PASS").unwrap_or_else(|_| "".into());
        let database_name = env::var("ARANGODB_DB").unwrap_or_else(|_| "_system".into());

        ConnectionConfig {
            hosts: vec![host],
            port: Some(port),
            username: Some(username),
            password: Some(password),
            database_name: Some(database_name),
            timeout_seconds: None,
            max_connections: None,
            provider_config: vec![],
        }
    }
    fn create_test_transaction() -> crate::Transaction {
        let host = env::var("ARANGODB_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port: u16 = env::var("ARANGODB_PORT")
            .unwrap_or_else(|_| "8529".to_string())
            .parse()
            .expect("Invalid ARANGODB_PORT");
        let user = env::var("ARANGODB_USER").unwrap_or_else(|_| "root".to_string());
        let pass = env::var("ARANGODB_PASS").unwrap_or_else(|_| "".to_string());
        let db = env::var("ARANGODB_DB").unwrap_or_else(|_| "_system".to_string());

        let api = ArangoDbApi::new(&host, port, &user, &pass, &db);
        let tx_id = api.begin_transaction(false).unwrap();
        crate::Transaction::new(std::sync::Arc::new(api), tx_id)
    }

    #[test]
    fn test_successful_connection() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_successful_connection: ARANGODB_HOST not set");
            return;
        }
        let cfg = get_test_config();
        let graph = GraphArangoDbComponent::connect_internal(&cfg);
        assert!(graph.is_ok(), "connect_internal should succeed");
    }

    #[test]
    fn test_failed_connection_bad_credentials() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_successful_connection: ARANGODB_HOST not set");
            return;
        }
        let mut cfg = get_test_config();
        cfg.username = Some("bad_user".into());
        cfg.password = Some("bad_pass".into());

        let api = ArangoDbApi::new(
            &cfg.hosts[0],
            cfg.port.unwrap(),
            cfg.username.as_deref().unwrap(),
            cfg.password.as_deref().unwrap(),
            cfg.database_name.as_deref().unwrap(),
        );
        assert!(api.begin_transaction(false).is_err());
    }

    #[test]
    fn test_durability_of_committed_data() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_successful_connection: ARANGODB_HOST not set");
            return;
        }

        let tx1 = create_test_transaction();
        let unique_id = "dur_test_123".to_string();
        let created = tx1
            .create_vertex(
                "DurTest".into(),
                vec![(
                    "test_id".into(),
                    PropertyValue::StringValue(unique_id.clone()),
                )],
            )
            .unwrap();
        tx1.commit().unwrap();

        let tx2 = create_test_transaction();
        let fetched = tx2.get_vertex(created.id.clone()).unwrap();
        assert!(fetched.is_some());

        tx2.delete_vertex(created.id, true).unwrap();
        tx2.commit().unwrap();
    }
}
