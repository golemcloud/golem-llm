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

        let api = JanusGraphApi::new(host, port, username, password)?;
        api.execute("g.tx().open()", None)?;
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
