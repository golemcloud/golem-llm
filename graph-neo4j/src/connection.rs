use crate::{Graph, Transaction};
use golem_graph::{
    durability::ProviderGraph,
    golem::graph::{
        connection::{GraphStatistics, GuestGraph},
        errors::GraphError,
        transactions::Transaction as TransactionResource,
    },
};
use serde_json::json;

impl ProviderGraph for Graph {
    type Transaction = Transaction;
}

impl GuestGraph for Graph {
    fn begin_transaction(&self) -> Result<TransactionResource, GraphError> {
        let transaction_url = self.api.begin_transaction()?;
        let transaction = Transaction::new(self.api.clone(), transaction_url);
        Ok(TransactionResource::new(transaction))
    }

    fn begin_read_transaction(&self) -> Result<TransactionResource, GraphError> {
        let transaction_url = self.api.begin_transaction()?;
        let transaction = Transaction::new(self.api.clone(), transaction_url);
        Ok(TransactionResource::new(transaction))
    }

    fn ping(&self) -> Result<(), GraphError> {
        let transaction_url = self.api.begin_transaction()?;
        self.api.rollback_transaction(&transaction_url)
    }

    fn close(&self) -> Result<(), GraphError> {
        Ok(())
    }

    fn get_statistics(&self) -> Result<GraphStatistics, GraphError> {
        let transaction_url = self.api.begin_transaction()?;

        let statement = json!({
            "statement": "CALL db.stats.retrieve('GRAPH_COUNTS') YIELD nodeCount, relCount RETURN nodeCount, relCount",
            "parameters": {}
        });
        let statements = json!({ "statements": [statement] });

        let response_result = self
            .api
            .execute_in_transaction(&transaction_url, statements);
        let rollback_result = self.api.rollback_transaction(&transaction_url);

        let response = response_result?;

        let result = response["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| {
                GraphError::InternalError(
                    "Invalid response structure from Neo4j for get_statistics".to_string(),
                )
            })?;

        if let Some(errors) = result["errors"].as_array() {
            if !errors.is_empty() {
                return Err(GraphError::InvalidQuery(errors[0].to_string()));
            }
        }

        let data = result["data"]
            .as_array()
            .and_then(|d| d.first())
            .ok_or_else(|| {
                GraphError::InternalError("Missing data in get_statistics response".to_string())
            })?;

        let row = data["row"].as_array().ok_or_else(|| {
            GraphError::InternalError("Missing row data for get_statistics".to_string())
        })?;

        if row.len() < 2 {
            return Err(GraphError::InternalError(
                "Invalid row data for get_statistics, expected at least 2 columns".to_string(),
            ));
        }

        let vertex_count = row[0].as_u64();
        let edge_count = row[1].as_u64();

        rollback_result?;

        Ok(GraphStatistics {
            vertex_count,
            edge_count,
            label_count: None,
            property_count: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphNeo4jComponent;
    use golem_graph::durability::ExtendedGuest;
    use golem_graph::golem::graph::{connection::ConnectionConfig, transactions::GuestTransaction};
    use std::env;

    fn get_test_graph() -> Graph {
        let host = env::var("NEO4J_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port = env::var("NEO4J_PORT")
            .unwrap_or_else(|_| "7474".to_string())
            .parse()
            .unwrap();
        let user = env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string());
        let password = env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".to_string());
        let database = env::var("NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".to_string());

        let config = ConnectionConfig {
            hosts: vec![host],
            port: Some(port),
            username: Some(user),
            password: Some(password),
            database_name: Some(database),
            timeout_seconds: None,
            max_connections: None,
            provider_config: vec![],
        };

        GraphNeo4jComponent::connect_internal(&config).unwrap()
    }

    #[test]
    fn test_ping() {
        if env::var("NEO4J_HOST").is_err() {
            println!("Skipping test_ping: NEO4J_HOST not set");
            return;
        }
        let graph = get_test_graph();
        let result = graph.ping();
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_statistics() {
        if env::var("NEO4J_HOST").is_err() {
            println!("Skipping test_get_statistics: NEO4J_HOST not set");
            return;
        }

        let graph = get_test_graph();
        let tx = Graph::begin_transaction(&graph).unwrap();

        let initial_stats = graph.get_statistics().unwrap();

        let v1 = tx.create_vertex("StatNode".to_string(), vec![]).unwrap();
        let v2 = tx.create_vertex("StatNode".to_string(), vec![]).unwrap();
        tx.create_edge("STAT_EDGE".to_string(), v1.id, v2.id, vec![])
            .unwrap();
        tx.commit().unwrap();

        let new_stats = graph.get_statistics().unwrap();
        assert_eq!(
            new_stats.vertex_count,
            Some(initial_stats.vertex_count.unwrap_or(0) + 2)
        );
        assert_eq!(
            new_stats.edge_count,
            Some(initial_stats.edge_count.unwrap_or(0) + 1)
        );

        let cleanup_tx = Graph::begin_transaction(&graph).unwrap();
        cleanup_tx
            .execute_query("MATCH (n:StatNode) DETACH DELETE n".to_string(), None, None)
            .unwrap();
        cleanup_tx.commit().unwrap();
    }
}
