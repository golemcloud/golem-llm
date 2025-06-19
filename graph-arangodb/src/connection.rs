use crate::{Graph, Transaction};
use golem_graph::{
    durability::ProviderGraph,
    golem::graph::{
        connection::{GraphStatistics, GuestGraph},
        errors::GraphError,
        transactions::Transaction as TransactionResource,
    },
};

impl ProviderGraph for Graph {
    type Transaction = Transaction;
}

impl GuestGraph for Graph {
    fn begin_transaction(&self) -> Result<TransactionResource, GraphError> {
        let transaction_id = self.api.begin_transaction(false)?;
        let transaction = Transaction::new(self.api.clone(), transaction_id);
        Ok(TransactionResource::new(transaction))
    }

    fn begin_read_transaction(&self) -> Result<TransactionResource, GraphError> {
        let transaction_id = self.api.begin_transaction(true)?;
        let transaction = Transaction::new(self.api.clone(), transaction_id);
        Ok(TransactionResource::new(transaction))
    }

    fn ping(&self) -> Result<(), GraphError> {
        self.api.ping()
    }

    fn close(&self) -> Result<(), GraphError> {
        // The ArangoDB client uses a connection pool, so a specific close is not needed.
        Ok(())
    }

    fn get_statistics(&self) -> Result<GraphStatistics, GraphError> {
        let stats = self.api.get_database_statistics()?;

        Ok(GraphStatistics {
            vertex_count: Some(stats.vertex_count),
            edge_count: Some(stats.edge_count),
            label_count: None, // ArangoDB doesn't have a direct concept of "labels" count
            property_count: None, // Too expensive to calculate across the whole DB
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ArangoDbApi;
    use golem_graph::golem::graph::transactions::GuestTransaction;
    use std::env;
    use std::sync::Arc;

    fn get_test_graph() -> Graph {
        let host = env::var("ARANGODB_HOST").unwrap_or_else(|_| "localhost".into());
        let port: u16 = env::var("ARANGODB_PORT")
            .unwrap_or_else(|_| "8529".into())
            .parse()
            .expect("Invalid ARANGODB_PORT");

        let user = env::var("ARANGODB_USER").unwrap_or_default();
        let pass = env::var("ARANGODB_PASS").unwrap_or_default();
        let database = env::var("ARANGODB_DB").unwrap_or_else(|_| "_system".into());

        let api = ArangoDbApi::new(&host, port, &user, &pass, &database);
        Graph { api: Arc::new(api) }
    }

    fn create_test_transaction() -> Transaction {
        let graph = get_test_graph();
        let tx_id = graph
            .api
            .begin_transaction(false)
            .expect("Failed to begin transaction");
        Transaction::new(graph.api.clone(), tx_id)
    }

    #[test]
    fn test_ping() {
        if env::var("ARANGODB_HOST").is_err() {
            eprintln!("Skipping test_ping: ARANGODB_HOST not set");
            return;
        }
        let graph = get_test_graph();
        assert!(graph.ping().is_ok(), "Ping should succeed");
    }

    #[test]
    fn test_get_statistics() {
        if env::var("ARANGODB_HOST").is_err() {
            eprintln!("Skipping test_get_statistics: ARANGODB_HOST not set");
            return;
        }

        let graph = get_test_graph();
        let tx = create_test_transaction();

        // initial stats
        let initial = graph.get_statistics().unwrap_or(GraphStatistics {
            vertex_count: Some(0),
            edge_count: Some(0),
            label_count: None,
            property_count: None,
        });

        let v1 = tx.create_vertex("StatNode".into(), vec![]).expect("v1");
        let v2 = tx.create_vertex("StatNode".into(), vec![]).expect("v2");

        tx.create_edge("STAT_EDGE".into(), v1.id.clone(), v2.id.clone(), vec![])
            .expect("edge");
        tx.commit().expect("commit");

        let updated = graph.get_statistics().expect("get_statistics failed");
        assert_eq!(
            updated.vertex_count,
            initial.vertex_count.map(|c| c + 2).or(Some(2)),
            "Vertex count should increase by 2"
        );
        assert_eq!(
            updated.edge_count,
            initial.edge_count.map(|c| c + 1).or(Some(1)),
            "Edge count should increase by 1"
        );

        let tx2 = create_test_transaction();
        let cleanup_aql = r#"
            FOR doc IN StatNode
              REMOVE doc IN StatNode
        "#;
        tx2.execute_query(cleanup_aql.to_string(), None, None)
            .expect("cleanup");
        tx2.commit().expect("cleanup commit");
    }
}
