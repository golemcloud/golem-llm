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
        let transaction = Transaction::new(self.api.clone());
        Ok(TransactionResource::new(transaction))
    }

    fn begin_read_transaction(&self) -> Result<TransactionResource, GraphError> {
        self.begin_transaction()
    }

    fn ping(&self) -> Result<(), GraphError> {
        self.api.execute("1+1", None)?;
        Ok(())
    }

    fn close(&self) -> Result<(), GraphError> {
        // The underlying HTTP client doesn't need explicit closing for this implementation.
        Ok(())
    }

    fn get_statistics(&self) -> Result<GraphStatistics, GraphError> {
        let vertex_count_res = self.api.execute("g.V().count()", None)?;
        let edge_count_res = self.api.execute("g.E().count()", None)?;

        let vertex_count = vertex_count_res
            .get("result")
            .and_then(|r| r.get("data"))
            .and_then(|d| d.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_u64());

        let edge_count = edge_count_res
            .get("result")
            .and_then(|r| r.get("data"))
            .and_then(|d| d.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_u64());

        Ok(GraphStatistics {
            vertex_count,
            edge_count,
            label_count: None, // JanusGraph requires a more complex query for this
            property_count: None, // JanusGraph requires a more complex query for this
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::JanusGraphApi;
    use golem_graph::golem::graph::transactions::GuestTransaction;
    use std::{env, sync::Arc};

    fn get_test_graph() -> Graph {
        let host = env::var("JANUSGRAPH_HOST").unwrap_or_else(|_| "localhost".into());
        let port: u16 = env::var("JANUSGRAPH_PORT")
            .unwrap_or_else(|_| "8182".into())
            .parse()
            .unwrap();
        let api = JanusGraphApi::new(&host, port, None, None).unwrap();
        Graph { api: Arc::new(api) }
    }

    fn create_test_transaction() -> Transaction {
        let host = env::var("JANUSGRAPH_HOST").unwrap_or_else(|_| "localhost".into());
        let port: u16 = env::var("JANUSGRAPH_PORT")
            .unwrap_or_else(|_| "8182".into())
            .parse()
            .unwrap();
        let api = JanusGraphApi::new(&host, port, None, None).unwrap();
        // this returns your crate::Transaction
        Transaction { api: Arc::new(api) }
    }

    #[test]
    fn test_ping() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_ping: JANUSGRAPH_HOST not set");
            return;
        }
        let graph = get_test_graph();
        assert!(graph.ping().is_ok());
    }

    #[test]
    fn test_get_statistics() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_get_statistics: JANUSGRAPH_HOST not set");
            return;
        }

        let graph = get_test_graph();
        let tx = create_test_transaction();

        let initial = graph.get_statistics().unwrap_or(GraphStatistics {
            vertex_count: Some(0),
            edge_count: Some(0),
            label_count: None,
            property_count: None,
        });

        let v1 = tx.create_vertex("StatNode".to_string(), vec![]).unwrap();
        let v2 = tx.create_vertex("StatNode".to_string(), vec![]).unwrap();
        tx.create_edge(
            "STAT_EDGE".to_string(),
            v1.id.clone(),
            v2.id.clone(),
            vec![],
        )
        .unwrap();
        tx.commit().unwrap();

        let updated = graph.get_statistics().unwrap();
        assert_eq!(
            updated.vertex_count,
            initial.vertex_count.map(|c| c + 2).or(Some(2))
        );
        assert_eq!(
            updated.edge_count,
            initial.edge_count.map(|c| c + 1).or(Some(1))
        );

        let tx2 = create_test_transaction();
        tx2.execute_query("g.V().hasLabel('StatNode').drop()".to_string(), None, None)
            .unwrap();
        tx2.commit().unwrap();
    }
}
