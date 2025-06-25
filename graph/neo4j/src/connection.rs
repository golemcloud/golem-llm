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

        // Query for node count
        let node_count_stmt = serde_json::json!({
            "statement": "MATCH (n) RETURN count(n) as nodeCount",
            "parameters": {}
        });
        let node_count_resp = self.api.execute_in_transaction(
            &transaction_url,
            serde_json::json!({ "statements": [node_count_stmt] }),
        )?;
        let node_count = node_count_resp["results"]
            .as_array()
            .and_then(|r| r.first())
            .and_then(|result| result["data"].as_array())
            .and_then(|d| d.first())
            .and_then(|data| data["row"].as_array())
            .and_then(|row| row.first())
            .and_then(|v| v.as_u64());

        // Query for relationship count
        let rel_count_stmt = serde_json::json!({
            "statement": "MATCH ()-[r]->() RETURN count(r) as relCount",
            "parameters": {}
        });
        let rel_count_resp = self.api.execute_in_transaction(
            &transaction_url,
            serde_json::json!({ "statements": [rel_count_stmt] }),
        )?;
        let rel_count = rel_count_resp["results"]
            .as_array()
            .and_then(|r| r.first())
            .and_then(|result| result["data"].as_array())
            .and_then(|d| d.first())
            .and_then(|data| data["row"].as_array())
            .and_then(|row| row.first())
            .and_then(|v| v.as_u64());

        self.api.rollback_transaction(&transaction_url)?;

        Ok(GraphStatistics {
            vertex_count: node_count,
            edge_count: rel_count,
            label_count: None,
            property_count: None,
        })
    }
}
