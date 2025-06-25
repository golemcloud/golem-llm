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
        // Ensure common collections exist before starting transaction
        // This is act as just helper for testing purposes, in production we would not need this
        // let common_collections = vec![
        //     (
        //         "Person",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "TempUser",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "Company",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "Employee",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "Node",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "Product",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "User",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "KNOWS",
        //         golem_graph::golem::graph::schema::ContainerType::EdgeContainer,
        //     ),
        //     (
        //         "WORKS_FOR",
        //         golem_graph::golem::graph::schema::ContainerType::EdgeContainer,
        //     ),
        //     (
        //         "CONNECTS",
        //         golem_graph::golem::graph::schema::ContainerType::EdgeContainer,
        //     ),
        //     (
        //         "FOLLOWS",
        //         golem_graph::golem::graph::schema::ContainerType::EdgeContainer,
        //     ),
        // ];

        // // Create collections if they don't exist
        // for (name, container_type) in common_collections {
        //     let _ = self.api.ensure_collection_exists(name, container_type);
        // }

        let transaction_id = self.api.begin_dynamic_transaction(false)?;
        let transaction = Transaction::new(self.api.clone(), transaction_id);
        Ok(TransactionResource::new(transaction))
    }

    fn begin_read_transaction(&self) -> Result<TransactionResource, GraphError> {
        // Ensure common collections exist before starting transaction
        // This is act as just helper for testing purposes, in production we would not need this
        // let common_collections = vec![
        //     (
        //         "Person",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "TempUser",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "Company",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "Employee",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "Node",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "Product",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "User",
        //         golem_graph::golem::graph::schema::ContainerType::VertexContainer,
        //     ),
        //     (
        //         "KNOWS",
        //         golem_graph::golem::graph::schema::ContainerType::EdgeContainer,
        //     ),
        //     (
        //         "WORKS_FOR",
        //         golem_graph::golem::graph::schema::ContainerType::EdgeContainer,
        //     ),
        //     (
        //         "CONNECTS",
        //         golem_graph::golem::graph::schema::ContainerType::EdgeContainer,
        //     ),
        //     (
        //         "FOLLOWS",
        //         golem_graph::golem::graph::schema::ContainerType::EdgeContainer,
        //     ),
        // ];

        // // Create collections if they don't exist
        // for (name, container_type) in common_collections {
        //     let _ = self.api.ensure_collection_exists(name, container_type);
        // }

        let transaction_id = self.api.begin_dynamic_transaction(true)?;
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
            property_count: None,
        })
    }
}
