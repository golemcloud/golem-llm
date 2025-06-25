use crate::golem::graph::{
    connection::{self, ConnectionConfig, GuestGraph},
    errors::GraphError,
    query::{Guest as QueryGuest, QueryExecutionResult, QueryOptions},
    schema::{Guest as SchemaGuest, SchemaManager},
    transactions::{self, Guest as TransactionGuest, GuestTransaction},
    traversal::{Guest as TraversalGuest, Path, PathOptions, Subgraph},
};
use std::marker::PhantomData;

pub trait TransactionBorrowExt<'a, T> {
    fn get(&self) -> &'a T;
}

pub struct DurableGraph<Impl> {
    _phantom: PhantomData<Impl>,
}

pub trait ExtendedGuest: 'static
where
    Self::Graph: ProviderGraph + 'static,
{
    type Graph: connection::GuestGraph;
    fn connect_internal(config: &ConnectionConfig) -> Result<Self::Graph, GraphError>;
}

pub trait ProviderGraph: connection::GuestGraph {
    type Transaction: transactions::GuestTransaction;
}

/// When the durability feature flag is off, wrapping with `DurableGraph` is just a passthrough
#[cfg(not(feature = "durability"))]
mod passthrough_impl {
    use super::*;

    impl<Impl: ExtendedGuest> connection::Guest for DurableGraph<Impl>
    where
        Impl::Graph: ProviderGraph + 'static,
    {
        type Graph = Impl::Graph;

        fn connect(config: ConnectionConfig) -> Result<connection::Graph, GraphError> {
            let graph = Impl::connect_internal(&config)?;
            Ok(connection::Graph::new(graph))
        }
    }

    impl<Impl: ExtendedGuest + TransactionGuest> TransactionGuest for DurableGraph<Impl>
    where
        Impl::Graph: ProviderGraph + 'static,
    {
        type Transaction = Impl::Transaction;
    }

    impl<Impl: ExtendedGuest + SchemaGuest> SchemaGuest for DurableGraph<Impl>
    where
        Impl::Graph: ProviderGraph + 'static,
    {
        type SchemaManager = Impl::SchemaManager;

        fn get_schema_manager() -> Result<SchemaManager, GraphError> {
            Impl::get_schema_manager()
        }
    }

    impl<Impl: ExtendedGuest + TraversalGuest> TraversalGuest for DurableGraph<Impl>
    where
        Impl::Graph: ProviderGraph + 'static,
    {
        fn find_shortest_path(
            transaction: transactions::TransactionBorrow<'_>,
            from_vertex: crate::golem::graph::types::ElementId,
            to_vertex: crate::golem::graph::types::ElementId,
            options: Option<PathOptions>,
        ) -> Result<Option<Path>, GraphError> {
            Impl::find_shortest_path(transaction, from_vertex, to_vertex, options)
        }

        fn find_all_paths(
            transaction: transactions::TransactionBorrow<'_>,
            from_vertex: crate::golem::graph::types::ElementId,
            to_vertex: crate::golem::graph::types::ElementId,
            options: Option<PathOptions>,
            limit: Option<u32>,
        ) -> Result<Vec<Path>, GraphError> {
            Impl::find_all_paths(transaction, from_vertex, to_vertex, options, limit)
        }

        fn get_neighborhood(
            transaction: transactions::TransactionBorrow<'_>,
            center: crate::golem::graph::types::ElementId,
            options: crate::golem::graph::traversal::NeighborhoodOptions,
        ) -> Result<Subgraph, GraphError> {
            Impl::get_neighborhood(transaction, center, options)
        }

        fn path_exists(
            transaction: transactions::TransactionBorrow<'_>,
            from_vertex: crate::golem::graph::types::ElementId,
            to_vertex: crate::golem::graph::types::ElementId,
            options: Option<PathOptions>,
        ) -> Result<bool, GraphError> {
            Impl::path_exists(transaction, from_vertex, to_vertex, options)
        }

        fn get_vertices_at_distance(
            transaction: transactions::TransactionBorrow<'_>,
            source: crate::golem::graph::types::ElementId,
            distance: u32,
            direction: crate::golem::graph::types::Direction,
            edge_types: Option<Vec<String>>,
        ) -> Result<Vec<crate::golem::graph::types::Vertex>, GraphError> {
            Impl::get_vertices_at_distance(transaction, source, distance, direction, edge_types)
        }
    }

    impl<Impl: ExtendedGuest + QueryGuest> QueryGuest for DurableGraph<Impl>
    where
        Impl::Graph: ProviderGraph + 'static,
    {
        fn execute_query(
            transaction: transactions::TransactionBorrow<'_>,
            query: String,
            parameters: Option<Vec<(String, crate::golem::graph::types::PropertyValue)>>,
            options: Option<QueryOptions>,
        ) -> Result<QueryExecutionResult, GraphError> {
            Impl::execute_query(transaction, query, parameters, options)
        }
    }
}

#[cfg(feature = "durability")]
mod durable_impl {
    use super::*;
    use golem_rust::bindings::golem::durability::durability::WrappedFunctionType;
    use golem_rust::durability::Durability;
    use golem_rust::{with_persistence_level, FromValueAndType, IntoValue, PersistenceLevel};

    #[derive(Debug, Clone, FromValueAndType, IntoValue)]
    pub(super) struct Unit;

    #[derive(Debug)]
    pub struct DurableGraphResource<G> {
        graph: G,
    }

    #[derive(Debug)]
    pub struct DurableTransaction<T: GuestTransaction> {
        pub inner: T,
    }

    impl<T: GuestTransaction> DurableTransaction<T> {
        pub fn new(inner: T) -> Self {
            Self { inner }
        }
    }

    impl<Impl: ExtendedGuest> connection::Guest for DurableGraph<Impl>
    where
        Impl::Graph: ProviderGraph + 'static,
    {
        type Graph = DurableGraphResource<Impl::Graph>;
        fn connect(config: ConnectionConfig) -> Result<connection::Graph, GraphError> {
            let durability = Durability::<Unit, GraphError>::new(
                "golem_graph",
                "connect",
                WrappedFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = Impl::connect_internal(&config);
                let persist_result = result.as_ref().map(|_| Unit).map_err(|e| e.clone());
                durability.persist(config.clone(), persist_result)?;
                result.map(|g| connection::Graph::new(DurableGraphResource::new(g)))
            } else {
                let _unit: Unit = durability.replay::<Unit, GraphError>()?;
                let graph = Impl::connect_internal(&config)?;
                Ok(connection::Graph::new(DurableGraphResource::new(graph)))
            }
        }
    }

    impl<Impl: ExtendedGuest + TransactionGuest> TransactionGuest for DurableGraph<Impl>
    where
        Impl::Graph: ProviderGraph + 'static,
    {
        type Transaction = DurableTransaction<Impl::Transaction>;
    }

    impl<Impl: ExtendedGuest + SchemaGuest> SchemaGuest for DurableGraph<Impl>
    where
        Impl::Graph: ProviderGraph + 'static,
    {
        type SchemaManager = Impl::SchemaManager;

        fn get_schema_manager() -> Result<SchemaManager, GraphError> {
            Impl::get_schema_manager()
        }
    }

    impl<Impl: ExtendedGuest + TraversalGuest> TraversalGuest for DurableGraph<Impl>
    where
        Impl::Graph: ProviderGraph + 'static,
    {
        fn find_shortest_path(
            transaction: transactions::TransactionBorrow<'_>,
            from_vertex: crate::golem::graph::types::ElementId,
            to_vertex: crate::golem::graph::types::ElementId,
            options: Option<PathOptions>,
        ) -> Result<Option<Path>, GraphError> {
            Impl::find_shortest_path(transaction, from_vertex, to_vertex, options)
        }

        fn find_all_paths(
            transaction: transactions::TransactionBorrow<'_>,
            from_vertex: crate::golem::graph::types::ElementId,
            to_vertex: crate::golem::graph::types::ElementId,
            options: Option<PathOptions>,
            limit: Option<u32>,
        ) -> Result<Vec<Path>, GraphError> {
            Impl::find_all_paths(transaction, from_vertex, to_vertex, options, limit)
        }

        fn get_neighborhood(
            transaction: transactions::TransactionBorrow<'_>,
            center: crate::golem::graph::types::ElementId,
            options: crate::golem::graph::traversal::NeighborhoodOptions,
        ) -> Result<Subgraph, GraphError> {
            Impl::get_neighborhood(transaction, center, options)
        }

        fn path_exists(
            transaction: transactions::TransactionBorrow<'_>,
            from_vertex: crate::golem::graph::types::ElementId,
            to_vertex: crate::golem::graph::types::ElementId,
            options: Option<PathOptions>,
        ) -> Result<bool, GraphError> {
            Impl::path_exists(transaction, from_vertex, to_vertex, options)
        }

        fn get_vertices_at_distance(
            transaction: transactions::TransactionBorrow<'_>,
            source: crate::golem::graph::types::ElementId,
            distance: u32,
            direction: crate::golem::graph::types::Direction,
            edge_types: Option<Vec<String>>,
        ) -> Result<Vec<crate::golem::graph::types::Vertex>, GraphError> {
            Impl::get_vertices_at_distance(transaction, source, distance, direction, edge_types)
        }
    }

    impl<Impl: ExtendedGuest + QueryGuest> QueryGuest for DurableGraph<Impl>
    where
        Impl::Graph: ProviderGraph + 'static,
    {
        fn execute_query(
            transaction: transactions::TransactionBorrow<'_>,
            query: String,
            parameters: Option<Vec<(String, crate::golem::graph::types::PropertyValue)>>,
            options: Option<QueryOptions>,
        ) -> Result<QueryExecutionResult, GraphError> {
            let durability: Durability<QueryExecutionResult, GraphError> = Durability::new(
                "golem_graph_query",
                "execute_query",
                WrappedFunctionType::WriteRemote,
            );

            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    Impl::execute_query(transaction, query.clone(), parameters.clone(), options)
                });
                durability.persist(
                    ExecuteQueryParams {
                        query,
                        parameters,
                        options,
                    },
                    result,
                )
            } else {
                durability.replay()
            }
        }
    }

    impl<G: ProviderGraph + 'static> connection::GuestGraph for DurableGraphResource<G> {
        fn begin_transaction(&self) -> Result<transactions::Transaction, GraphError> {
            self.graph.begin_transaction().map(|tx_wrapper| {
                let provider_transaction = tx_wrapper.into_inner::<G::Transaction>();
                transactions::Transaction::new(DurableTransaction::new(provider_transaction))
            })
        }

        fn begin_read_transaction(&self) -> Result<transactions::Transaction, GraphError> {
            self.graph.begin_read_transaction().map(|tx_wrapper| {
                let provider_transaction = tx_wrapper.into_inner::<G::Transaction>();
                transactions::Transaction::new(DurableTransaction::new(provider_transaction))
            })
        }

        fn ping(&self) -> Result<(), GraphError> {
            self.graph.ping()
        }

        fn get_statistics(
            &self,
        ) -> Result<crate::golem::graph::connection::GraphStatistics, GraphError> {
            self.graph.get_statistics()
        }

        fn close(&self) -> Result<(), GraphError> {
            self.graph.close()
        }
    }

    impl<G: GuestGraph> DurableGraphResource<G> {
        pub fn new(graph: G) -> Self {
            Self { graph }
        }
    }

    impl<T: GuestTransaction> GuestTransaction for DurableTransaction<T> {
        fn commit(&self) -> Result<(), GraphError> {
            let durability = Durability::<Unit, GraphError>::new(
                "golem_graph_transaction",
                "commit",
                WrappedFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.commit()
                });
                durability.persist(Unit, result.map(|_| Unit))?;
                Ok(())
            } else {
                durability.replay::<Unit, GraphError>()?;
                Ok(())
            }
        }

        fn rollback(&self) -> Result<(), GraphError> {
            let durability = Durability::<Unit, GraphError>::new(
                "golem_graph_transaction",
                "rollback",
                WrappedFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.rollback()
                });
                durability.persist(Unit, result.map(|_| Unit))?;
                Ok(())
            } else {
                durability.replay::<Unit, GraphError>()?;
                Ok(())
            }
        }

        fn create_vertex(
            &self,
            vertex_type: String,
            properties: crate::golem::graph::types::PropertyMap,
        ) -> Result<crate::golem::graph::types::Vertex, GraphError> {
            let durability: Durability<crate::golem::graph::types::Vertex, GraphError> =
                Durability::new(
                    "golem_graph_transaction",
                    "create_vertex",
                    WrappedFunctionType::WriteRemote,
                );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner
                        .create_vertex(vertex_type.clone(), properties.clone())
                });
                durability.persist((vertex_type, properties), result)
            } else {
                durability.replay()
            }
        }

        fn is_active(&self) -> bool {
            self.inner.is_active()
        }

        fn get_vertex(
            &self,
            id: crate::golem::graph::types::ElementId,
        ) -> Result<Option<crate::golem::graph::types::Vertex>, GraphError> {
            self.inner.get_vertex(id)
        }

        fn create_vertex_with_labels(
            &self,
            vertex_type: String,
            additional_labels: Vec<String>,
            properties: crate::golem::graph::types::PropertyMap,
        ) -> Result<crate::golem::graph::types::Vertex, GraphError> {
            let durability: Durability<crate::golem::graph::types::Vertex, GraphError> =
                Durability::new(
                    "golem_graph_transaction",
                    "create_vertex_with_labels",
                    WrappedFunctionType::WriteRemote,
                );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.create_vertex_with_labels(
                        vertex_type.clone(),
                        additional_labels.clone(),
                        properties.clone(),
                    )
                });
                durability.persist((vertex_type, additional_labels, properties), result)
            } else {
                durability.replay()
            }
        }

        fn update_vertex(
            &self,
            id: crate::golem::graph::types::ElementId,
            properties: crate::golem::graph::types::PropertyMap,
        ) -> Result<crate::golem::graph::types::Vertex, GraphError> {
            let durability: Durability<crate::golem::graph::types::Vertex, GraphError> =
                Durability::new(
                    "golem_graph_transaction",
                    "update_vertex",
                    WrappedFunctionType::WriteRemote,
                );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.update_vertex(id.clone(), properties.clone())
                });
                durability.persist((id, properties), result)
            } else {
                durability.replay()
            }
        }

        fn update_vertex_properties(
            &self,
            id: crate::golem::graph::types::ElementId,
            updates: crate::golem::graph::types::PropertyMap,
        ) -> Result<crate::golem::graph::types::Vertex, GraphError> {
            let durability: Durability<crate::golem::graph::types::Vertex, GraphError> =
                Durability::new(
                    "golem_graph_transaction",
                    "update_vertex_properties",
                    WrappedFunctionType::WriteRemote,
                );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner
                        .update_vertex_properties(id.clone(), updates.clone())
                });
                durability.persist((id, updates), result)
            } else {
                durability.replay()
            }
        }

        fn delete_vertex(
            &self,
            id: crate::golem::graph::types::ElementId,
            delete_edges: bool,
        ) -> Result<(), GraphError> {
            let durability: Durability<Unit, GraphError> = Durability::new(
                "golem_graph_transaction",
                "delete_vertex",
                WrappedFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.delete_vertex(id.clone(), delete_edges)
                });
                durability.persist((id, delete_edges), result.map(|_| Unit))?;
                Ok(())
            } else {
                durability.replay::<Unit, GraphError>()?;
                Ok(())
            }
        }

        fn find_vertices(
            &self,
            vertex_type: Option<String>,
            filters: Option<Vec<crate::golem::graph::types::FilterCondition>>,
            sort: Option<Vec<crate::golem::graph::types::SortSpec>>,
            limit: Option<u32>,
            offset: Option<u32>,
        ) -> Result<Vec<crate::golem::graph::types::Vertex>, GraphError> {
            self.inner
                .find_vertices(vertex_type, filters, sort, limit, offset)
        }

        fn create_edge(
            &self,
            edge_type: String,
            from_vertex: crate::golem::graph::types::ElementId,
            to_vertex: crate::golem::graph::types::ElementId,
            properties: crate::golem::graph::types::PropertyMap,
        ) -> Result<crate::golem::graph::types::Edge, GraphError> {
            let durability: Durability<crate::golem::graph::types::Edge, GraphError> =
                Durability::new(
                    "golem_graph_transaction",
                    "create_edge",
                    WrappedFunctionType::WriteRemote,
                );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.create_edge(
                        edge_type.clone(),
                        from_vertex.clone(),
                        to_vertex.clone(),
                        properties.clone(),
                    )
                });
                durability.persist(
                    CreateEdgeParams {
                        edge_type,
                        from_vertex,
                        to_vertex,
                        properties,
                    },
                    result,
                )
            } else {
                durability.replay()
            }
        }

        fn get_edge(
            &self,
            id: crate::golem::graph::types::ElementId,
        ) -> Result<Option<crate::golem::graph::types::Edge>, GraphError> {
            self.inner.get_edge(id)
        }

        fn update_edge(
            &self,
            id: crate::golem::graph::types::ElementId,
            properties: crate::golem::graph::types::PropertyMap,
        ) -> Result<crate::golem::graph::types::Edge, GraphError> {
            let durability: Durability<crate::golem::graph::types::Edge, GraphError> =
                Durability::new(
                    "golem_graph_transaction",
                    "update_edge",
                    WrappedFunctionType::WriteRemote,
                );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.update_edge(id.clone(), properties.clone())
                });
                durability.persist((id, properties), result)
            } else {
                durability.replay()
            }
        }

        fn update_edge_properties(
            &self,
            id: crate::golem::graph::types::ElementId,
            updates: crate::golem::graph::types::PropertyMap,
        ) -> Result<crate::golem::graph::types::Edge, GraphError> {
            let durability: Durability<crate::golem::graph::types::Edge, GraphError> =
                Durability::new(
                    "golem_graph_transaction",
                    "update_edge_properties",
                    WrappedFunctionType::WriteRemote,
                );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner
                        .update_edge_properties(id.clone(), updates.clone())
                });
                durability.persist((id, updates), result)
            } else {
                durability.replay()
            }
        }

        fn delete_edge(&self, id: crate::golem::graph::types::ElementId) -> Result<(), GraphError> {
            let durability: Durability<Unit, GraphError> = Durability::new(
                "golem_graph_transaction",
                "delete_edge",
                WrappedFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.delete_edge(id.clone())
                });
                durability.persist(id, result.map(|_| Unit))?;
                Ok(())
            } else {
                durability.replay::<Unit, GraphError>()?;
                Ok(())
            }
        }

        fn find_edges(
            &self,
            edge_types: Option<Vec<String>>,
            filters: Option<Vec<crate::golem::graph::types::FilterCondition>>,
            sort: Option<Vec<crate::golem::graph::types::SortSpec>>,
            limit: Option<u32>,
            offset: Option<u32>,
        ) -> Result<Vec<crate::golem::graph::types::Edge>, GraphError> {
            self.inner
                .find_edges(edge_types, filters, sort, limit, offset)
        }

        fn get_adjacent_vertices(
            &self,
            vertex_id: crate::golem::graph::types::ElementId,
            direction: crate::golem::graph::types::Direction,
            edge_types: Option<Vec<String>>,
            limit: Option<u32>,
        ) -> Result<Vec<crate::golem::graph::types::Vertex>, GraphError> {
            self.inner
                .get_adjacent_vertices(vertex_id, direction, edge_types, limit)
        }

        fn get_connected_edges(
            &self,
            vertex_id: crate::golem::graph::types::ElementId,
            direction: crate::golem::graph::types::Direction,
            edge_types: Option<Vec<String>>,
            limit: Option<u32>,
        ) -> Result<Vec<crate::golem::graph::types::Edge>, GraphError> {
            self.inner
                .get_connected_edges(vertex_id, direction, edge_types, limit)
        }

        fn create_vertices(
            &self,
            vertices: Vec<crate::golem::graph::transactions::VertexSpec>,
        ) -> Result<Vec<crate::golem::graph::types::Vertex>, GraphError> {
            let durability: Durability<Vec<crate::golem::graph::types::Vertex>, GraphError> =
                Durability::new(
                    "golem_graph_transaction",
                    "create_vertices",
                    WrappedFunctionType::WriteRemote,
                );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.create_vertices(vertices.clone())
                });
                durability.persist(vertices, result)
            } else {
                durability.replay()
            }
        }

        fn create_edges(
            &self,
            edges: Vec<crate::golem::graph::transactions::EdgeSpec>,
        ) -> Result<Vec<crate::golem::graph::types::Edge>, GraphError> {
            let durability: Durability<Vec<crate::golem::graph::types::Edge>, GraphError> =
                Durability::new(
                    "golem_graph_transaction",
                    "create_edges",
                    WrappedFunctionType::WriteRemote,
                );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.create_edges(edges.clone())
                });
                durability.persist(edges, result)
            } else {
                durability.replay()
            }
        }

        fn upsert_vertex(
            &self,
            id: Option<crate::golem::graph::types::ElementId>,
            vertex_type: String,
            properties: crate::golem::graph::types::PropertyMap,
        ) -> Result<crate::golem::graph::types::Vertex, GraphError> {
            let durability: Durability<crate::golem::graph::types::Vertex, GraphError> =
                Durability::new(
                    "golem_graph_transaction",
                    "upsert_vertex",
                    WrappedFunctionType::WriteRemote,
                );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner
                        .upsert_vertex(id.clone(), vertex_type.clone(), properties.clone())
                });
                durability.persist((id, vertex_type, properties), result)
            } else {
                durability.replay()
            }
        }

        fn upsert_edge(
            &self,
            id: Option<crate::golem::graph::types::ElementId>,
            edge_type: String,
            from_vertex: crate::golem::graph::types::ElementId,
            to_vertex: crate::golem::graph::types::ElementId,
            properties: crate::golem::graph::types::PropertyMap,
        ) -> Result<crate::golem::graph::types::Edge, GraphError> {
            let durability: Durability<crate::golem::graph::types::Edge, GraphError> =
                Durability::new(
                    "golem_graph_transaction",
                    "upsert_edge",
                    WrappedFunctionType::WriteRemote,
                );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    self.inner.upsert_edge(
                        id.clone(),
                        edge_type.clone(),
                        from_vertex.clone(),
                        to_vertex.clone(),
                        properties.clone(),
                    )
                });
                durability.persist(
                    UpsertEdgeParams {
                        id,
                        edge_type,
                        from_vertex,
                        to_vertex,
                        properties,
                    },
                    result,
                )
            } else {
                durability.replay()
            }
        }
    }

    #[derive(Debug, Clone, FromValueAndType, IntoValue, PartialEq)]
    struct CreateEdgeParams {
        edge_type: String,
        from_vertex: crate::golem::graph::types::ElementId,
        to_vertex: crate::golem::graph::types::ElementId,
        properties: crate::golem::graph::types::PropertyMap,
    }

    #[derive(Debug, Clone, FromValueAndType, IntoValue, PartialEq)]
    struct UpsertEdgeParams {
        id: Option<crate::golem::graph::types::ElementId>,
        edge_type: String,
        from_vertex: crate::golem::graph::types::ElementId,
        to_vertex: crate::golem::graph::types::ElementId,
        properties: crate::golem::graph::types::PropertyMap,
    }

    #[derive(Debug, Clone, FromValueAndType, IntoValue, PartialEq)]
    struct ExecuteQueryParams {
        query: String,
        parameters: Option<Vec<(String, crate::golem::graph::types::PropertyValue)>>,
        options: Option<QueryOptions>,
    }
}

#[cfg(test)]
mod tests {
    use crate::golem::graph::{
        connection::ConnectionConfig,
        errors::GraphError,
        query::{QueryExecutionResult, QueryResult},
        transactions::{EdgeSpec, VertexSpec},
        types::{Edge, ElementId, Path, PropertyValue, Vertex},
    };
    use golem_rust::value_and_type::{FromValueAndType, IntoValueAndType};
    use std::fmt::Debug;

    fn roundtrip_test<T: Debug + Clone + PartialEq + IntoValueAndType + FromValueAndType>(
        value: T,
    ) {
        let vnt = value.clone().into_value_and_type();
        let extracted = T::from_value_and_type(vnt).unwrap();
        assert_eq!(value, extracted);
    }

    #[test]
    fn element_id_roundtrip() {
        roundtrip_test(ElementId::StringValue("test-id".to_string()));
        roundtrip_test(ElementId::Int64(12345));
        roundtrip_test(ElementId::Uuid(
            "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11".to_string(),
        ));
    }

    #[test]
    fn property_value_roundtrip() {
        roundtrip_test(PropertyValue::NullValue);
        roundtrip_test(PropertyValue::Boolean(true));
        roundtrip_test(PropertyValue::Int8(123));
        roundtrip_test(PropertyValue::Int16(12345));
        roundtrip_test(PropertyValue::Int32(12345678));
        roundtrip_test(PropertyValue::Int64(123456789012345));
        roundtrip_test(PropertyValue::Uint8(255));
        roundtrip_test(PropertyValue::Uint16(65535));
        roundtrip_test(PropertyValue::Uint32(1234567890));
        roundtrip_test(PropertyValue::Uint64(12345678901234567890));
        roundtrip_test(PropertyValue::Float32Value(123.456));
        roundtrip_test(PropertyValue::Float64Value(123.456789012345));
        roundtrip_test(PropertyValue::StringValue("hello world".to_string()));
        roundtrip_test(PropertyValue::Bytes(vec![1, 2, 3, 4, 5]));
    }

    #[test]
    fn graph_error_roundtrip() {
        roundtrip_test(GraphError::UnsupportedOperation(
            "This is not supported".to_string(),
        ));
        roundtrip_test(GraphError::ConnectionFailed(
            "Could not connect".to_string(),
        ));
        roundtrip_test(GraphError::ElementNotFound(ElementId::Int64(404)));
        roundtrip_test(GraphError::InvalidQuery("Syntax error".to_string()));
        roundtrip_test(GraphError::TransactionConflict);
    }

    #[test]
    fn vertex_and_edge_roundtrip() {
        let vertex = Vertex {
            id: ElementId::StringValue("v1".to_string()),
            vertex_type: "person".to_string(),
            additional_labels: vec!["employee".to_string()],
            properties: vec![],
        };
        roundtrip_test(vertex.clone());

        let edge = Edge {
            id: ElementId::Int64(101),
            edge_type: "knows".to_string(),
            from_vertex: ElementId::StringValue("v1".to_string()),
            to_vertex: ElementId::StringValue("v2".to_string()),
            properties: vec![],
        };
        roundtrip_test(edge);
    }

    #[test]
    fn specs_roundtrip() {
        let vertex_spec = VertexSpec {
            vertex_type: "company".to_string(),
            additional_labels: Some(vec!["startup".to_string()]),
            properties: vec![],
        };
        roundtrip_test(vertex_spec);

        let edge_spec = EdgeSpec {
            edge_type: "employs".to_string(),
            from_vertex: ElementId::StringValue("c1".to_string()),
            to_vertex: ElementId::StringValue("p1".to_string()),
            properties: vec![],
        };
        roundtrip_test(edge_spec);
    }

    #[test]
    fn query_result_roundtrip() {
        let vertex1 = Vertex {
            id: ElementId::StringValue("v1".to_string()),
            vertex_type: "person".to_string(),
            additional_labels: vec![],
            properties: vec![],
        };
        let vertex2 = Vertex {
            id: ElementId::StringValue("v2".to_string()),
            vertex_type: "person".to_string(),
            additional_labels: vec![],
            properties: vec![],
        };
        let edge = Edge {
            id: ElementId::Int64(1),
            edge_type: "knows".to_string(),
            from_vertex: ElementId::StringValue("v1".to_string()),
            to_vertex: ElementId::StringValue("v2".to_string()),
            properties: vec![],
        };
        let path = Path {
            vertices: vec![vertex1.clone(), vertex2.clone()],
            edges: vec![edge.clone()],
            length: 1,
        };

        let result_vertices = QueryExecutionResult {
            query_result_value: QueryResult::Vertices(vec![vertex1]),
            execution_time_ms: Some(10),
            rows_affected: Some(1),
            explanation: None,
            profile_data: None,
        };
        roundtrip_test(result_vertices);

        let result_edges = QueryExecutionResult {
            query_result_value: QueryResult::Edges(vec![edge]),
            execution_time_ms: Some(5),
            rows_affected: Some(1),
            explanation: None,
            profile_data: None,
        };
        roundtrip_test(result_edges);

        let result_paths = QueryExecutionResult {
            query_result_value: QueryResult::Paths(vec![path]),
            execution_time_ms: Some(20),
            rows_affected: Some(1),
            explanation: None,
            profile_data: None,
        };
        roundtrip_test(result_paths);

        let result_maps = QueryExecutionResult {
            query_result_value: QueryResult::Maps(vec![vec![
                (
                    "name".to_string(),
                    PropertyValue::StringValue("Alice".to_string()),
                ),
                ("age".to_string(), PropertyValue::Int32(30)),
            ]]),
            execution_time_ms: None,
            rows_affected: Some(1),
            explanation: None,
            profile_data: None,
        };
        roundtrip_test(result_maps);

        let result_values = QueryExecutionResult {
            query_result_value: QueryResult::Values(vec![PropertyValue::Int64(42)]),
            execution_time_ms: Some(1),
            rows_affected: Some(1),
            explanation: None,
            profile_data: None,
        };
        roundtrip_test(result_values);
    }

    #[test]
    fn connection_config_roundtrip() {
        let config = ConnectionConfig {
            hosts: vec!["localhost".to_string(), "golem.cloud".to_string()],
            port: Some(7687),
            database_name: Some("prod".to_string()),
            username: Some("user".to_string()),
            password: Some("pass".to_string()),
            timeout_seconds: Some(60),
            max_connections: Some(10),
            provider_config: vec![("retries".to_string(), "3".to_string())],
        };
        roundtrip_test(config);
    }
}
