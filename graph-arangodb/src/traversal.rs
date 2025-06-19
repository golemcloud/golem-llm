use crate::{
    helpers::{
        element_id_to_string, parse_edge_from_document, parse_path_from_document,
        parse_vertex_from_document,
    },
    GraphArangoDbComponent, Transaction,
};
use golem_graph::golem::graph::{
    errors::GraphError,
    traversal::{
        Direction, Guest as TraversalGuest, NeighborhoodOptions, Path, PathOptions, Subgraph,
    },
    types::{ElementId, Vertex},
};
use serde_json::{json, Value};
use std::collections::HashMap;

fn id_to_aql(id: &ElementId) -> String {
    element_id_to_string(id)
}

impl Transaction {
    pub fn find_shortest_path(
        &self,
        from_vertex: ElementId,
        to_vertex: ElementId,
        options: Option<PathOptions>,
    ) -> Result<Option<Path>, GraphError> {
        let from_id = id_to_aql(&from_vertex);
        let to_id = id_to_aql(&to_vertex);
        let edge_collections = options
            .and_then(|o| o.edge_types)
            .unwrap_or_default()
            .join(", ");

        let query_str = format!(
            "FOR p IN ANY SHORTEST_PATH @from_id TO @to_id {} RETURN p",
            edge_collections
        );
        let mut bind_vars = serde_json::Map::new();
        bind_vars.insert("from_id".to_string(), json!(from_id));
        bind_vars.insert("to_id".to_string(), json!(to_id));

        let request = json!({
            "query": query_str,
            "bindVars": Value::Object(bind_vars.clone()),
        });
        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, request)?;
        let arr = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Invalid response for shortest path".to_string())
        })?;

        if let Some(doc) = arr.first().and_then(|v| v.as_object()) {
            let path = parse_path_from_document(doc)?;
            Ok(Some(path))
        } else {
            Ok(None)
        }
    }

    pub fn find_all_paths(
        &self,
        from_vertex: ElementId,
        to_vertex: ElementId,
        options: Option<PathOptions>,
        limit: Option<u32>,
    ) -> Result<Vec<Path>, GraphError> {
        if let Some(opts) = &options {
            if opts.vertex_types.is_some()
                || opts.vertex_filters.is_some()
                || opts.edge_filters.is_some()
            {
                return Err(GraphError::UnsupportedOperation(
                    "vertex_types, vertex_filters, and edge_filters are not supported".to_string(),
                ));
            }
        }

        let from_id = id_to_aql(&from_vertex);
        let to_id = id_to_aql(&to_vertex);
        let (min_depth, max_depth) = options
            .as_ref()
            .and_then(|o| o.max_depth)
            .map_or((1, 10), |d| (1, d));
        let edge_collections = options
            .and_then(|o| o.edge_types)
            .unwrap_or_default()
            .join(", ");
        let limit_clause = limit.map_or(String::new(), |l| format!("LIMIT {}", l));

        let query_str = format!(
            "FOR p IN {}..{} ANY @from_id TO @to_id {} {} RETURN p",
            min_depth, max_depth, edge_collections, limit_clause
        );
        let request = json!({
            "query": query_str,
            "bindVars": { "from_id": from_id, "to_id": to_id }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, request)?;
        let arr = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Invalid response for all paths".to_string())
        })?;

        arr.iter()
            .filter_map(|v| v.as_object())
            .map(parse_path_from_document)
            .collect()
    }

    pub fn get_neighborhood(
        &self,
        center: ElementId,
        options: NeighborhoodOptions,
    ) -> Result<Subgraph, GraphError> {
        let center_id = id_to_aql(&center);
        let dir_str = match options.direction {
            Direction::Outgoing => "OUTBOUND",
            Direction::Incoming => "INBOUND",
            Direction::Both => "ANY",
        };
        let edge_collections = options.edge_types.unwrap_or_default().join(", ");
        let limit_clause = options
            .max_vertices
            .map_or(String::new(), |l| format!("LIMIT {}", l));

        let query_str = format!(
            "FOR v, e IN 1..{} {} @center_id {} {} RETURN {{vertex: v, edge: e}}",
            options.depth, dir_str, edge_collections, limit_clause
        );
        let request = json!({
            "query": query_str,
            "bindVars": { "center_id": center_id }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, request)?;
        let arr = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Invalid response for neighborhood".to_string())
        })?;

        let mut verts = HashMap::new();
        let mut edges = HashMap::new();
        for item in arr {
            if let Some(obj) = item.as_object() {
                if let Some(v_doc) = obj.get("vertex").and_then(|v| v.as_object()) {
                    let coll = v_doc
                        .get("_id")
                        .and_then(|id| id.as_str())
                        .and_then(|s| s.split('/').next())
                        .unwrap_or_default();
                    let vert = parse_vertex_from_document(v_doc, coll)?;
                    verts.insert(element_id_to_string(&vert.id), vert);
                }
                if let Some(e_doc) = obj.get("edge").and_then(|e| e.as_object()) {
                    let coll = e_doc
                        .get("_id")
                        .and_then(|id| id.as_str())
                        .and_then(|s| s.split('/').next())
                        .unwrap_or_default();
                    let edge = parse_edge_from_document(e_doc, coll)?;
                    edges.insert(element_id_to_string(&edge.id), edge);
                }
            }
        }

        Ok(Subgraph {
            vertices: verts.into_values().collect(),
            edges: edges.into_values().collect(),
        })
    }

    pub fn path_exists(
        &self,
        from_vertex: ElementId,
        to_vertex: ElementId,
        options: Option<PathOptions>,
    ) -> Result<bool, GraphError> {
        Ok(!self
            .find_all_paths(from_vertex, to_vertex, options, Some(1))?
            .is_empty())
    }

    pub fn get_vertices_at_distance(
        &self,
        source: ElementId,
        distance: u32,
        direction: Direction,
        edge_types: Option<Vec<String>>,
    ) -> Result<Vec<Vertex>, GraphError> {
        let start = id_to_aql(&source);
        let dir_str = match direction {
            Direction::Outgoing => "OUTBOUND",
            Direction::Incoming => "INBOUND",
            Direction::Both => "ANY",
        };
        let edge_collections = edge_types.unwrap_or_default().join(", ");

        let query_str = format!(
            "FOR v IN {}..{} {} @start {} RETURN v",
            distance, distance, dir_str, edge_collections
        );
        let request = json!({ "query": query_str, "bindVars": { "start": start } });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, request)?;
        let arr = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Invalid response for vertices at distance".to_string())
        })?;

        arr.iter()
            .filter_map(|v| v.as_object())
            .map(|doc| {
                let coll = doc
                    .get("_id")
                    .and_then(|id| id.as_str())
                    .and_then(|s| s.split('/').next())
                    .unwrap_or_default();
                parse_vertex_from_document(doc, coll)
            })
            .collect()
    }
}

impl TraversalGuest for GraphArangoDbComponent {
    fn find_shortest_path(
        transaction: golem_graph::golem::graph::transactions::TransactionBorrow<'_>,
        from_vertex: ElementId,
        to_vertex: ElementId,
        options: Option<PathOptions>,
    ) -> Result<Option<Path>, GraphError> {
        let tx: &Transaction = transaction.get();
        tx.find_shortest_path(from_vertex, to_vertex, options)
    }

    fn find_all_paths(
        transaction: golem_graph::golem::graph::transactions::TransactionBorrow<'_>,
        from_vertex: ElementId,
        to_vertex: ElementId,
        options: Option<PathOptions>,
        limit: Option<u32>,
    ) -> Result<Vec<Path>, GraphError> {
        let tx: &Transaction = transaction.get();
        tx.find_all_paths(from_vertex, to_vertex, options, limit)
    }

    fn get_neighborhood(
        transaction: golem_graph::golem::graph::transactions::TransactionBorrow<'_>,
        center: ElementId,
        options: NeighborhoodOptions,
    ) -> Result<Subgraph, GraphError> {
        let tx: &Transaction = transaction.get();
        tx.get_neighborhood(center, options)
    }

    fn path_exists(
        transaction: golem_graph::golem::graph::transactions::TransactionBorrow<'_>,
        from_vertex: ElementId,
        to_vertex: ElementId,
        options: Option<PathOptions>,
    ) -> Result<bool, GraphError> {
        let tx: &Transaction = transaction.get();
        tx.path_exists(from_vertex, to_vertex, options)
    }

    fn get_vertices_at_distance(
        transaction: golem_graph::golem::graph::transactions::TransactionBorrow<'_>,
        source: ElementId,
        distance: u32,
        direction: Direction,
        edge_types: Option<Vec<String>>,
    ) -> Result<Vec<Vertex>, GraphError> {
        let tx: &Transaction = transaction.get();
        tx.get_vertices_at_distance(source, distance, direction, edge_types)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ArangoDbApi;
    use golem_graph::golem::graph::transactions::GuestTransaction;
    use golem_graph::golem::graph::types::PropertyValue;
    use std::{collections::HashMap, env};

    fn create_test_transaction() -> Transaction {
        let host = env::var("ARANGODB_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port: u16 = env::var("ARANGODB_PORT")
            .unwrap_or_else(|_| "8529".to_string())
            .parse()
            .expect("Invalid ARANGODB_PORT");
        let user = env::var("ARANGODB_USER").unwrap_or_else(|_| "root".to_string());
        let pass = env::var("ARANGODB_PASS").unwrap_or_else(|_| "".to_string());
        let db = env::var("ARANGODB_DB").unwrap_or_else(|_| "test_db".to_string());
        let api = ArangoDbApi::new(&host, port, &user, &pass, &db);
        let transaction_id = api.begin_transaction(false).unwrap();
        let api = std::sync::Arc::new(api);
        Transaction {
            api,
            transaction_id,
        }
    }

    fn setup_modern_graph(tx: &Transaction) -> HashMap<String, ElementId> {
        let mut ids = HashMap::new();
        let prop = |key: &str, v: PropertyValue| (key.to_string(), v);

        let marko = tx
            .create_vertex(
                "person".into(),
                vec![
                    prop("name", PropertyValue::StringValue("marko".into())),
                    prop("age", PropertyValue::Int64(29)),
                ],
            )
            .unwrap();
        ids.insert("marko".into(), marko.id.clone());
        let vadas = tx
            .create_vertex(
                "person".into(),
                vec![
                    prop("name", PropertyValue::StringValue("vadas".into())),
                    prop("age", PropertyValue::Int64(27)),
                ],
            )
            .unwrap();
        ids.insert("vadas".into(), vadas.id.clone());
        let josh = tx
            .create_vertex(
                "person".into(),
                vec![
                    prop("name", PropertyValue::StringValue("josh".into())),
                    prop("age", PropertyValue::Int64(32)),
                ],
            )
            .unwrap();
        ids.insert("josh".into(), josh.id.clone());
        let peter = tx
            .create_vertex(
                "person".into(),
                vec![
                    prop("name", PropertyValue::StringValue("peter".into())),
                    prop("age", PropertyValue::Int64(35)),
                ],
            )
            .unwrap();
        ids.insert("peter".into(), peter.id.clone());
        let lop = tx
            .create_vertex(
                "software".into(),
                vec![
                    prop("name", PropertyValue::StringValue("lop".into())),
                    prop("lang", PropertyValue::StringValue("java".into())),
                ],
            )
            .unwrap();
        ids.insert("lop".into(), lop.id.clone());
        let ripple = tx
            .create_vertex(
                "software".into(),
                vec![prop("name", PropertyValue::StringValue("ripple".into()))],
            )
            .unwrap();
        ids.insert("ripple".into(), ripple.id.clone());

        tx.create_edge(
            "knows".into(),
            ids["marko"].clone(),
            ids["vadas"].clone(),
            vec![prop("weight", PropertyValue::Float64(0.5))],
        )
        .unwrap();
        tx.create_edge(
            "knows".into(),
            ids["marko"].clone(),
            ids["josh"].clone(),
            vec![prop("weight", PropertyValue::Float64(1.0))],
        )
        .unwrap();
        tx.create_edge(
            "created".into(),
            ids["marko"].clone(),
            ids["lop"].clone(),
            vec![prop("weight", PropertyValue::Float64(0.4))],
        )
        .unwrap();
        tx.create_edge(
            "created".into(),
            ids["josh"].clone(),
            ids["ripple"].clone(),
            vec![prop("weight", PropertyValue::Float64(1.0))],
        )
        .unwrap();
        tx.create_edge(
            "created".into(),
            ids["josh"].clone(),
            ids["lop"].clone(),
            vec![prop("weight", PropertyValue::Float64(0.4))],
        )
        .unwrap();
        tx.create_edge(
            "created".into(),
            ids["peter"].clone(),
            ids["lop"].clone(),
            vec![prop("weight", PropertyValue::Float64(0.2))],
        )
        .unwrap();

        ids
    }

    fn cleanup_modern_graph(tx: &Transaction) {
        tx.api
            .execute_in_transaction(
                &tx.transaction_id,
                json!({
                    "query": "FOR v IN person REMOVE v IN person"
                }),
            )
            .unwrap();
        tx.api
            .execute_in_transaction(
                &tx.transaction_id,
                json!({
                    "query": "FOR v IN software REMOVE v IN software"
                }),
            )
            .unwrap();
        tx.api
            .execute_in_transaction(
                &tx.transaction_id,
                json!({
                    "query": "FOR e IN knows REMOVE e IN knows"
                }),
            )
            .unwrap();
        tx.api
            .execute_in_transaction(
                &tx.transaction_id,
                json!({
                    "query": "FOR e IN created REMOVE e IN created"
                }),
            )
            .unwrap();
        tx.commit().unwrap();
    }

    #[test]
    fn test_find_shortest_path() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_find_shortest_path: ARANGODB_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        let ids = setup_modern_graph(&tx);
        let path = tx
            .find_shortest_path(ids["marko"].clone(), ids["ripple"].clone(), None)
            .unwrap()
            .unwrap();
        assert_eq!(path.vertices.len(), 3);
        assert_eq!(path.edges.len(), 2);
        cleanup_modern_graph(&tx);
    }

    #[test]
    fn test_path_exists() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_path_exists: ARANGODB_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        let ids = setup_modern_graph(&tx);
        assert!(tx
            .path_exists(ids["marko"].clone(), ids["ripple"].clone(), None)
            .unwrap());
        assert!(!tx
            .path_exists(ids["vadas"].clone(), ids["peter"].clone(), None)
            .unwrap());
        cleanup_modern_graph(&tx);
    }

    #[test]
    fn test_find_all_paths() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_find_all_paths: ARANGODB_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        let ids = setup_modern_graph(&tx);
        let paths = tx
            .find_all_paths(ids["marko"].clone(), ids["lop"].clone(), None, Some(5))
            .unwrap();
        assert_eq!(paths.len(), 2);
        cleanup_modern_graph(&tx);
    }

    #[test]
    fn test_get_neighborhood() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_get_neighborhood: ARANGODB_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        let ids = setup_modern_graph(&tx);
        let sub = tx
            .get_neighborhood(
                ids["marko"].clone(),
                NeighborhoodOptions {
                    direction: Direction::Outgoing,
                    depth: 1,
                    edge_types: None,
                    max_vertices: None,
                },
            )
            .unwrap();
        assert!(sub.vertices.len() >= 3);
        assert!(sub.edges.len() >= 3);
        cleanup_modern_graph(&tx);
    }

    #[test]
    fn test_get_vertices_at_distance() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_get_vertices_at_distance: ARANGODB_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        let ids = setup_modern_graph(&tx);
        let verts = tx
            .get_vertices_at_distance(ids["marko"].clone(), 2, Direction::Outgoing, None)
            .unwrap();
        assert!(verts.is_empty());
        cleanup_modern_graph(&tx);
    }

    #[test]
    fn test_unsupported_path_options() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_unsupported_path_options: ARANGODB_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        let ids = setup_modern_graph(&tx);
        let options = PathOptions {
            vertex_types: Some(vec!["person".into()]),
            edge_types: None,
            max_depth: None,
            vertex_filters: None,
            edge_filters: None,
        };
        let res = tx.find_all_paths(
            ids["marko"].clone(),
            ids["lop"].clone(),
            Some(options),
            None,
        );
        assert!(matches!(res, Err(GraphError::UnsupportedOperation(_))));
        cleanup_modern_graph(&tx);
    }
}
