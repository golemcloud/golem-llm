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
        let edge_collections = options.and_then(|o| o.edge_types).unwrap_or_default();

        let edge_collections_str = if edge_collections.is_empty() {
            "knows, created".to_string()
        } else {
            edge_collections.join(", ")
        };

        let query_str = format!(
            "FOR vertex, edge IN ANY SHORTEST_PATH @from_id TO @to_id {} RETURN {{vertex: vertex, edge: edge}}",
            edge_collections_str
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

        if arr.is_empty() {
            return Ok(None);
        }

        // Build vertices and edges from the traversal result
        let mut vertices = vec![];
        let mut edges = vec![];

        for item in arr {
            if let Some(obj) = item.as_object() {
                if let Some(v_doc) = obj.get("vertex").and_then(|v| v.as_object()) {
                    let coll = v_doc
                        .get("_id")
                        .and_then(|id| id.as_str())
                        .and_then(|s| s.split('/').next())
                        .unwrap_or_default();
                    let vertex = parse_vertex_from_document(v_doc, coll)?;
                    vertices.push(vertex);
                }
                if let Some(e_doc) = obj.get("edge").and_then(|e| e.as_object()) {
                    let coll = e_doc
                        .get("_id")
                        .and_then(|id| id.as_str())
                        .and_then(|s| s.split('/').next())
                        .unwrap_or_default();
                    let edge = parse_edge_from_document(e_doc, coll)?;
                    edges.push(edge);
                }
            }
        }

        let length = edges.len() as u32;
        Ok(Some(Path {
            vertices,
            edges,
            length,
        }))
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
        let edge_collections = options.and_then(|o| o.edge_types).unwrap_or_default();

        let edge_collections_str = if edge_collections.is_empty() {
            "knows, created".to_string()
        } else {
            edge_collections.join(", ")
        };
        let limit_clause = limit.map_or(String::new(), |l| format!("LIMIT {}", l));

        let query_str = format!(
            "FOR v, e, p IN {}..{} OUTBOUND @from_id {} OPTIONS {{uniqueVertices: 'path'}} FILTER v._id == @to_id {} RETURN {{vertices: p.vertices, edges: p.edges}}",
            min_depth, max_depth, edge_collections_str, limit_clause
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
        let edge_collections = options.edge_types.unwrap_or_default();
        let edge_collections_str = if edge_collections.is_empty() {
            "knows, created".to_string()
        } else {
            edge_collections.join(", ")
        };
        let limit_clause = options
            .max_vertices
            .map_or(String::new(), |l| format!("LIMIT {}", l));

        let query_str = format!(
            "FOR v, e IN 1..{} {} @center_id {} {} RETURN {{vertex: v, edge: e}}",
            options.depth, dir_str, edge_collections_str, limit_clause
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
        let edge_collections = edge_types.unwrap_or_default();
        let edge_collections_str = if edge_collections.is_empty() {
            "knows, created".to_string()
        } else {
            edge_collections.join(", ")
        };

        let query_str = format!(
            "FOR v IN {}..{} {} @start {} RETURN v",
            distance, distance, dir_str, edge_collections_str
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
