use crate::helpers::parse_vertex_from_graph_data;
use crate::{
    helpers::{element_id_to_key, parse_path_from_data},
    GraphNeo4jComponent, Transaction,
};
use golem_graph::golem::graph::{
    errors::GraphError,
    traversal::{
        Direction, Guest as TraversalGuest, NeighborhoodOptions, Path, PathOptions, Subgraph,
    },
    types::{Edge, ElementId, Vertex},
};
use serde_json::json;
use std::collections::HashMap;

impl Transaction {
    pub fn find_shortest_path(
        &self,
        from_vertex: ElementId,
        to_vertex: ElementId,
        _options: Option<PathOptions>,
    ) -> Result<Option<Path>, GraphError> {
        // from_vertex/to_vertex are ElementId::StringValue(s)
        let from_id = match from_vertex {
            ElementId::StringValue(s) => s,
            _ => return Err(GraphError::InvalidQuery("expected string elementId".into())),
        };
        let to_id = match to_vertex {
            ElementId::StringValue(s) => s,
            _ => return Err(GraphError::InvalidQuery("expected string elementId".into())),
        };

        // Combine both matching strategies
        let statement = json!({
            "statement": r#"
                MATCH (a), (b)
                WHERE
                  (elementId(a) = $from_id OR id(a) = toInteger($from_id))
                  AND
                  (elementId(b) = $to_id   OR id(b) = toInteger($to_id))
                MATCH p = shortestPath((a)-[*]-(b))
                RETURN p
            "#,
            "resultDataContents": ["row","graph"],
            "parameters": { "from_id": from_id, "to_id": to_id }
        });

        let response = self.api.execute_in_transaction(
            &self.transaction_url,
            json!({
                "statements": [statement]
            }),
        )?;

        let result = response["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| GraphError::InternalError("Missing 'results'".into()))?;

        if let Some(errors) = result["errors"].as_array() {
            if !errors.is_empty() {
                return Err(GraphError::InternalError(format!(
                    "Neo4j error: {}",
                    errors[0]
                )));
            }
        }

        // If no row, return Ok(None)
        let data_opt = result["data"].as_array().and_then(|d| d.first());
        if let Some(data) = data_opt {
            let path = parse_path_from_data(data)?;
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
        let from_id = match from_vertex {
            ElementId::StringValue(s) => s,
            ElementId::Int64(i) => i.to_string(),
            ElementId::Uuid(u) => u,
        };

        let to_id = match to_vertex {
            ElementId::StringValue(s) => s,
            ElementId::Int64(i) => i.to_string(),
            ElementId::Uuid(u) => u,
        };

        let path_spec = match options {
            Some(opts) => {
                if opts.vertex_types.is_some()
                    || opts.vertex_filters.is_some()
                    || opts.edge_filters.is_some()
                {
                    return Err(GraphError::UnsupportedOperation(
                        "vertex_types, vertex_filters, and edge_filters are not yet supported in find_all_paths"
                            .to_string(),
                    ));
                }
                let edge_types = opts.edge_types.map_or("".to_string(), |types| {
                    if types.is_empty() {
                        "".to_string()
                    } else {
                        format!(":{}", types.join("|"))
                    }
                });
                let depth = opts
                    .max_depth
                    .map_or("*".to_string(), |d| format!("*1..{}", d));
                format!("-[{}]-", format_args!("r{}{}", edge_types, depth))
            }
            None => "-[*]-".to_string(),
        };

        let limit_clause = limit.map_or("".to_string(), |l| format!("LIMIT {}", l));
        let statement_str = format!(
            "MATCH p = (a){}(b) WHERE elementId(a) = $from_id AND elementId(b) = $to_id RETURN p {}",
            path_spec, limit_clause
        );

        let statement = json!({
            "statement": statement_str,
            "parameters": {
                "from_id": from_id,
                "to_id": to_id,
            },
            "resultDataContents": ["row","graph"]
        });

        let statements = json!({ "statements": [statement] });
        let response = self
            .api
            .execute_in_transaction(&self.transaction_url, statements)?;

        let result = response["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| {
                GraphError::InternalError(
                    "Invalid response from Neo4j for find_all_paths".to_string(),
                )
            })?;

        if let Some(errors) = result["errors"].as_array() {
            if !errors.is_empty() {
                return Err(GraphError::InvalidQuery(errors[0].to_string()));
            }
        }

        let empty_vec = vec![];
        let data = result["data"].as_array().unwrap_or(&empty_vec);
        let mut paths = Vec::new();
        for item in data {
            let path = parse_path_from_data(item)?;
            paths.push(path);
        }

        Ok(paths)
    }

    pub fn get_neighborhood(
        &self,
        center: ElementId,
        options: NeighborhoodOptions,
    ) -> Result<Subgraph, GraphError> {
        let center_id = match center {
            ElementId::StringValue(s) => s,
            ElementId::Int64(i) => i.to_string(),
            ElementId::Uuid(u) => u,
        };

        let (left_arrow, right_arrow) = match options.direction {
            Direction::Outgoing => ("", "->"),
            Direction::Incoming => ("<-", ""),
            Direction::Both => ("-", "-"),
        };

        let edge_type_str = options.edge_types.map_or("".to_string(), |types| {
            if types.is_empty() {
                "".to_string()
            } else {
                format!(":{}", types.join("|"))
            }
        });

        let depth = options.depth;
        let limit_clause = options
            .max_vertices
            .map_or("".to_string(), |l| format!("LIMIT {}", l));

        let full_query = format!(
            "MATCH p = (c){}[r{}*1..{}]{}(n)\
          WHERE ( elementId(c) = $id OR id(c) = toInteger($id) )\
          RETURN p {}",
            left_arrow, edge_type_str, depth, right_arrow, limit_clause
        );

        let statement = json!({
            "statement": full_query,
          "resultDataContents": ["row","graph"],
          "parameters": { "id": center_id }
        });

        let statements = json!({ "statements": [statement] });
        let response = self
            .api
            .execute_in_transaction(&self.transaction_url, statements)?;

        let result = response["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| {
                GraphError::InternalError(
                    "Invalid response from Neo4j for get_neighborhood".to_string(),
                )
            })?;

        if let Some(errors) = result["errors"].as_array() {
            if !errors.is_empty() {
                return Err(GraphError::InvalidQuery(errors[0].to_string()));
            }
        }

        let empty_vec = vec![];
        let data = result["data"].as_array().unwrap_or(&empty_vec);
        let mut all_vertices: HashMap<String, Vertex> = HashMap::new();
        let mut all_edges: HashMap<String, Edge> = HashMap::new();

        for item in data {
            let path = parse_path_from_data(item)?;
            for v in path.vertices {
                all_vertices.insert(element_id_to_key(&v.id), v);
            }
            for e in path.edges {
                all_edges.insert(element_id_to_key(&e.id), e);
            }
        }

        Ok(Subgraph {
            vertices: all_vertices.into_values().collect(),
            edges: all_edges.into_values().collect(),
        })
    }

    pub fn path_exists(
        &self,
        from_vertex: ElementId,
        to_vertex: ElementId,
        options: Option<PathOptions>,
    ) -> Result<bool, GraphError> {
        self.find_all_paths(from_vertex, to_vertex, options, Some(1))
            .map(|paths| !paths.is_empty())
    }

    pub fn get_vertices_at_distance(
        &self,
        source: ElementId,
        distance: u32,
        direction: Direction,
        edge_types: Option<Vec<String>>,
    ) -> Result<Vec<Vertex>, GraphError> {
        let source_id = element_id_to_key(&source);

        let (left_arrow, right_arrow) = match direction {
            Direction::Outgoing => ("", "->"),
            Direction::Incoming => ("<-", ""),
            Direction::Both => ("-", "-"),
        };

        let edge_type_str = edge_types.map_or("".to_string(), |types| {
            if types.is_empty() {
                "".to_string()
            } else {
                format!(":{}", types.join("|"))
            }
        });

        let query = format!(
            "MATCH (a){}[{}*{}]{}(b) WHERE elementId(a) = $id RETURN DISTINCT b",
            left_arrow, edge_type_str, distance, right_arrow
        );

        let statement = json!({
            "statement": query,
            "parameters": { "id": source_id }
        });

        let statements = json!({ "statements": [statement] });
        let response = self
            .api
            .execute_in_transaction(&self.transaction_url, statements)?;

        let result = response["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| {
                GraphError::InternalError(
                    "Invalid response from Neo4j for get_vertices_at_distance".to_string(),
                )
            })?;

        let empty_vec = vec![];
        let data = result["data"].as_array().unwrap_or(&empty_vec);
        let mut vertices = Vec::new();
        for item in data {
            if let Some(graph_node) = item["graph"]["nodes"].as_array().and_then(|n| n.first()) {
                let vertex = parse_vertex_from_graph_data(graph_node, None)?;
                vertices.push(vertex);
            }
        }

        Ok(vertices)
    }
}

impl TraversalGuest for GraphNeo4jComponent {
    fn find_shortest_path(
        transaction: golem_graph::golem::graph::transactions::TransactionBorrow<'_>,
        from_vertex: ElementId,
        to_vertex: ElementId,
        _options: Option<PathOptions>,
    ) -> Result<Option<Path>, GraphError> {
        let tx: &Transaction = transaction.get();
        tx.find_shortest_path(from_vertex, to_vertex, _options)
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
