use crate::{
    helpers::{element_id_to_key, parse_path_from_gremlin, parse_vertex_from_gremlin},
    GraphJanusGraphComponent, Transaction,
};
use golem_graph::golem::graph::{
    errors::GraphError,
    traversal::{
        Direction, Guest as TraversalGuest, NeighborhoodOptions, Path, PathOptions, Subgraph,
    },
    types::{ElementId, Vertex},
};
use serde_json::{json, Value};

fn id_to_json(id: ElementId) -> Value {
    match id {
        ElementId::StringValue(s) => json!(s),
        ElementId::Int64(i) => json!(i),
        ElementId::Uuid(u) => json!(u.to_string()),
    }
}

fn build_traversal_step(
    dir: &Direction,
    edge_types: &Option<Vec<String>>,
    bindings: &mut serde_json::Map<String, Value>,
) -> String {
    let step = match dir {
        Direction::Outgoing => "out",
        Direction::Incoming => "in",
        Direction::Both => "both",
    };
    if let Some(labels) = edge_types {
        if !labels.is_empty() {
            let key = format!("edge_labels_{}", bindings.len());
            bindings.insert(key.clone(), json!(labels));
            return format!("{}({})", step, key);
        }
    }
    format!("{}()", step)
}

impl Transaction {
    pub fn find_shortest_path(
        &self,
        from_vertex: ElementId,
        to_vertex: ElementId,
        options: Option<PathOptions>,
    ) -> Result<Option<Path>, GraphError> {
        let mut bindings = serde_json::Map::new();
        let edge_types = options.and_then(|o| o.edge_types);
        let step = build_traversal_step(&Direction::Both, &edge_types, &mut bindings);
        bindings.insert("from_id".to_string(), id_to_json(from_vertex));
        bindings.insert("to_id".to_string(), id_to_json(to_vertex));

        let gremlin = format!(
            "g.V(from_id).repeat({}.simplePath()).until(hasId(to_id)).path().limit(1)",
            step
        );
        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        if let Some(arr) = response["result"]["data"].as_array() {
            if let Some(path_val) = arr.first() {
                let path = parse_path_from_gremlin(path_val)?;
                return Ok(Some(path));
            }
        }
        Ok(None)
    }

    pub fn find_all_paths(
        &self,
        from_vertex: ElementId,
        to_vertex: ElementId,
        options: Option<PathOptions>,
        limit: Option<u32>,
    ) -> Result<Vec<Path>, GraphError> {
        // ←— Unsuppported‑options guard
        if let Some(opts) = &options {
            if opts.vertex_types.is_some()
                || opts.vertex_filters.is_some()
                || opts.edge_filters.is_some()
            {
                return Err(GraphError::UnsupportedOperation(
                    "vertex_types, vertex_filters, and edge_filters are not yet supported in find_all_paths"
                        .to_string(),
                ));
            }
        }

        let mut bindings = serde_json::Map::new();
        let edge_types = options.and_then(|o| o.edge_types);
        let step = build_traversal_step(&Direction::Both, &edge_types, &mut bindings);
        bindings.insert("from_id".to_string(), id_to_json(from_vertex));
        bindings.insert("to_id".to_string(), id_to_json(to_vertex));

        let mut gremlin = format!(
            "g.V(from_id).repeat({}.simplePath()).until(hasId(to_id)).path()",
            step
        );
        if let Some(lim) = limit {
            gremlin.push_str(&format!(".limit({})", lim));
        }

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;
        let data = response["result"]["data"].as_array().ok_or_else(|| {
            GraphError::InternalError("Invalid response for find_all_paths".to_string())
        })?;

        data.iter().map(parse_path_from_gremlin).collect()
    }

    pub fn get_neighborhood(
        &self,
        center: ElementId,
        options: NeighborhoodOptions,
    ) -> Result<Subgraph, GraphError> {
        let mut bindings = serde_json::Map::new();
        bindings.insert("center_id".to_string(), id_to_json(center.clone()));

        let edge_step = match options.direction {
            Direction::Outgoing => "outE",
            Direction::Incoming => "inE",
            Direction::Both => "bothE",
        };
        let mut gremlin = format!(
            "g.V(center_id).repeat({}().otherV().simplePath()).times({}).path()",
            edge_step, options.depth
        );
        if let Some(lim) = options.max_vertices {
            gremlin.push_str(&format!(".limit({})", lim));
        }

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;
        let data = response["result"]["data"].as_array().ok_or_else(|| {
            GraphError::InternalError("Invalid response for get_neighborhood".to_string())
        })?;

        let mut verts = std::collections::HashMap::new();
        let mut edges = std::collections::HashMap::new();
        for p in data {
            let path = parse_path_from_gremlin(p)?;
            for v in path.vertices {
                verts.insert(element_id_to_key(&v.id), v);
            }
            for e in path.edges {
                edges.insert(element_id_to_key(&e.id), e);
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
        self.find_all_paths(from_vertex, to_vertex, options, Some(1))
            .map(|p| !p.is_empty())
    }

    pub fn get_vertices_at_distance(
        &self,
        source: ElementId,
        distance: u32,
        direction: Direction,
        edge_types: Option<Vec<String>>,
    ) -> Result<Vec<Vertex>, GraphError> {
        let mut bindings = serde_json::Map::new();
        bindings.insert("source_id".to_string(), id_to_json(source));

        let step = match direction {
            Direction::Outgoing => "out",
            Direction::Incoming => "in",
            Direction::Both => "both",
        }
        .to_string();
        let label_key = if let Some(labels) = &edge_types {
            if !labels.is_empty() {
                bindings.insert("edge_labels".to_string(), json!(labels));
                "edge_labels".to_string()
            } else {
                "".to_string()
            }
        } else {
            "".to_string()
        };

        let gremlin = format!(
            "g.V(source_id).repeat({}({})).times({}).dedup().path()",
            step, label_key, distance
        );
        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let data = response["result"]["data"].as_array().ok_or_else(|| {
            GraphError::InternalError("Invalid response for get_vertices_at_distance".to_string())
        })?;
        let mut verts = Vec::new();
        for item in data {
            // Gremlin path returns a list: [v0, e0, v1, e1, ...]
            // We extract unique vertex elements via parse_vertex_from_gremlin on elementMap result
            if let Some(vmap) = item
                .as_array()
                .and_then(|arr| arr.iter().find(|x| x.is_object()))
            {
                verts.push(parse_vertex_from_gremlin(vmap)?);
            }
        }
        Ok(verts)
    }
}

impl TraversalGuest for GraphJanusGraphComponent {
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
    use crate::client::JanusGraphApi;
    use golem_graph::golem::graph::transactions::GuestTransaction;
    use golem_graph::golem::graph::types::PropertyValue;
    use std::sync::Arc;
    use std::{collections::HashMap, env};

    fn create_test_transaction() -> Transaction {
        let host = env::var("JANUSGRAPH_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port = env::var("JANUSGRAPH_PORT")
            .unwrap_or_else(|_| "8182".to_string())
            .parse()
            .unwrap();
        let api = JanusGraphApi::new(&host, port, None, None).unwrap();
        Transaction { api: Arc::new(api) }
    }

    fn setup_modern_graph(tx: &Transaction) -> HashMap<String, ElementId> {
        let mut ids = HashMap::new();
        let props = |_name, label: &str, val: &str| {
            (
                label.to_string(),
                PropertyValue::StringValue(val.to_string()),
            )
        };
        let marko = tx
            .create_vertex(
                "person".to_string(),
                vec![
                    props("name", "name", "marko"),
                    ("age".to_string(), PropertyValue::Int64(29)),
                ],
            )
            .unwrap();
        ids.insert("marko".to_string(), marko.id.clone());
        let vadas = tx
            .create_vertex(
                "person".to_string(),
                vec![
                    props("name", "name", "vadas"),
                    ("age".to_string(), PropertyValue::Int64(27)),
                ],
            )
            .unwrap();
        ids.insert("vadas".to_string(), vadas.id.clone());
        let josh = tx
            .create_vertex(
                "person".to_string(),
                vec![
                    props("name", "name", "josh"),
                    ("age".to_string(), PropertyValue::Int64(32)),
                ],
            )
            .unwrap();
        ids.insert("josh".to_string(), josh.id.clone());
        let peter = tx
            .create_vertex(
                "person".to_string(),
                vec![
                    props("name", "name", "peter"),
                    ("age".to_string(), PropertyValue::Int64(35)),
                ],
            )
            .unwrap();
        ids.insert("peter".to_string(), peter.id.clone());
        let lop = tx
            .create_vertex(
                "software".to_string(),
                vec![props("name", "name", "lop"), props("lang", "lang", "java")],
            )
            .unwrap();
        ids.insert("lop".to_string(), lop.id.clone());
        let ripple = tx
            .create_vertex(
                "software".to_string(),
                vec![props("name", "name", "ripple")],
            )
            .unwrap();
        ids.insert("ripple".to_string(), ripple.id.clone());

        tx.create_edge(
            "knows".to_string(),
            ids["marko"].clone(),
            ids["vadas"].clone(),
            vec![("weight".to_string(), PropertyValue::Float64(0.5))],
        )
        .unwrap();
        tx.create_edge(
            "knows".to_string(),
            ids["marko"].clone(),
            ids["josh"].clone(),
            vec![("weight".to_string(), PropertyValue::Float64(1.0))],
        )
        .unwrap();
        tx.create_edge(
            "created".to_string(),
            ids["marko"].clone(),
            ids["lop"].clone(),
            vec![("weight".to_string(), PropertyValue::Float64(0.4))],
        )
        .unwrap();
        tx.create_edge(
            "created".to_string(),
            ids["josh"].clone(),
            ids["ripple"].clone(),
            vec![("weight".to_string(), PropertyValue::Float64(1.0))],
        )
        .unwrap();
        tx.create_edge(
            "created".to_string(),
            ids["josh"].clone(),
            ids["lop"].clone(),
            vec![("weight".to_string(), PropertyValue::Float64(0.4))],
        )
        .unwrap();
        tx.create_edge(
            "created".to_string(),
            ids["peter"].clone(),
            ids["lop"].clone(),
            vec![("weight".to_string(), PropertyValue::Float64(0.2))],
        )
        .unwrap();
        ids
    }

    fn cleanup_modern_graph(tx: &Transaction) {
        tx.execute_query("g.V().hasLabel('person').drop()".to_string(), None, None)
            .unwrap();
        tx.execute_query("g.V().hasLabel('software').drop()".to_string(), None, None)
            .unwrap();
        tx.commit().unwrap();
    }

    #[test]
    fn test_find_shortest_path() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_find_shortest_path: JANUSGRAPH_HOST not set");
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
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_path_exists: JANUSGRAPH_HOST not set");
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
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_find_all_paths: JANUSGRAPH_HOST not set");
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
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_get_neighborhood: JANUSGRAPH_HOST not set");
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
        assert_eq!(sub.vertices.len(), 4);
        assert_eq!(sub.edges.len(), 3);

        cleanup_modern_graph(&tx);
    }

    #[test]
    fn test_get_vertices_at_distance() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_get_vertices_at_distance: JANUSGRAPH_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        let ids = setup_modern_graph(&tx);

        let verts = tx
            .get_vertices_at_distance(ids["marko"].clone(), 2, Direction::Outgoing, None)
            .unwrap();
        assert_eq!(verts.len(), 2);

        cleanup_modern_graph(&tx);
    }

    #[test]
    fn test_unsupported_path_options() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_unsupported_path_options: JANUSGRAPH_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        let ids = setup_modern_graph(&tx);

        let options = PathOptions {
            vertex_types: Some(vec!["person".to_string()]),
            edge_types: None,
            max_depth: None,
            vertex_filters: None,
            edge_filters: None,
        };

        let result = tx.find_all_paths(
            ids["marko"].clone(),
            ids["lop"].clone(),
            Some(options),
            None,
        );
        assert!(matches!(result, Err(GraphError::UnsupportedOperation(_))));

        cleanup_modern_graph(&tx);
    }
}
