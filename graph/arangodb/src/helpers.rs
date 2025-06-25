use golem_graph::golem::graph::{
    connection::ConnectionConfig,
    errors::GraphError,
    types::{Edge, ElementId, Path, Vertex},
};
use serde_json::{Map, Value};
use std::env;

use crate::conversions;
use crate::helpers;

pub(crate) fn parse_vertex_from_document(
    doc: &Map<String, Value>,
    collection: &str,
) -> Result<Vertex, GraphError> {
    let id_str = doc
        .get("_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| GraphError::InternalError("Missing _id in vertex document".to_string()))?;

    let mut properties = Map::new();
    for (key, value) in doc {
        if !key.starts_with('_') {
            properties.insert(key.clone(), value.clone());
        }
    }

    let additional_labels = if let Some(labels_val) = doc.get("_additional_labels") {
        serde_json::from_value(labels_val.clone()).unwrap_or_else(|_| vec![])
    } else {
        vec![]
    };

    Ok(Vertex {
        id: ElementId::StringValue(id_str.to_string()),
        vertex_type: collection.to_string(),
        additional_labels,
        properties: conversions::from_arango_properties(properties)?,
    })
}

pub(crate) fn parse_edge_from_document(
    doc: &Map<String, Value>,
    collection: &str,
) -> Result<Edge, GraphError> {
    let id_str = doc
        .get("_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| GraphError::InternalError("Missing _id in edge document".to_string()))?;

    let from_str = doc
        .get("_from")
        .and_then(|v| v.as_str())
        .ok_or_else(|| GraphError::InternalError("Missing _from in edge document".to_string()))?;

    let to_str = doc
        .get("_to")
        .and_then(|v| v.as_str())
        .ok_or_else(|| GraphError::InternalError("Missing _to in edge document".to_string()))?;

    let mut properties = Map::new();
    for (key, value) in doc {
        if !key.starts_with('_') {
            properties.insert(key.clone(), value.clone());
        }
    }

    Ok(Edge {
        id: ElementId::StringValue(id_str.to_string()),
        edge_type: collection.to_string(),
        from_vertex: ElementId::StringValue(from_str.to_string()),
        to_vertex: ElementId::StringValue(to_str.to_string()),
        properties: conversions::from_arango_properties(properties)?,
    })
}

pub(crate) fn parse_path_from_document(doc: &Map<String, Value>) -> Result<Path, GraphError> {
    let vertices_val = doc
        .get("vertices")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            GraphError::InternalError("Missing or invalid 'vertices' in path result".to_string())
        })?;
    let edges_val = doc.get("edges").and_then(|e| e.as_array()).ok_or_else(|| {
        GraphError::InternalError("Missing or invalid 'edges' in path result".to_string())
    })?;

    let mut vertices = vec![];
    for v_val in vertices_val {
        if let Some(v_doc) = v_val.as_object() {
            let collection = v_doc
                .get("_id")
                .and_then(|v| v.as_str())
                .and_then(|s| s.split('/').next())
                .unwrap_or_default();
            vertices.push(helpers::parse_vertex_from_document(v_doc, collection)?);
        }
    }

    let mut edges = vec![];
    for e_val in edges_val {
        if let Some(e_doc) = e_val.as_object() {
            let collection = e_doc
                .get("_id")
                .and_then(|v| v.as_str())
                .and_then(|s| s.split('/').next())
                .unwrap_or_default();
            edges.push(helpers::parse_edge_from_document(e_doc, collection)?);
        }
    }

    Ok(Path {
        vertices,
        edges,
        length: edges_val.len() as u32,
    })
}

pub(crate) fn element_id_to_key(id: &ElementId) -> Result<String, GraphError> {
    match id {
        ElementId::StringValue(s) => {
            // ArangoDB document keys are part of the _id field, e.g., "collection/key"
            if let Some(key) = s.split('/').nth(1) {
                Ok(key.to_string())
            } else {
                Ok(s.clone())
            }
        }
        _ => Err(GraphError::InvalidQuery(
            "ArangoDB only supports string-based element IDs".to_string(),
        )),
    }
}

pub(crate) fn collection_from_element_id(id: &ElementId) -> Result<&str, GraphError> {
    match id {
        ElementId::StringValue(s) => s.split('/').next().ok_or_else(|| {
            GraphError::InvalidQuery(
                "ElementId must be a full _id string (e.g., 'collection/key')".to_string(),
            )
        }),
        _ => Err(GraphError::InvalidQuery(
            "ArangoDB only supports string-based element IDs".to_string(),
        )),
    }
}

pub(crate) fn element_id_to_string(id: &ElementId) -> String {
    match id {
        ElementId::StringValue(s) => s.clone(),
        ElementId::Int64(i) => i.to_string(),
        ElementId::Uuid(u) => u.clone(),
    }
}

pub(crate) fn config_from_env() -> Result<ConnectionConfig, GraphError> {
    let host = env::var("ARANGO_HOST")
        .or_else(|_| env::var("ARANGODB_HOST"))
        .map_err(|_| {
            GraphError::ConnectionFailed("Missing ARANGO_HOST or ARANGODB_HOST env var".to_string())
        })?;
    let port = env::var("ARANGO_PORT")
        .or_else(|_| env::var("ARANGODB_PORT"))
        .map_or(Ok(None), |p| {
            p.parse::<u16>().map(Some).map_err(|e| {
                GraphError::ConnectionFailed(format!("Invalid ARANGO_PORT/ARANGODB_PORT: {}", e))
            })
        })?;
    let username = env::var("ARANGO_USER")
        .or_else(|_| env::var("ARANGODB_USER"))
        .map_err(|_| {
            GraphError::ConnectionFailed("Missing ARANGO_USER or ARANGODB_USER env var".to_string())
        })?;
    let password = env::var("ARANGO_PASSWORD")
        .or_else(|_| env::var("ARANGODB_PASSWORD"))
        .map_err(|_| {
            GraphError::ConnectionFailed(
                "Missing ARANGO_PASSWORD or ARANGODB_PASSWORD env var".to_string(),
            )
        })?;
    let database_name = env::var("ARANGO_DATABASE")
        .or_else(|_| env::var("ARANGODB_DATABASE"))
        .ok();

    Ok(ConnectionConfig {
        hosts: vec![host],
        port,
        database_name,
        username: Some(username),
        password: Some(password),
        timeout_seconds: None,
        max_connections: None,
        provider_config: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use golem_graph::golem::graph::errors::GraphError;
    use golem_graph::golem::graph::types::ElementId;
    use serde_json::{json, Map, Value};
    use std::env;

    /// Helper to construct a JSON document map
    fn make_doc(map: Vec<(&str, Value)>) -> Map<String, Value> {
        let mut m = Map::new();
        for (k, v) in map {
            m.insert(k.to_string(), v);
        }
        m
    }

    #[test]
    fn test_parse_vertex_from_document_basic() {
        let doc = make_doc(vec![
            ("_id", json!("users/alice")),
            ("name", json!("Alice")),
            ("age", json!(30)),
        ]);
        let vertex = parse_vertex_from_document(&doc, "users").unwrap();
        assert_eq!(vertex.id, ElementId::StringValue("users/alice".to_string()));
        assert_eq!(vertex.vertex_type, "users");
        assert!(vertex.additional_labels.is_empty());
        assert_eq!(vertex.properties.len(), 2);
    }

    #[test]
    fn test_parse_vertex_with_additional_labels() {
        let labels = json!(["VIP", "Premium"]);
        let doc = make_doc(vec![
            ("_id", json!("customers/bob")),
            ("_additional_labels", labels.clone()),
            ("score", json!(99)),
        ]);
        let vertex = parse_vertex_from_document(&doc, "customers").unwrap();
        assert_eq!(vertex.additional_labels, vec!["VIP", "Premium"]);
        assert_eq!(vertex.properties.len(), 1);
    }

    #[test]
    fn test_parse_edge_from_document_basic() {
        let doc = make_doc(vec![
            ("_id", json!("knows/e1")),
            ("_from", json!("users/alice")),
            ("_to", json!("users/bob")),
            ("since", json!(2021)),
        ]);
        let edge = parse_edge_from_document(&doc, "knows").unwrap();
        assert_eq!(edge.id, ElementId::StringValue("knows/e1".to_string()));
        assert_eq!(edge.edge_type, "knows");
        assert_eq!(
            edge.from_vertex,
            ElementId::StringValue("users/alice".to_string())
        );
        assert_eq!(
            edge.to_vertex,
            ElementId::StringValue("users/bob".to_string())
        );
        assert_eq!(edge.properties.len(), 1);
    }

    #[test]
    fn test_parse_path_from_document() {
        let v1 = json!({"_id": "vcol/v1", "name": "V1"});
        let v2 = json!({"_id": "vcol/v2", "name": "V2"});
        let e1 = json!({"_id": "ecol/e1", "_from": "vcol/v1", "_to": "vcol/v2", "rel": "connects"});
        let path_doc = make_doc(vec![
            ("vertices", Value::Array(vec![v1, v2])),
            ("edges", Value::Array(vec![e1])),
        ]);
        let path = parse_path_from_document(&path_doc).unwrap();
        assert_eq!(path.vertices.len(), 2);
        assert_eq!(path.edges.len(), 1);
        assert_eq!(path.length, 1);
    }

    #[test]
    fn test_element_id_to_key_and_collection() {
        let full_id = ElementId::StringValue("col/key123".to_string());
        let key = element_id_to_key(&full_id).unwrap();
        assert_eq!(key, "key123");
        let collection = collection_from_element_id(&full_id).unwrap();
        assert_eq!(collection, "col");

        let int_id = ElementId::Int64(10);
        assert!(element_id_to_key(&int_id).is_err());
        assert!(collection_from_element_id(&int_id).is_err());
    }

    #[test]
    fn test_element_id_to_string() {
        let s = ElementId::StringValue("col/1".to_string());
        let i = ElementId::Int64(42);
        let u = ElementId::Uuid("uuid-1234".to_string());
        assert_eq!(element_id_to_string(&s), "col/1");
        assert_eq!(element_id_to_string(&i), "42");
        assert_eq!(element_id_to_string(&u), "uuid-1234");
    }

    #[test]
    fn test_config_from_env_success_and_failure() {
        // Preserve original environment variables
        let orig_host = env::var_os("ARANGODB_HOST");
        let orig_user = env::var_os("ARANGODB_USER");
        let orig_pass = env::var_os("ARANGODB_PASSWORD");
        let orig_port = env::var_os("ARANGODB_PORT");
        let orig_db = env::var_os("ARANGODB_DATABASE");
        let orig_arango_host = env::var_os("ARANGO_HOST");
        let orig_arango_user = env::var_os("ARANGO_USER");
        let orig_arango_pass = env::var_os("ARANGO_PASSWORD");
        let orig_arango_port = env::var_os("ARANGO_PORT");
        let orig_arango_db = env::var_os("ARANGO_DATABASE");

        // Test missing host scenario - remove both variants
        env::remove_var("ARANGODB_HOST");
        env::remove_var("ARANGO_HOST");
        env::remove_var("ARANGODB_USER");
        env::remove_var("ARANGO_USER");
        env::remove_var("ARANGODB_PASSWORD");
        env::remove_var("ARANGO_PASSWORD");
        env::remove_var("ARANGODB_PORT");
        env::remove_var("ARANGO_PORT");
        env::remove_var("ARANGODB_DATABASE");
        env::remove_var("ARANGO_DATABASE");

        let err = config_from_env().unwrap_err();
        match err {
            GraphError::ConnectionFailed(msg) => assert!(msg.contains("Missing ARANGO_HOST")),
            _ => panic!("Expected ConnectionFailed error"),
        }

        env::set_var("ARANGODB_HOST", "localhost");
        env::set_var("ARANGODB_USER", "user1");
        env::set_var("ARANGODB_PASSWORD", "pass1");
        env::set_var("ARANGODB_PORT", "8529");
        // Don't set database - should remain None
        let cfg = config_from_env().unwrap();
        assert_eq!(cfg.hosts, vec!["localhost".to_string()]);
        assert_eq!(cfg.port, Some(8529));
        assert_eq!(cfg.username, Some("user1".to_string()));
        assert_eq!(cfg.password, Some("pass1".to_string()));
        assert!(cfg.database_name.is_none());

        // Restore original environment variables
        if let Some(val) = orig_host {
            env::set_var("ARANGODB_HOST", val);
        } else {
            env::remove_var("ARANGODB_HOST");
        }
        if let Some(val) = orig_user {
            env::set_var("ARANGODB_USER", val);
        } else {
            env::remove_var("ARANGODB_USER");
        }
        if let Some(val) = orig_pass {
            env::set_var("ARANGODB_PASSWORD", val);
        } else {
            env::remove_var("ARANGODB_PASSWORD");
        }
        if let Some(val) = orig_port {
            env::set_var("ARANGODB_PORT", val);
        } else {
            env::remove_var("ARANGODB_PORT");
        }
        if let Some(val) = orig_db {
            env::set_var("ARANGODB_DATABASE", val);
        } else {
            env::remove_var("ARANGODB_DATABASE");
        }

        // Restore ARANGO_* variants
        if let Some(val) = orig_arango_host {
            env::set_var("ARANGO_HOST", val);
        } else {
            env::remove_var("ARANGO_HOST");
        }
        if let Some(val) = orig_arango_user {
            env::set_var("ARANGO_USER", val);
        } else {
            env::remove_var("ARANGO_USER");
        }
        if let Some(val) = orig_arango_pass {
            env::set_var("ARANGO_PASSWORD", val);
        } else {
            env::remove_var("ARANGO_PASSWORD");
        }
        if let Some(val) = orig_arango_port {
            env::set_var("ARANGO_PORT", val);
        } else {
            env::remove_var("ARANGO_PORT");
        }
        if let Some(val) = orig_arango_db {
            env::set_var("ARANGO_DATABASE", val);
        } else {
            env::remove_var("ARANGO_DATABASE");
        }
    }
}
