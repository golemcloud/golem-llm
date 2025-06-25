use crate::conversions::from_cypher_element_id;
use golem_graph::golem::graph::{
    connection::ConnectionConfig,
    errors::GraphError,
    schema::PropertyType,
    types::{Edge, ElementId, Path, Vertex},
};
use serde_json::Value;
use std::env;

pub(crate) fn parse_vertex_from_graph_data(
    node_val: &serde_json::Value,
    id_override: Option<ElementId>,
) -> Result<Vertex, GraphError> {
    let id = if let Some(id_val) = id_override {
        id_val
    } else {
        // Use elementId first (Neo4j 5.x), fallback to id (Neo4j 4.x)
        if let Some(element_id) = node_val.get("elementId") {
            from_cypher_element_id(element_id)?
        } else {
            from_cypher_element_id(&node_val["id"])?
        }
    };

    let labels: Vec<String> = node_val["labels"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default();

    let properties = if let Some(props) = node_val["properties"].as_object() {
        crate::conversions::from_cypher_properties(props.clone())?
    } else {
        vec![]
    };

    let (vertex_type, additional_labels) = labels
        .split_first()
        .map_or((String::new(), Vec::new()), |(first, rest)| {
            (first.clone(), rest.to_vec())
        });

    Ok(Vertex {
        id,
        vertex_type,
        additional_labels,
        properties,
    })
}

pub(crate) fn parse_edge_from_row(row: &[Value]) -> Result<Edge, GraphError> {
    if row.len() < 5 {
        return Err(GraphError::InternalError(
            "Invalid row data for edge".to_string(),
        ));
    }

    let id = from_cypher_element_id(&row[0])?;
    let edge_type = row[1]
        .as_str()
        .ok_or_else(|| GraphError::InternalError("Edge type is not a string".to_string()))?
        .to_string();

    let properties = if let Some(props) = row[2].as_object() {
        crate::conversions::from_cypher_properties(props.clone())?
    } else {
        vec![]
    };

    let from_vertex = from_cypher_element_id(&row[3])?;
    let to_vertex = from_cypher_element_id(&row[4])?;

    Ok(Edge {
        id,
        edge_type,
        from_vertex,
        to_vertex,
        properties,
    })
}

pub(crate) fn parse_path_from_data(data: &serde_json::Value) -> Result<Path, GraphError> {
    let nodes_val = data["graph"]["nodes"]
        .as_array()
        .ok_or_else(|| GraphError::InternalError("Missing nodes in path response".to_string()))?;
    let rels_val = data["graph"]["relationships"].as_array().ok_or_else(|| {
        GraphError::InternalError("Missing relationships in path response".to_string())
    })?;

    let mut vertices = Vec::new();
    for node_val in nodes_val {
        vertices.push(parse_vertex_from_graph_data(node_val, None)?);
    }

    let mut edges = Vec::new();
    for rel_val in rels_val {
        let id = from_cypher_element_id(&rel_val["id"])?;
        let edge_type = rel_val["type"].as_str().unwrap_or_default().to_string();
        let properties = if let Some(props) = rel_val["properties"].as_object() {
            crate::conversions::from_cypher_properties(props.clone())?
        } else {
            vec![]
        };
        let from_vertex = from_cypher_element_id(&rel_val["startNode"])?;
        let to_vertex = from_cypher_element_id(&rel_val["endNode"])?;
        edges.push(Edge {
            id,
            edge_type,
            from_vertex,
            to_vertex,
            properties,
        });
    }

    Ok(Path {
        vertices,
        edges,
        length: rels_val.len() as u32,
    })
}

pub(crate) fn map_neo4j_type_to_wit(neo4j_type: &str) -> PropertyType {
    match neo4j_type {
        "String" => PropertyType::StringType,
        "Integer" => PropertyType::Int64,
        "Float" => PropertyType::Float64Type,
        "Boolean" => PropertyType::Boolean,
        "Date" => PropertyType::Date,
        "DateTime" => PropertyType::Datetime,
        "Point" => PropertyType::Point,
        "ByteArray" => PropertyType::Bytes,
        _ => PropertyType::StringType, // Default for mixed or unknown types
    }
}

pub(crate) fn config_from_env() -> Result<ConnectionConfig, GraphError> {
    let host = env::var("NEO4J_HOST")
        .map_err(|_| GraphError::ConnectionFailed("Missing NEO4J_HOST env var".to_string()))?;
    let port = env::var("NEO4J_PORT").map_or(Ok(None), |p| {
        p.parse::<u16>()
            .map(Some)
            .map_err(|e| GraphError::ConnectionFailed(format!("Invalid NEO4J_PORT: {}", e)))
    })?;
    let username = env::var("NEO4J_USER")
        .map_err(|_| GraphError::ConnectionFailed("Missing NEO4J_USER env var".to_string()))?;
    let password = env::var("NEO4J_PASSWORD")
        .map_err(|_| GraphError::ConnectionFailed("Missing NEO4J_PASSWORD env var".to_string()))?;

    Ok(ConnectionConfig {
        hosts: vec![host],
        port,
        database_name: None,
        username: Some(username),
        password: Some(password),
        timeout_seconds: None,
        max_connections: None,
        provider_config: vec![],
    })
}

pub(crate) fn element_id_to_key(id: &ElementId) -> String {
    match id {
        ElementId::StringValue(s) => format!("s:{}", s),
        ElementId::Int64(i) => format!("i:{}", i),
        ElementId::Uuid(u) => format!("u:{}", u),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use golem_graph::golem::graph::types::PropertyValue;
    use serde_json::json;

    #[test]
    fn test_parse_vertex() {
        let node_val = json!({
            "id": "123",
            "labels": ["User", "Person"],
            "properties": {
                "name": "Alice",
                "age": 30
            }
        });

        let vertex = parse_vertex_from_graph_data(&node_val, None).unwrap();
        assert_eq!(vertex.id, ElementId::StringValue("123".to_string()));
        assert_eq!(vertex.vertex_type, "User");
        assert_eq!(vertex.additional_labels, vec!["Person"]);
        assert_eq!(vertex.properties.len(), 2);
    }

    #[test]
    fn test_parse_edge_from_row() {
        let row_val = vec![
            json!("456"),
            json!("KNOWS"),
            json!({"since": 2020}),
            json!("123"),
            json!("789"),
        ];

        let edge = parse_edge_from_row(&row_val).unwrap();
        assert_eq!(edge.id, ElementId::StringValue("456".to_string()));
        assert_eq!(edge.edge_type, "KNOWS");
        assert_eq!(edge.properties.len(), 1);
        assert_eq!(edge.properties[0].1, PropertyValue::Int64(2020));
        assert_eq!(edge.from_vertex, ElementId::StringValue("123".to_string()));
        assert_eq!(edge.to_vertex, ElementId::StringValue("789".to_string()));
    }

    #[test]
    fn test_map_neo4j_type_to_wit() {
        assert_eq!(map_neo4j_type_to_wit("String"), PropertyType::StringType);
        assert_eq!(map_neo4j_type_to_wit("Integer"), PropertyType::Int64);
        assert_eq!(map_neo4j_type_to_wit("Float"), PropertyType::Float64Type);
        assert_eq!(map_neo4j_type_to_wit("Boolean"), PropertyType::Boolean);
        assert_eq!(map_neo4j_type_to_wit("Date"), PropertyType::Date);
        assert_eq!(map_neo4j_type_to_wit("DateTime"), PropertyType::Datetime);
        assert_eq!(map_neo4j_type_to_wit("Point"), PropertyType::Point);
        assert_eq!(map_neo4j_type_to_wit("ByteArray"), PropertyType::Bytes);
        assert_eq!(
            map_neo4j_type_to_wit("UnknownType"),
            PropertyType::StringType
        );
    }

    #[test]
    fn test_element_id_to_key() {
        assert_eq!(
            element_id_to_key(&ElementId::StringValue("abc".to_string())),
            "s:abc"
        );
        assert_eq!(element_id_to_key(&ElementId::Int64(123)), "i:123");
        let uuid = "a1a2a3a4-b1b2-c1c2-d1d2-d3d4d5d6d7d8";
        assert_eq!(
            element_id_to_key(&ElementId::Uuid(uuid.to_string())),
            format!("u:{}", uuid)
        );
    }
}
