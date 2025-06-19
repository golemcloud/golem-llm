use crate::conversions::from_gremlin_value;
use golem_graph::golem::graph::{
    connection::ConnectionConfig,
    errors::GraphError,
    types::{Edge, ElementId, Path, PropertyMap, Vertex},
};
use serde_json::Value;
use std::env;

pub(crate) fn config_from_env() -> Result<ConnectionConfig, GraphError> {
    dotenvy::dotenv().ok();
    let host = env::var("JANUSGRAPH_HOST")
        .map_err(|_| GraphError::ConnectionFailed("Missing JANUSGRAPH_HOST env var".to_string()))?;
    let port = env::var("JANUSGRAPH_PORT").map_or(Ok(None), |p| {
        p.parse::<u16>()
            .map(Some)
            .map_err(|e| GraphError::ConnectionFailed(format!("Invalid JANUSGRAPH_PORT: {}", e)))
    })?;
    let username = env::var("JANUSGRAPH_USER").ok();
    let password = env::var("JANUSGRAPH_PASSWORD").ok();

    Ok(ConnectionConfig {
        hosts: vec![host],
        port,
        database_name: None,
        username,
        password,
        timeout_seconds: None,
        max_connections: None,
        provider_config: vec![],
    })
}

pub(crate) fn parse_vertex_from_gremlin(value: &Value) -> Result<Vertex, GraphError> {
    let obj = value.as_object().ok_or_else(|| {
        GraphError::InternalError("Gremlin vertex value is not a JSON object".to_string())
    })?;

    let id =
        from_gremlin_id(obj.get("id").ok_or_else(|| {
            GraphError::InternalError("Missing 'id' in Gremlin vertex".to_string())
        })?)?;

    let label = obj
        .get("label")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let properties_val = obj.get("properties").ok_or_else(|| {
        GraphError::InternalError("Missing 'properties' in Gremlin vertex".to_string())
    })?;

    let properties = from_gremlin_properties(properties_val)?;

    Ok(Vertex {
        id,
        vertex_type: label,
        additional_labels: vec![],
        properties,
    })
}

fn from_gremlin_id(value: &Value) -> Result<ElementId, GraphError> {
    if let Some(id) = value.as_i64() {
        Ok(ElementId::Int64(id))
    } else if let Some(id) = value.as_str() {
        Ok(ElementId::StringValue(id.to_string()))
    } else if let Some(id_obj) = value.as_object() {
        if let Some(id_val) = id_obj.get("@value") {
            return from_gremlin_id(id_val);
        }
        Err(GraphError::InvalidPropertyType(
            "Unsupported element ID object from Gremlin".to_string(),
        ))
    } else {
        Err(GraphError::InvalidPropertyType(
            "Unsupported element ID type from Gremlin".to_string(),
        ))
    }
}

pub(crate) fn from_gremlin_properties(properties_value: &Value) -> Result<PropertyMap, GraphError> {
    let props_obj = properties_value.as_object().ok_or_else(|| {
        GraphError::InternalError("Gremlin properties value is not a JSON object".to_string())
    })?;

    let mut prop_map = Vec::new();
    for (key, value) in props_obj {
        let prop_value = if let Some(arr) = value.as_array() {
            arr.first().and_then(|p| p.get("value")).unwrap_or(value)
        } else if let Some(obj) = value.as_object() {
            if obj.contains_key("@type") && obj.contains_key("@value") {
                &obj["@value"]
            } else {
                value
            }
        } else {
            value
        };

        prop_map.push((key.clone(), from_gremlin_value(prop_value)?));
    }

    Ok(prop_map)
}

pub(crate) fn parse_edge_from_gremlin(value: &Value) -> Result<Edge, GraphError> {
    let obj = value.as_object().ok_or_else(|| {
        GraphError::InternalError("Gremlin edge value is not a JSON object".to_string())
    })?;

    let id =
        from_gremlin_id(obj.get("id").ok_or_else(|| {
            GraphError::InternalError("Missing 'id' in Gremlin edge".to_string())
        })?)?;

    let label = obj
        .get("label")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let in_v =
        from_gremlin_id(obj.get("inV").ok_or_else(|| {
            GraphError::InternalError("Missing 'inV' in Gremlin edge".to_string())
        })?)?;

    let out_v =
        from_gremlin_id(obj.get("outV").ok_or_else(|| {
            GraphError::InternalError("Missing 'outV' in Gremlin edge".to_string())
        })?)?;

    let properties = if let Some(properties_val) = obj.get("properties") {
        from_gremlin_properties(properties_val)?
    } else {
        vec![]
    };

    Ok(Edge {
        id,
        edge_type: label,
        from_vertex: out_v,
        to_vertex: in_v,
        properties,
    })
}

pub(crate) fn parse_path_from_gremlin(value: &Value) -> Result<Path, GraphError> {
    let path_array = value.as_array().ok_or_else(|| {
        GraphError::InternalError("Gremlin path value is not a JSON array".to_string())
    })?;

    let mut vertices = Vec::new();
    let mut edges = Vec::new();

    for element_value in path_array {
        let obj = element_value.as_object().ok_or_else(|| {
            GraphError::InternalError("Path element is not a JSON object".to_string())
        })?;

        if obj.contains_key("inV") && obj.contains_key("outV") {
            edges.push(parse_edge_from_gremlin(element_value)?);
        } else {
            vertices.push(parse_vertex_from_gremlin(element_value)?);
        }
    }

    Ok(Path {
        vertices,
        length: edges.len() as u32,
        edges,
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
    fn test_parse_vertex_from_gremlin() {
        let value = json!({
            "id": 1,
            "label": "Person",
            "properties": {
                "name": [{"id": "p1", "value": "Alice"}],
                "age": [{"id": "p2", "value": 30}]
            }
        });

        let vertex = parse_vertex_from_gremlin(&value).unwrap();
        assert_eq!(vertex.id, ElementId::Int64(1));
        assert_eq!(vertex.vertex_type, "Person");
        assert_eq!(vertex.additional_labels, Vec::<String>::new());
        assert_eq!(vertex.properties.len(), 2);
    }

    #[test]
    fn test_parse_edge_from_gremlin() {
        let value = json!({
            "id": "e123",
            "label": "KNOWS",
            "inV": 2,
            "outV": 1,
            "properties": {
                "since": {"@type": "g:Int64", "@value": 2020}
            }
        });

        let edge = parse_edge_from_gremlin(&value).unwrap();
        assert_eq!(edge.id, ElementId::StringValue("e123".to_string()));
        assert_eq!(edge.edge_type, "KNOWS");
        assert_eq!(edge.from_vertex, ElementId::Int64(1));
        assert_eq!(edge.to_vertex, ElementId::Int64(2));
        assert_eq!(edge.properties.len(), 1);
        assert_eq!(edge.properties[0].1, PropertyValue::Int64(2020));
    }

    #[test]
    fn test_parse_path_from_gremlin() {
        let path = json!([
            {
                "id": 1,
                "label": "Person",
                "properties": {
                    "name": [{"id": "p1", "value": "Alice"}]
                }
            },
            {
                "id": "e123",
                "label": "KNOWS",
                "inV": 2,
                "outV": 1,
                "properties": {
                    "since": {"@type": "g:Int64", "@value": 2020}
                }
            },
            {
                "id": 2,
                "label": "Person",
                "properties": {
                    "name": [{"id": "p2", "value": "Bob"}]
                }
            }
        ]);

        let path_obj = parse_path_from_gremlin(&path).unwrap();
        assert_eq!(path_obj.vertices.len(), 2);
        assert_eq!(path_obj.edges.len(), 1);
        assert_eq!(path_obj.length, 1);
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
