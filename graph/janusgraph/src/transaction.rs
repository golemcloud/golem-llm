use crate::conversions;
use crate::helpers;
use crate::query_utils;
use crate::Transaction;
use golem_graph::golem::graph::{
    errors::GraphError,
    transactions::{EdgeSpec, GuestTransaction, VertexSpec},
    types::{Direction, Edge, ElementId, FilterCondition, PropertyMap, SortSpec, Vertex},
};
use serde_json::{json, Value};

/// Given a GraphSON Map element, turn it into a serde_json::Value::Object
fn graphson_map_to_object(data: &Value) -> Result<Value, GraphError> {
    let arr = data
        .get("@value")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            GraphError::InternalError("Expected GraphSON Map with @value array".into())
        })?;

    let mut obj = serde_json::Map::new();
    let mut iter = arr.iter();
    while let (Some(k), Some(v)) = (iter.next(), iter.next()) {
        let key = if let Some(s) = k.as_str() {
            s.to_string()
        } else if let Some(inner) = k.get("@value").and_then(Value::as_str) {
            inner.to_string()
        } else {
            return Err(GraphError::InternalError(format!(
                "Expected string key in GraphSON Map, got {}",
                k
            )));
        };

        let val = if let Some(inner) = v.get("@value") {
            inner.clone()
        } else {
            v.clone()
        };

        obj.insert(key, val);
    }

    Ok(Value::Object(obj))
}

fn unwrap_list(data: &Value) -> Result<&Vec<Value>, GraphError> {
    data.get("@value")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            GraphError::InternalError("Expected `@value: List` in Gremlin response".into())
        })
}
fn first_list_item(data: &Value) -> Result<&Value, GraphError> {
    unwrap_list(data)?
        .first()
        .ok_or_else(|| GraphError::InternalError("Empty result list from Gremlin".into()))
}

impl GuestTransaction for Transaction {
    fn commit(&self) -> Result<(), GraphError> {
        Ok(())
    }

    fn rollback(&self) -> Result<(), GraphError> {
        Ok(())
    }

    fn create_vertex(
        &self,
        vertex_type: String,
        properties: PropertyMap,
    ) -> Result<Vertex, GraphError> {
        self.create_vertex_with_labels(vertex_type, vec![], properties)
    }

    fn create_vertex_with_labels(
        &self,
        vertex_type: String,
        _additional_labels: Vec<String>,
        properties: PropertyMap,
    ) -> Result<Vertex, GraphError> {
        let mut gremlin = "g.addV(vertex_label)".to_string();
        let mut bindings = serde_json::Map::new();
        bindings.insert("vertex_label".to_string(), json!(vertex_type));

        for (i, (key, value)) in properties.into_iter().enumerate() {
            let binding_key = format!("p{}", i);
            gremlin.push_str(&format!(".property(k{}, {})", i, binding_key));
            bindings.insert(format!("k{}", i), json!(key));
            bindings.insert(binding_key, conversions::to_json_value(value)?);
        }
        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;
        eprintln!(
            "[JanusGraphApi] Raw vertex creation response: {:?}",
            response
        );
        let element = first_list_item(&response["result"]["data"])?;
        let obj = graphson_map_to_object(element)?;

        helpers::parse_vertex_from_gremlin(&obj)
    }

    fn get_vertex(&self, id: ElementId) -> Result<Option<Vertex>, GraphError> {
        let gremlin = "g.V(vertex_id).elementMap()".to_string();

        let mut bindings = serde_json::Map::new();
        bindings.insert(
            "vertex_id".to_string(),
            match id.clone() {
                ElementId::StringValue(s) => json!(s),
                ElementId::Int64(i) => json!(i),
                ElementId::Uuid(u) => json!(u.to_string()),
            },
        );

        let resp = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let data = &resp["result"]["data"];
        let list: Vec<Value> = if let Some(arr) = data.as_array() {
            arr.clone()
        } else if let Some(inner) = data.get("@value").and_then(Value::as_array) {
            inner.clone()
        } else {
            vec![]
        };

        if let Some(row) = list.into_iter().next() {
            let obj = if row.get("@type") == Some(&json!("g:Map")) {
                let vals = row.get("@value").and_then(Value::as_array).unwrap();
                let mut m = serde_json::Map::new();
                let mut it = vals.iter();
                while let (Some(kv), Some(vv)) = (it.next(), it.next()) {
                    let key = if kv.is_string() {
                        kv.as_str().unwrap().to_string()
                    } else {
                        kv.get("@value")
                            .and_then(Value::as_str)
                            .unwrap()
                            .to_string()
                    };
                    let val = if vv.is_object() {
                        vv.get("@value").cloned().unwrap_or(vv.clone())
                    } else {
                        vv.clone()
                    };
                    m.insert(key, val);
                }
                Value::Object(m)
            } else {
                row.clone()
            };

            let vertex = helpers::parse_vertex_from_gremlin(&obj)?;
            Ok(Some(vertex))
        } else {
            Ok(None)
        }
    }

    fn update_vertex(&self, id: ElementId, properties: PropertyMap) -> Result<Vertex, GraphError> {
        let mut gremlin = "g.V(vertex_id).sideEffect(properties().drop())".to_string();
        let mut bindings = serde_json::Map::new();
        bindings.insert(
            "vertex_id".to_string(),
            match id.clone() {
                ElementId::StringValue(s) => json!(s),
                ElementId::Int64(i) => json!(i),
                ElementId::Uuid(u) => json!(u.to_string()),
            },
        );

        for (i, (k, v)) in properties.into_iter().enumerate() {
            let kb = format!("k{}", i);
            let vb = format!("v{}", i);
            gremlin.push_str(&format!(".property({}, {})", kb, vb));
            bindings.insert(kb.clone(), json!(k));
            bindings.insert(vb.clone(), conversions::to_json_value(v)?);
        }

        gremlin.push_str(".elementMap()");

        let resp = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let data = &resp["result"]["data"];
        let maybe_row = data
            .as_array()
            .and_then(|arr| arr.first().cloned())
            .or_else(|| {
                data.get("@value")
                    .and_then(Value::as_array)
                    .and_then(|arr| arr.first().cloned())
            });
        let row = maybe_row.ok_or(GraphError::ElementNotFound(id.clone()))?;

        let mut flat = serde_json::Map::new();
        if row.get("@type") == Some(&json!("g:Map")) {
            let vals = row.get("@value").and_then(Value::as_array).unwrap();
            let mut it = vals.iter();
            while let (Some(kv), Some(vv)) = (it.next(), it.next()) {
                // key: plain string or wrapped
                let key = if kv.is_string() {
                    kv.as_str().unwrap().to_string()
                } else {
                    kv.get("@value")
                        .and_then(Value::as_str)
                        .unwrap()
                        .to_string()
                };
                let val = if vv.is_object() {
                    vv.get("@value").cloned().unwrap_or(vv.clone())
                } else {
                    vv.clone()
                };
                flat.insert(key, val);
            }
        } else if let Some(obj) = row.as_object() {
            flat = obj.clone();
        } else {
            return Err(GraphError::InternalError(
                "Unexpected Gremlin row format".into(),
            ));
        }

        let mut obj = serde_json::Map::new();
        obj.insert("id".to_string(), flat["id"].clone());
        obj.insert("label".to_string(), flat["label"].clone());

        let mut props = serde_json::Map::new();
        for (k, v) in flat.into_iter() {
            if k != "id" && k != "label" {
                props.insert(k, v);
            }
        }
        obj.insert("properties".to_string(), Value::Object(props));

        helpers::parse_vertex_from_gremlin(&Value::Object(obj))
    }

    fn update_vertex_properties(
        &self,
        id: ElementId,
        updates: PropertyMap,
    ) -> Result<Vertex, GraphError> {
        if updates.is_empty() {
            return self
                .get_vertex(id.clone())?
                .ok_or(GraphError::ElementNotFound(id));
        }

        let mut gremlin = "g.V(vertex_id)".to_string();
        let mut bindings = serde_json::Map::new();
        let id_clone = id.clone();
        let id_json = match id.clone() {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        bindings.insert("vertex_id".to_string(), id_json);

        for (i, (k, v)) in updates.into_iter().enumerate() {
            let kb = format!("k{}", i);
            let vb = format!("v{}", i);
            gremlin.push_str(&format!(".property({}, {})", kb, vb));
            bindings.insert(kb, json!(k));
            bindings.insert(vb, conversions::to_json_value(v)?);
        }

        gremlin.push_str(".elementMap()");

        let resp = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;
        let data = &resp["result"]["data"];

        let row = if let Some(arr) = data.as_array() {
            arr.first()
        } else if let Some(inner) = data.get("@value").and_then(Value::as_array) {
            inner.first()
        } else {
            None
        }
        .ok_or_else(|| GraphError::ElementNotFound(id_clone.clone()))?;

        println!("[DEBUG update_vertex] raw row = {:#}", row);

        let mut flat = serde_json::Map::new();
        if row.get("@type") == Some(&json!("g:Map")) {
            let vals = row.get("@value").and_then(Value::as_array).unwrap(); // we know it's an array
            let mut it = vals.iter();
            while let (Some(kv), Some(vv)) = (it.next(), it.next()) {
                // key:
                let key = if let Some(s) = kv.as_str() {
                    s.to_string()
                } else if kv.get("@type") == Some(&json!("g:T")) {
                    kv.get("@value")
                        .and_then(Value::as_str)
                        .unwrap()
                        .to_string()
                } else {
                    return Err(GraphError::InternalError(
                        "Unexpected key format in Gremlin map".into(),
                    ));
                };
                let val = if let Some(obj) = vv.as_object() {
                    obj.get("@value")
                        .cloned()
                        .unwrap_or(Value::Object(obj.clone()))
                } else {
                    vv.clone()
                };
                flat.insert(key, val);
            }
        } else if let Some(obj) = row.as_object() {
            flat = obj.clone();
        } else {
            return Err(GraphError::InternalError(
                "Unexpected Gremlin row format".into(),
            ));
        }

        let mut vertex_json = serde_json::Map::new();
        vertex_json.insert("id".to_string(), flat["id"].clone());
        vertex_json.insert("label".to_string(), flat["label"].clone());

        let mut props = serde_json::Map::new();
        for (k, v) in flat.into_iter() {
            if k == "id" || k == "label" {
                continue;
            }
            props.insert(k, v);
        }
        vertex_json.insert("properties".to_string(), Value::Object(props));

        println!(
            "[DEBUG update_vertex] parser input = {:#}",
            Value::Object(vertex_json.clone())
        );

        helpers::parse_vertex_from_gremlin(&Value::Object(vertex_json))
    }

    fn delete_vertex(&self, id: ElementId, _detach: bool) -> Result<(), GraphError> {
        // Note: JanusGraph handles edge cleanup automatically during vertex deletion
        let gremlin = "g.V(vertex_id).drop().toList()";
        let mut bindings = serde_json::Map::new();
        bindings.insert(
            "vertex_id".to_string(),
            match id.clone() {
                ElementId::StringValue(s) => json!(s),
                ElementId::Int64(i) => json!(i),
                ElementId::Uuid(u) => json!(u.to_string()),
            },
        );

        for attempt in 1..=2 {
            let resp = self
                .api
                .execute(gremlin, Some(Value::Object(bindings.clone())));
            match resp {
                Ok(_) => {
                    log::info!(
                        "[delete_vertex] dropped vertex {:?} (attempt {})",
                        id,
                        attempt
                    );
                    return Ok(());
                }
                Err(GraphError::InvalidQuery(msg))
                    if msg.contains("Lock expired") && attempt == 1 =>
                {
                    log::warn!(
                        "[delete_vertex] Lock expired on vertex {:?}, retrying drop (1/2)",
                        id
                    );
                    continue;
                }
                Err(GraphError::InvalidQuery(msg)) if msg.contains("Lock expired") => {
                    log::warn!(
                        "[delete_vertex] Lock expired again on {:?}, ignoring cleanup",
                        id
                    );
                    return Ok(());
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    fn find_vertices(
        &self,
        vertex_type: Option<String>,
        filters: Option<Vec<FilterCondition>>,
        sort: Option<Vec<SortSpec>>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<Vertex>, GraphError> {
        let mut gremlin = "g.V()".to_string();
        let mut bindings = serde_json::Map::new();

        if let Some(label) = vertex_type {
            gremlin.push_str(".hasLabel(vertex_label)");
            bindings.insert("vertex_label".to_string(), json!(label));
        }

        if let Some(filter_conditions) = filters {
            for condition in &filter_conditions {
                gremlin.push_str(&query_utils::build_gremlin_filter_step(
                    condition,
                    &mut bindings,
                )?);
            }
        }

        if let Some(sort_specs) = sort {
            gremlin.push_str(&query_utils::build_gremlin_sort_clause(&sort_specs));
        }

        if let Some(off) = offset {
            gremlin.push_str(&format!(
                ".range({}, {})",
                off,
                off + limit.unwrap_or(10_000)
            ));
        } else if let Some(lim) = limit {
            gremlin.push_str(&format!(".limit({})", lim));
        }

        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;
        println!(
            "[DEBUG][find_vertices] Raw Gremlin response: {:?}",
            response
        );

        // Handle GraphSON g:List structure
        let data = &response["result"]["data"];
        let result_data = if let Some(arr) = data.as_array() {
            arr.clone()
        } else if let Some(inner) = data.get("@value").and_then(Value::as_array) {
            inner.clone()
        } else {
            return Err(GraphError::InternalError(
                "Invalid response from Gremlin for find_vertices".to_string(),
            ));
        };

        result_data
            .iter()
            .map(|item| {
                let result = helpers::parse_vertex_from_gremlin(item);
                if let Err(ref e) = result {
                    println!(
                        "[DEBUG][find_vertices] Parse error for item {:?}: {:?}",
                        item, e
                    );
                }
                result
            })
            .collect()
    }

    fn create_edge(
        &self,
        edge_type: String,
        from_vertex: ElementId,
        to_vertex: ElementId,
        properties: PropertyMap,
    ) -> Result<Edge, GraphError> {
        let mut gremlin = "g.V(from_id).addE(edge_label).to(__.V(to_id))".to_string();
        let mut bindings = serde_json::Map::new();
        let from_clone = from_vertex.clone();

        bindings.insert(
            "from_id".into(),
            match from_vertex {
                ElementId::StringValue(s) => json!(s),
                ElementId::Int64(i) => json!(i),
                ElementId::Uuid(u) => json!(u.to_string()),
            },
        );
        bindings.insert(
            "to_id".into(),
            match to_vertex {
                ElementId::StringValue(s) => json!(s),
                ElementId::Int64(i) => json!(i),
                ElementId::Uuid(u) => json!(u.to_string()),
            },
        );
        bindings.insert("edge_label".into(), json!(edge_type));

        for (i, (k, v)) in properties.into_iter().enumerate() {
            let kb = format!("k{}", i);
            let vb = format!("v{}", i);
            gremlin.push_str(&format!(".property({}, {})", kb, vb));
            bindings.insert(kb.clone(), json!(k));
            bindings.insert(vb.clone(), conversions::to_json_value(v)?);
            println!("[LOG create_edge] bound {} -> {:?}", kb, bindings[&kb]);
        }

        gremlin.push_str(".elementMap()");

        let resp = self
            .api
            .execute(&gremlin, Some(Value::Object(bindings.clone())))?;
        let data = &resp["result"]["data"];

        let row = if let Some(arr) = data.as_array() {
            arr.first().cloned()
        } else if let Some(inner) = data.get("@value").and_then(Value::as_array) {
            inner.first().cloned()
        } else {
            println!("[ERROR create_edge] no data row");
            None
        }
        .ok_or_else(|| GraphError::ElementNotFound(from_clone.clone()))?;

        let mut flat = serde_json::Map::new();
        if row.get("@type") == Some(&json!("g:Map")) {
            let vals = row.get("@value").and_then(Value::as_array).unwrap();
            let mut it = vals.iter();
            while let (Some(kv), Some(vv)) = (it.next(), it.next()) {
                let key = if kv.is_string() {
                    kv.as_str().unwrap().to_string()
                } else {
                    kv.get("@value")
                        .and_then(Value::as_str)
                        .unwrap()
                        .to_string()
                };
                let val = if vv.is_object() {
                    vv.get("@value").cloned().unwrap_or(vv.clone())
                } else {
                    vv.clone()
                };
                flat.insert(key.clone(), val.clone());
            }
        } else if let Some(obj) = row.as_object() {
            flat = obj.clone();
        } else {
            println!("[ERROR create_edge] unexpected row format: {:#?}", row);
            return Err(GraphError::InternalError("Unexpected row format".into()));
        }

        let mut edge_json = serde_json::Map::new();

        let id_field = &flat["id"];
        let real_id = if let Some(rel) = id_field.get("relationId").and_then(Value::as_str) {
            json!(rel)
        } else {
            id_field.clone()
        };
        edge_json.insert("id".into(), real_id.clone());

        let lbl = flat["label"].clone();
        edge_json.insert("label".into(), lbl.clone());

        if let Some(arr) = flat.get("OUT").and_then(Value::as_array) {
            if let Some(vv) = arr.get(1).and_then(|v| v.get("@value")).cloned() {
                edge_json.insert("outV".into(), vv.clone());
            }
        }
        if let Some(arr) = flat.get("IN").and_then(Value::as_array) {
            if let Some(vv) = arr.get(1).and_then(|v| v.get("@value")).cloned() {
                edge_json.insert("inV".into(), vv.clone());
            }
        }

        edge_json.insert("properties".into(), json!({}));

        helpers::parse_edge_from_gremlin(&Value::Object(edge_json))
    }

    fn get_edge(&self, id: ElementId) -> Result<Option<Edge>, GraphError> {
        let gremlin = "g.E(edge_id).elementMap()".to_string();
        let mut bindings = serde_json::Map::new();
        bindings.insert(
            "edge_id".into(),
            match id.clone() {
                ElementId::StringValue(s) => json!(s),
                ElementId::Int64(i) => json!(i),
                ElementId::Uuid(u) => json!(u.to_string()),
            },
        );

        let resp = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let data = &resp["result"]["data"];
        let maybe_row = data
            .as_array()
            .and_then(|arr| arr.first().cloned())
            .or_else(|| {
                data.get("@value")
                    .and_then(Value::as_array)
                    .and_then(|arr| arr.first().cloned())
            });
        let row = if let Some(r) = maybe_row {
            r
        } else {
            return Ok(None);
        };
        println!("[LOG get_edge] unwrapped row = {:#?}", row);

        let mut flat = serde_json::Map::new();
        if row.get("@type") == Some(&json!("g:Map")) {
            let vals = row.get("@value").and_then(Value::as_array).unwrap();
            let mut it = vals.iter();
            while let (Some(kv), Some(vv)) = (it.next(), it.next()) {
                let key = if kv.is_string() {
                    kv.as_str().unwrap().to_string()
                } else if kv.get("@type") == Some(&json!("g:T"))
                    || kv.get("@type") == Some(&json!("g:Direction"))
                {
                    kv.get("@value")
                        .and_then(Value::as_str)
                        .unwrap()
                        .to_string()
                } else {
                    return Err(GraphError::InternalError(
                        "Unexpected key format in Gremlin map".into(),
                    ));
                };

                let val = if vv.is_object() {
                    if vv.get("@type") == Some(&json!("g:Map")) {
                        vv.get("@value").cloned().unwrap()
                    } else {
                        vv.get("@value").cloned().unwrap_or(vv.clone())
                    }
                } else {
                    vv.clone()
                };
                flat.insert(key.clone(), val.clone());
            }
        } else if let Some(obj) = row.as_object() {
            flat = obj.clone();
        } else {
            return Err(GraphError::InternalError(
                "Unexpected Gremlin row format".into(),
            ));
        }

        let mut edge_json = serde_json::Map::new();

        let id_field = &flat["id"];
        let real_id = id_field
            .get("relationId")
            .and_then(Value::as_str)
            .map(|s| json!(s))
            .unwrap_or_else(|| id_field.clone());
        edge_json.insert("id".into(), real_id.clone());

        let lbl = flat["label"].clone();
        edge_json.insert("label".into(), lbl.clone());

        if let Some(arr) = flat.get("OUT").and_then(Value::as_array) {
            let ov = arr[1].get("@value").cloned().unwrap();
            edge_json.insert("outV".into(), ov.clone());
        }
        if let Some(arr) = flat.get("IN").and_then(Value::as_array) {
            let iv = arr[1].get("@value").cloned().unwrap();
            edge_json.insert("inV".into(), iv.clone());
        }

        let mut props = serde_json::Map::new();
        for (k, v) in flat.into_iter() {
            if k != "id" && k != "label" && k != "IN" && k != "OUT" {
                props.insert(k.clone(), v.clone());
            }
        }
        edge_json.insert("properties".into(), Value::Object(props.clone()));

        let edge = helpers::parse_edge_from_gremlin(&Value::Object(edge_json))?;
        Ok(Some(edge))
    }

    fn update_edge(&self, id: ElementId, properties: PropertyMap) -> Result<Edge, GraphError> {
        let id_json = match &id {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };

        // 2) STEP 1: Drop all props & set the new ones
        let mut gremlin_update = "g.E(edge_id).sideEffect(properties().drop())".to_string();
        let mut bindings = serde_json::Map::new();
        bindings.insert("edge_id".to_string(), id_json.clone());

        for (i, (k, v)) in properties.iter().enumerate() {
            let kb = format!("k{}", i);
            let vb = format!("v{}", i);
            gremlin_update.push_str(&format!(".sideEffect(property({}, {}))", kb, vb));
            bindings.insert(kb.clone(), json!(k));
            bindings.insert(vb.clone(), conversions::to_json_value(v.clone())?);
        }

        self.api
            .execute(&gremlin_update, Some(Value::Object(bindings)))?;

        let gremlin_fetch = "g.E(edge_id).elementMap()";
        let fetch_bindings = json!({ "edge_id": id_json });

        let resp = self.api.execute(gremlin_fetch, Some(fetch_bindings))?;

        let data = &resp["result"]["data"];
        let row = data
            .as_array()
            .and_then(|arr| arr.first().cloned())
            .or_else(|| {
                data.get("@value")
                    .and_then(Value::as_array)
                    .and_then(|a| a.first().cloned())
            })
            .ok_or_else(|| GraphError::ElementNotFound(id.clone()))?;

        let mut flat = serde_json::Map::new();
        if row.get("@type") == Some(&json!("g:Map")) {
            let vals = row.get("@value").and_then(Value::as_array).unwrap();
            let mut it = vals.iter();
            while let (Some(kv), Some(vv)) = (it.next(), it.next()) {
                let key = if kv.is_string() {
                    kv.as_str().unwrap().to_string()
                } else {
                    kv.get("@value")
                        .and_then(Value::as_str)
                        .unwrap()
                        .to_string()
                };
                let val = if vv.is_object() {
                    vv.get("@value").cloned().unwrap_or(vv.clone())
                } else {
                    vv.clone()
                };
                flat.insert(key.clone(), val.clone());
                log::info!("[update_edge] flat[{}] = {:#?}", key, val);
            }
        } else if let Some(obj) = row.as_object() {
            flat = obj.clone();
        } else {
            return Err(GraphError::InternalError("Unexpected row format".into()));
        }

        let mut ej = serde_json::Map::new();

        let id_field = &flat["id"];
        let real_id = id_field
            .get("relationId")
            .and_then(Value::as_str)
            .map(|s| json!(s))
            .unwrap_or_else(|| id_field.clone());
        ej.insert("id".into(), real_id.clone());

        ej.insert("label".into(), flat["label"].clone());

        if let Some(arr) = flat.get("OUT").and_then(Value::as_array) {
            let ov = arr[1].get("@value").cloned().unwrap();
            ej.insert("outV".into(), ov.clone());
        }
        if let Some(arr) = flat.get("IN").and_then(Value::as_array) {
            let iv = arr[1].get("@value").cloned().unwrap();
            ej.insert("inV".into(), iv.clone());
        }

        let mut props = serde_json::Map::new();
        for (k, v) in flat.into_iter() {
            if k != "id" && k != "label" && k != "IN" && k != "OUT" {
                props.insert(k.clone(), v.clone());
            }
        }
        ej.insert("properties".into(), Value::Object(props.clone()));

        let edge = helpers::parse_edge_from_gremlin(&Value::Object(ej))?;
        Ok(edge)
    }

    fn update_edge_properties(
        &self,
        id: ElementId,
        updates: PropertyMap,
    ) -> Result<Edge, GraphError> {
        if updates.is_empty() {
            return self
                .get_edge(id.clone())?
                .ok_or(GraphError::ElementNotFound(id));
        }

        let mut gremlin = "g.E(edge_id)".to_string();
        let mut bindings = serde_json::Map::new();
        let id_clone = id.clone();
        let id_json = match id.clone() {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        bindings.insert("edge_id".into(), id_json);

        for (i, (k, v)) in updates.into_iter().enumerate() {
            let kb = format!("k{}", i);
            let vb = format!("v{}", i);
            gremlin.push_str(&format!(".property({}, {})", kb, vb));
            bindings.insert(kb.clone(), json!(k));
            bindings.insert(vb.clone(), conversions::to_json_value(v)?);
        }

        gremlin.push_str(".elementMap()");

        let resp = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let data = &resp["result"]["data"];
        let row = if let Some(arr) = data.as_array() {
            arr.first().cloned()
        } else if let Some(inner) = data.get("@value").and_then(Value::as_array) {
            inner.first().cloned()
        } else {
            return Err(GraphError::ElementNotFound(id_clone.clone()));
        }
        .unwrap();

        let mut flat = serde_json::Map::new();
        if row.get("@type") == Some(&json!("g:Map")) {
            let vals = row.get("@value").and_then(Value::as_array).unwrap();
            let mut it = vals.iter();
            while let (Some(kv), Some(vv)) = (it.next(), it.next()) {
                let key = if kv.is_string() {
                    kv.as_str().unwrap().to_string()
                } else if kv.get("@type") == Some(&json!("g:T"))
                    || kv.get("@type") == Some(&json!("g:Direction"))
                {
                    kv.get("@value")
                        .and_then(Value::as_str)
                        .unwrap()
                        .to_string()
                } else {
                    return Err(GraphError::InternalError(
                        "Unexpected key format in Gremlin map".into(),
                    ));
                };

                let val = if vv.is_object() {
                    if vv.get("@type") == Some(&json!("g:Map")) {
                        vv.get("@value").cloned().unwrap()
                    } else {
                        vv.get("@value").cloned().unwrap_or(vv.clone())
                    }
                } else {
                    vv.clone()
                };

                flat.insert(key.clone(), val.clone());
                println!("[LOG update_edge] flat[{}] = {:#?}", key, val);
            }
        } else if let Some(obj) = row.as_object() {
            flat = obj.clone();
            println!("[LOG update_edge] row is plain object");
        } else {
            return Err(GraphError::InternalError(
                "Unexpected Gremlin row format".into(),
            ));
        }

        let mut edge_json = serde_json::Map::new();

        let id_field = &flat["id"];
        let real_id = id_field
            .get("relationId")
            .and_then(Value::as_str)
            .map(|s| json!(s))
            .unwrap_or_else(|| id_field.clone());
        edge_json.insert("id".into(), real_id.clone());

        let lbl = flat["label"].clone();
        edge_json.insert("label".into(), lbl.clone());

        if let Some(arr) = flat.get("OUT").and_then(Value::as_array) {
            edge_json.insert("outV".into(), json!(arr[1].get("@value").unwrap()));
        }
        if let Some(arr) = flat.get("IN").and_then(Value::as_array) {
            edge_json.insert("inV".into(), json!(arr[1].get("@value").unwrap()));
        }

        let mut props = serde_json::Map::new();
        for (k, v) in flat.into_iter() {
            if k != "id" && k != "label" && k != "IN" && k != "OUT" {
                props.insert(k.clone(), v.clone());
            }
        }
        edge_json.insert("properties".into(), Value::Object(props.clone()));

        helpers::parse_edge_from_gremlin(&Value::Object(edge_json))
    }

    fn delete_edge(&self, id: ElementId) -> Result<(), GraphError> {
        let gremlin = "g.E(edge_id).drop().toList()".to_string();

        let id_json = match id {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        let mut bindings = serde_json::Map::new();
        bindings.insert("edge_id".to_string(), id_json);

        self.api.execute(&gremlin, Some(Value::Object(bindings)))?;
        Ok(())
    }

    fn find_edges(
        &self,
        edge_types: Option<Vec<String>>,
        filters: Option<Vec<FilterCondition>>,
        sort: Option<Vec<SortSpec>>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<Edge>, GraphError> {
        let mut gremlin = "g.E()".to_string();
        let mut bindings = serde_json::Map::new();

        if let Some(labels) = edge_types {
            if !labels.is_empty() {
                gremlin.push_str(".hasLabel(edge_labels)");
                bindings.insert("edge_labels".to_string(), json!(labels));
            }
        }

        if let Some(filter_conditions) = filters {
            for condition in &filter_conditions {
                gremlin.push_str(&query_utils::build_gremlin_filter_step(
                    condition,
                    &mut bindings,
                )?);
            }
        }

        if let Some(sort_specs) = sort {
            gremlin.push_str(&query_utils::build_gremlin_sort_clause(&sort_specs));
        }

        if let Some(off) = offset {
            gremlin.push_str(&format!(
                ".range({}, {})",
                off,
                off + limit.unwrap_or(10_000)
            ));
        } else if let Some(lim) = limit {
            gremlin.push_str(&format!(".limit({})", lim));
        }

        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"].as_array().ok_or_else(|| {
            GraphError::InternalError("Invalid response from Gremlin for find_edges".to_string())
        })?;

        result_data
            .iter()
            .map(helpers::parse_edge_from_gremlin)
            .collect()
    }

    fn get_adjacent_vertices(
        &self,
        vertex_id: ElementId,
        direction: Direction,
        edge_types: Option<Vec<String>>,
        limit: Option<u32>,
    ) -> Result<Vec<Vertex>, GraphError> {
        let mut bindings = serde_json::Map::new();
        let id_json = match vertex_id {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        bindings.insert("vertex_id".to_string(), id_json);

        let direction_step = match direction {
            Direction::Outgoing => "out",
            Direction::Incoming => "in",
            Direction::Both => "both",
        };

        let mut gremlin = if let Some(labels) = edge_types {
            if !labels.is_empty() {
                let label_bindings: Vec<String> = labels
                    .iter()
                    .enumerate()
                    .map(|(i, label)| {
                        let binding_key = format!("label_{}", i);
                        bindings.insert(binding_key.clone(), json!(label));
                        binding_key
                    })
                    .collect();
                let labels_str = label_bindings.join(", ");
                format!("g.V(vertex_id).{}({})", direction_step, labels_str)
            } else {
                format!("g.V(vertex_id).{}()", direction_step)
            }
        } else {
            format!("g.V(vertex_id).{}()", direction_step)
        };

        if let Some(lim) = limit {
            gremlin.push_str(&format!(".limit({})", lim));
        }

        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let data = &response["result"]["data"];
        let result_data = if let Some(arr) = data.as_array() {
            arr.clone()
        } else if let Some(inner) = data.get("@value").and_then(Value::as_array) {
            inner.clone()
        } else {
            return Err(GraphError::InternalError(
                "Invalid response from Gremlin for get_adjacent_vertices".to_string(),
            ));
        };

        result_data
            .iter()
            .map(helpers::parse_vertex_from_gremlin)
            .collect()
    }

    fn get_connected_edges(
        &self,
        vertex_id: ElementId,
        direction: Direction,
        edge_types: Option<Vec<String>>,
        limit: Option<u32>,
    ) -> Result<Vec<Edge>, GraphError> {
        let mut bindings = serde_json::Map::new();
        let id_json = match vertex_id {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        bindings.insert("vertex_id".to_string(), id_json);

        let direction_step = match direction {
            Direction::Outgoing => "outE",
            Direction::Incoming => "inE",
            Direction::Both => "bothE",
        };

        let mut gremlin = if let Some(labels) = edge_types {
            if !labels.is_empty() {
                let label_bindings: Vec<String> = labels
                    .iter()
                    .enumerate()
                    .map(|(i, label)| {
                        let binding_key = format!("edge_label_{}", i);
                        bindings.insert(binding_key.clone(), json!(label));
                        binding_key
                    })
                    .collect();
                let labels_str = label_bindings.join(", ");
                format!("g.V(vertex_id).{}({})", direction_step, labels_str)
            } else {
                format!("g.V(vertex_id).{}()", direction_step)
            }
        } else {
            format!("g.V(vertex_id).{}()", direction_step)
        };

        if let Some(lim) = limit {
            gremlin.push_str(&format!(".limit({})", lim));
        }

        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let data = &response["result"]["data"];
        let result_data = if let Some(arr) = data.as_array() {
            arr.clone()
        } else if let Some(inner) = data.get("@value").and_then(Value::as_array) {
            inner.clone()
        } else {
            return Err(GraphError::InternalError(
                "Invalid response from Gremlin for get_connected_edges".to_string(),
            ));
        };

        result_data
            .iter()
            .map(helpers::parse_edge_from_gremlin)
            .collect()
    }

    fn create_vertices(&self, vertices: Vec<VertexSpec>) -> Result<Vec<Vertex>, GraphError> {
        if vertices.is_empty() {
            return Ok(vec![]);
        }

        let mut gremlin = "g".to_string();
        let mut bindings = serde_json::Map::new();

        for (i, spec) in vertices.iter().enumerate() {
            let label_binding = format!("l{}", i);
            gremlin.push_str(&format!(".addV({})", label_binding));
            bindings.insert(label_binding, json!(spec.vertex_type));

            for (j, (key, value)) in spec.properties.iter().enumerate() {
                let key_binding = format!("k_{}_{}", i, j);
                let val_binding = format!("v_{}_{}", i, j);
                gremlin.push_str(&format!(".property({}, {})", key_binding, val_binding));
                bindings.insert(key_binding, json!(key));
                bindings.insert(val_binding, conversions::to_json_value(value.clone())?);
            }
        }

        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"].as_array().ok_or_else(|| {
            GraphError::InternalError(
                "Invalid response from Gremlin for create_vertices".to_string(),
            )
        })?;

        result_data
            .iter()
            .map(helpers::parse_vertex_from_gremlin)
            .collect()
    }

    fn create_edges(&self, edges: Vec<EdgeSpec>) -> Result<Vec<Edge>, GraphError> {
        if edges.is_empty() {
            return Ok(vec![]);
        }

        let mut gremlin = String::new();
        let mut bindings = serde_json::Map::new();
        let mut edge_queries = Vec::new();

        for (i, edge_spec) in edges.iter().enumerate() {
            let from_binding = format!("from_{}", i);
            let to_binding = format!("to_{}", i);
            let label_binding = format!("label_{}", i);

            let from_id_json = match &edge_spec.from_vertex {
                ElementId::StringValue(s) => json!(s),
                ElementId::Int64(val) => json!(val),
                ElementId::Uuid(u) => json!(u.to_string()),
            };
            bindings.insert(from_binding.clone(), from_id_json);

            let to_id_json = match &edge_spec.to_vertex {
                ElementId::StringValue(s) => json!(s),
                ElementId::Int64(val) => json!(val),
                ElementId::Uuid(u) => json!(u.to_string()),
            };
            bindings.insert(to_binding.clone(), to_id_json);
            bindings.insert(label_binding.clone(), json!(edge_spec.edge_type));

            let mut edge_query = format!(
                "g.V({}).addE({}).to(g.V({}))",
                from_binding, label_binding, to_binding
            );

            for (j, (key, value)) in edge_spec.properties.iter().enumerate() {
                let key_binding = format!("k_{}_{}", i, j);
                let val_binding = format!("v_{}_{}", i, j);
                edge_query.push_str(&format!(".property({}, {})", key_binding, val_binding));
                bindings.insert(key_binding, json!(key));
                bindings.insert(val_binding, conversions::to_json_value(value.clone())?);
            }

            edge_queries.push(edge_query);
        }

        gremlin.push_str(&edge_queries.join(".next();"));
        gremlin.push_str(".elementMap().toList()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"].as_array().ok_or_else(|| {
            GraphError::InternalError("Invalid response from Gremlin for create_edges".to_string())
        })?;

        result_data
            .iter()
            .map(helpers::parse_edge_from_gremlin)
            .collect()
    }

    fn upsert_vertex(
        &self,
        _id: Option<ElementId>,
        vertex_type: String,
        properties: PropertyMap,
    ) -> Result<Vertex, GraphError> {
        if properties.is_empty() {
            return Err(GraphError::UnsupportedOperation(
                "Upsert requires at least one property to match on.".to_string(),
            ));
        }

        let mut gremlin_match = "g.V()".to_string();
        let mut bindings = serde_json::Map::new();

        for (i, (key, value)) in properties.iter().enumerate() {
            let key_binding = format!("mk_{}", i);
            let val_binding = format!("mv_{}", i);
            gremlin_match.push_str(&format!(".has({}, {})", key_binding, val_binding));
            bindings.insert(key_binding, json!(key.clone()));
            bindings.insert(val_binding, conversions::to_json_value(value.clone())?);
        }

        let mut gremlin_create = format!("addV('{}')", vertex_type);
        for (i, (key, value)) in properties.iter().enumerate() {
            let key_binding = format!("ck_{}", i);
            let val_binding = format!("cv_{}", i);
            gremlin_create.push_str(&format!(".property({}, {})", key_binding, val_binding));
            bindings.insert(key_binding, json!(key.clone()));
            bindings.insert(val_binding, conversions::to_json_value(value.clone())?);
        }

        let gremlin = format!(
            "{}.fold().coalesce(unfold(), {}).elementMap()",
            gremlin_match, gremlin_create
        );

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"]
            .as_array()
            .and_then(|arr| arr.first())
            .ok_or_else(|| {
                GraphError::InternalError(
                    "Invalid response from Gremlin for upsert_vertex".to_string(),
                )
            })?;

        helpers::parse_vertex_from_gremlin(result_data)
    }

    fn upsert_edge(
        &self,
        _id: Option<ElementId>,
        edge_label: String,
        from: ElementId,
        to: ElementId,
        properties: PropertyMap,
    ) -> Result<Edge, GraphError> {
        if properties.is_empty() {
            return Err(GraphError::UnsupportedOperation(
                "Upsert requires at least one property to match on.".to_string(),
            ));
        }

        let mut gremlin_match = "g.E()".to_string();
        let mut bindings = serde_json::Map::new();

        gremlin_match.push_str(".hasLabel(edge_label).has(\"_from\", from_id).has(\"_to\", to_id)");
        bindings.insert("edge_label".into(), json!(edge_label.clone()));
        bindings.insert(
            "from_id".into(),
            match from.clone() {
                ElementId::StringValue(s) => json!(s),
                ElementId::Int64(i) => json!(i),
                ElementId::Uuid(u) => json!(u),
            },
        );
        bindings.insert(
            "to_id".into(),
            match to.clone() {
                ElementId::StringValue(s) => json!(s),
                ElementId::Int64(i) => json!(i),
                ElementId::Uuid(u) => json!(u),
            },
        );

        for (i, (k, v)) in properties.iter().enumerate() {
            let mk = format!("ek_{}", i);
            let mv = format!("ev_{}", i);
            gremlin_match.push_str(&format!(".has({}, {})", mk, mv));
            bindings.insert(mk, json!(k));
            bindings.insert(mv, conversions::to_json_value(v.clone())?);
        }

        let mut gremlin_create =
            format!("addE('{}').from(__.V(from_id)).to(__.V(to_id))", edge_label);
        for (i, (k, v)) in properties.into_iter().enumerate() {
            let ck = format!("ck_{}", i);
            let cv = format!("cv_{}", i);
            gremlin_create.push_str(&format!(".property({}, {})", ck, cv));
            bindings.insert(ck, json!(k));
            bindings.insert(cv, conversions::to_json_value(v)?);
        }

        let gremlin = format!(
            "{}.fold().coalesce(unfold(), {}).elementMap()",
            gremlin_match, gremlin_create
        );

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;
        let result_data = response["result"]["data"]
            .as_array()
            .and_then(|arr| arr.first())
            .ok_or_else(|| {
                GraphError::InternalError("Invalid response from Gremlin for upsert_edge".into())
            })?;
        helpers::parse_edge_from_gremlin(result_data)
    }

    fn is_active(&self) -> bool {
        true
    }
}
