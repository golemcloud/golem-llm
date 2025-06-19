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

impl GuestTransaction for Transaction {
    fn commit(&self) -> Result<(), GraphError> {
        // In a sessionless, per-request transaction model, each request is a transaction.
        // So, commit is implicitly handled.
        Ok(())
    }

    fn rollback(&self) -> Result<(), GraphError> {
        // In a sessionless, per-request transaction model, there's nothing to roll back
        // once a request has been made.
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

        let result_data = response["result"]["data"]
            .as_array()
            .and_then(|arr| arr.first())
            .ok_or_else(|| {
                GraphError::InternalError(
                    "Invalid response from Gremlin for create_vertex".to_string(),
                )
            })?;

        helpers::parse_vertex_from_gremlin(result_data)
    }

    fn get_vertex(&self, id: ElementId) -> Result<Option<Vertex>, GraphError> {
        let gremlin = "g.V(vertex_id).elementMap()".to_string();

        let id_json = match id.clone() {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };

        let mut bindings = serde_json::Map::new();
        bindings.insert("vertex_id".to_string(), id_json);

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"].as_array();

        match result_data {
            Some(arr) if !arr.is_empty() => {
                let vertex_value = &arr[0];
                let vertex = helpers::parse_vertex_from_gremlin(vertex_value)?;
                Ok(Some(vertex))
            }
            _ => Ok(None),
        }
    }

    fn update_vertex(&self, id: ElementId, properties: PropertyMap) -> Result<Vertex, GraphError> {
        // This Gremlin query finds the vertex, drops its existing properties as a side effect,
        // then adds the new properties from the bindings.
        let mut gremlin = "g.V(vertex_id).sideEffect(properties().drop())".to_string();
        let mut bindings = serde_json::Map::new();

        let id_json = match id.clone() {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        bindings.insert("vertex_id".to_string(), id_json);

        for (i, (key, value)) in properties.into_iter().enumerate() {
            let key_binding = format!("k{}", i);
            let val_binding = format!("v{}", i);
            gremlin.push_str(&format!(".property({}, {})", key_binding, val_binding));
            bindings.insert(key_binding, json!(key));
            bindings.insert(val_binding, conversions::to_json_value(value)?);
        }

        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"]
            .as_array()
            .and_then(|arr| arr.first())
            .ok_or(GraphError::ElementNotFound(id))?;

        helpers::parse_vertex_from_gremlin(result_data)
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

        let id_json = match id.clone() {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        bindings.insert("vertex_id".to_string(), id_json);

        for (i, (key, value)) in updates.into_iter().enumerate() {
            let key_binding = format!("k{}", i);
            let val_binding = format!("v{}", i);
            gremlin.push_str(&format!(".property({}, {})", key_binding, val_binding));
            bindings.insert(key_binding, json!(key));
            bindings.insert(val_binding, conversions::to_json_value(value)?);
        }

        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"]
            .as_array()
            .and_then(|arr| arr.first())
            .ok_or(GraphError::ElementNotFound(id))?;

        helpers::parse_vertex_from_gremlin(result_data)
    }

    fn delete_vertex(&self, id: ElementId, _delete_edges: bool) -> Result<(), GraphError> {
        // In Gremlin, drop() removes the vertex and all its incident edges, so `delete_edges` is implicitly true.
        let gremlin = "g.V(vertex_id).drop()".to_string();

        let id_json = match id {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };

        let mut bindings = serde_json::Map::new();
        bindings.insert("vertex_id".to_string(), id_json);

        self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

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

        let result_data = response["result"]["data"].as_array().ok_or_else(|| {
            GraphError::InternalError("Invalid response from Gremlin for find_vertices".to_string())
        })?;

        result_data
            .iter()
            .map(helpers::parse_vertex_from_gremlin)
            .collect()
    }

    fn create_edge(
        &self,
        edge_type: String,
        from_vertex: ElementId,
        to_vertex: ElementId,
        properties: PropertyMap,
    ) -> Result<Edge, GraphError> {
        let mut gremlin = "g.V(from_id).addE(edge_label).to(g.V(to_id))".to_string();
        let mut bindings = serde_json::Map::new();

        let from_id_json = match from_vertex {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        bindings.insert("from_id".to_string(), from_id_json);

        let to_id_json = match to_vertex {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        bindings.insert("to_id".to_string(), to_id_json);
        bindings.insert("edge_label".to_string(), json!(edge_type));

        for (i, (key, value)) in properties.into_iter().enumerate() {
            let binding_key = format!("p{}", i);
            gremlin.push_str(&format!(".property(k{}, {})", i, binding_key));
            bindings.insert(format!("k{}", i), json!(key));
            bindings.insert(binding_key, conversions::to_json_value(value)?);
        }

        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"]
            .as_array()
            .and_then(|arr| arr.first())
            .ok_or_else(|| {
                GraphError::InternalError(
                    "Invalid response from Gremlin for create_edge".to_string(),
                )
            })?;

        helpers::parse_edge_from_gremlin(result_data)
    }

    fn get_edge(&self, id: ElementId) -> Result<Option<Edge>, GraphError> {
        let gremlin = "g.E(edge_id).elementMap()".to_string();

        let id_json = match id.clone() {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };

        let mut bindings = serde_json::Map::new();
        bindings.insert("edge_id".to_string(), id_json);

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"].as_array();

        match result_data {
            Some(arr) if !arr.is_empty() => {
                let edge_value = &arr[0];
                let edge = helpers::parse_edge_from_gremlin(edge_value)?;
                Ok(Some(edge))
            }
            _ => Ok(None),
        }
    }

    fn update_edge(&self, id: ElementId, properties: PropertyMap) -> Result<Edge, GraphError> {
        let mut gremlin = "g.E(edge_id).sideEffect(properties().drop())".to_string();
        let mut bindings = serde_json::Map::new();

        let id_json = match id.clone() {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        bindings.insert("edge_id".to_string(), id_json);

        for (i, (key, value)) in properties.into_iter().enumerate() {
            let key_binding = format!("k{}", i);
            let val_binding = format!("v{}", i);
            gremlin.push_str(&format!(".property({}, {})", key_binding, val_binding));
            bindings.insert(key_binding, json!(key));
            bindings.insert(val_binding, conversions::to_json_value(value)?);
        }

        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"]
            .as_array()
            .and_then(|arr| arr.first())
            .ok_or(GraphError::ElementNotFound(id))?;

        helpers::parse_edge_from_gremlin(result_data)
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

        let id_json = match id.clone() {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        bindings.insert("edge_id".to_string(), id_json);

        for (i, (key, value)) in updates.into_iter().enumerate() {
            let key_binding = format!("k{}", i);
            let val_binding = format!("v{}", i);
            gremlin.push_str(&format!(".property({}, {})", key_binding, val_binding));
            bindings.insert(key_binding, json!(key));
            bindings.insert(val_binding, conversions::to_json_value(value)?);
        }

        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"]
            .as_array()
            .and_then(|arr| arr.first())
            .ok_or(GraphError::ElementNotFound(id))?;

        helpers::parse_edge_from_gremlin(result_data)
    }

    fn delete_edge(&self, id: ElementId) -> Result<(), GraphError> {
        let gremlin = "g.E(edge_id).drop()".to_string();

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
                // Gremlin's hasLabel can take multiple labels
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

        let mut labels_str = "".to_string();
        if let Some(labels) = edge_types {
            if !labels.is_empty() {
                bindings.insert("edge_labels".to_string(), json!(labels));
                labels_str = "edge_labels".to_string();
            }
        }

        let mut gremlin = format!("g.V(vertex_id).{}({})", direction_step, labels_str);

        if let Some(lim) = limit {
            gremlin.push_str(&format!(".limit({})", lim));
        }

        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"].as_array().ok_or_else(|| {
            GraphError::InternalError(
                "Invalid response from Gremlin for get_adjacent_vertices".to_string(),
            )
        })?;

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

        let mut labels_str = "".to_string();
        if let Some(labels) = edge_types {
            if !labels.is_empty() {
                bindings.insert("edge_labels".to_string(), json!(labels));
                labels_str = "edge_labels".to_string();
            }
        }

        let mut gremlin = format!("g.V(vertex_id).{}({})", direction_step, labels_str);

        if let Some(lim) = limit {
            gremlin.push_str(&format!(".limit({})", lim));
        }

        gremlin.push_str(".elementMap()");

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"].as_array().ok_or_else(|| {
            GraphError::InternalError(
                "Invalid response from Gremlin for get_connected_edges".to_string(),
            )
        })?;

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
        edge_type: String,
        from_vertex: ElementId,
        to_vertex: ElementId,
        properties: PropertyMap,
    ) -> Result<Edge, GraphError> {
        let mut bindings = serde_json::Map::new();

        let from_id_json = match from_vertex.clone() {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        bindings.insert("from_id".to_string(), from_id_json);

        let to_id_json = match to_vertex.clone() {
            ElementId::StringValue(s) => json!(s),
            ElementId::Int64(i) => json!(i),
            ElementId::Uuid(u) => json!(u.to_string()),
        };
        bindings.insert("to_id".to_string(), to_id_json);
        bindings.insert("edge_label".to_string(), json!(edge_type));

        let mut gremlin_create = "addE(edge_label).to(g.V(to_id))".to_string();
        for (i, (key, value)) in properties.iter().enumerate() {
            let key_binding = format!("ck_{}", i);
            let val_binding = format!("cv_{}", i);
            gremlin_create.push_str(&format!(".property({}, {})", key_binding, val_binding));
            bindings.insert(key_binding, json!(key.clone()));
            bindings.insert(val_binding, conversions::to_json_value(value.clone())?);
        }

        // The query finds an existing edge or creates a new one.
        // It's complex because we need to match direction and label.
        let gremlin = format!(
            "g.V(from_id).outE(edge_label).where(inV().hasId(to_id)).fold().coalesce(unfold(), {})",
            gremlin_create
        );

        let response = self.api.execute(&gremlin, Some(Value::Object(bindings)))?;

        let result_data = response["result"]["data"]
            .as_array()
            .and_then(|arr| arr.first())
            .ok_or_else(|| {
                GraphError::InternalError(
                    "Invalid response from Gremlin for upsert_edge".to_string(),
                )
            })?;

        helpers::parse_edge_from_gremlin(result_data)
    }

    fn is_active(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::JanusGraphApi;
    use golem_graph::golem::graph::types::PropertyValue;
    use std::env;
    use std::sync::Arc;

    fn create_test_transaction() -> Transaction {
        let host = env::var("JANUSGRAPH_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port = env::var("JANUSGRAPH_PORT")
            .unwrap_or_else(|_| "8182".to_string())
            .parse()
            .unwrap();
        let api = JanusGraphApi::new(&host, port, None, None).unwrap();
        Transaction { api: Arc::new(api) }
    }

    #[test]
    fn test_create_and_get_vertex() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_create_and_get_vertex: JANUSGRAPH_HOST not set");
            return;
        }

        let tx = create_test_transaction();
        let vertex_type = "person".to_string();
        let properties = vec![(
            "name".to_string(),
            PropertyValue::StringValue("Alice".to_string()),
        )];

        let created_vertex = tx
            .create_vertex(vertex_type.clone(), properties.clone())
            .unwrap();
        assert_eq!(created_vertex.vertex_type, vertex_type);

        let retrieved_vertex = tx.get_vertex(created_vertex.id.clone()).unwrap().unwrap();
        assert_eq!(retrieved_vertex.id, created_vertex.id);
        assert_eq!(
            retrieved_vertex.properties[0].1,
            PropertyValue::StringValue("Alice".to_string())
        );

        tx.delete_vertex(created_vertex.id, true).unwrap();
    }

    #[test]
    fn test_create_and_delete_edge() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_create_and_delete_edge: JANUSGRAPH_HOST not set");
            return;
        }

        let tx = create_test_transaction();

        let v1 = tx.create_vertex("person".to_string(), vec![]).unwrap();
        let v2 = tx.create_vertex("person".to_string(), vec![]).unwrap();

        let created_edge = tx
            .create_edge("knows".to_string(), v1.id.clone(), v2.id.clone(), vec![])
            .unwrap();
        assert_eq!(created_edge.edge_type, "knows");

        tx.delete_edge(created_edge.id.clone()).unwrap();
        let retrieved_edge = tx.get_edge(created_edge.id).unwrap();
        assert!(retrieved_edge.is_none());

        tx.delete_vertex(v1.id, true).unwrap();
        tx.delete_vertex(v2.id, true).unwrap();
    }

    #[test]
    fn test_update_vertex_properties() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_update_vertex_properties: JANUSGRAPH_HOST not set");
            return;
        }

        let tx = create_test_transaction();
        let vertex_type = "character".to_string();
        let initial_properties = vec![(
            "name".to_string(),
            PropertyValue::StringValue("Gandalf".to_string()),
        )];

        let created_vertex = tx
            .create_vertex(vertex_type.clone(), initial_properties)
            .unwrap();

        let updated_properties = vec![(
            "name".to_string(),
            PropertyValue::StringValue("Gandalf the White".to_string()),
        )];
        let updated_vertex = tx
            .update_vertex_properties(created_vertex.id.clone(), updated_properties)
            .unwrap();

        let retrieved_name = updated_vertex
            .properties
            .iter()
            .find(|(k, _)| k == "name")
            .unwrap();
        assert_eq!(
            retrieved_name.1,
            PropertyValue::StringValue("Gandalf the White".to_string())
        );

        tx.delete_vertex(created_vertex.id, true).unwrap();
    }

    #[test]
    fn test_update_edge_properties() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_update_edge_properties: JANUSGRAPH_HOST not set");
            return;
        }

        let tx = create_test_transaction();

        let v1 = tx.create_vertex("person".to_string(), vec![]).unwrap();
        let v2 = tx.create_vertex("person".to_string(), vec![]).unwrap();

        let initial_properties = vec![("weight".to_string(), PropertyValue::Float64(1.0))];
        let created_edge = tx
            .create_edge(
                "knows".to_string(),
                v1.id.clone(),
                v2.id.clone(),
                initial_properties,
            )
            .unwrap();

        let updated_properties = vec![("weight".to_string(), PropertyValue::Float64(2.0))];
        tx.update_edge_properties(created_edge.id.clone(), updated_properties)
            .unwrap();

        let retrieved_edge = tx.get_edge(created_edge.id.clone()).unwrap().unwrap();
        let retrieved_weight = retrieved_edge
            .properties
            .iter()
            .find(|(k, _)| k == "weight")
            .unwrap();
        assert_eq!(retrieved_weight.1, PropertyValue::Float64(2.0));

        tx.delete_vertex(v1.id, true).unwrap();
        tx.delete_vertex(v2.id, true).unwrap();
    }

    #[test]
    fn test_update_vertex_replaces_properties() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_update_vertex_replaces_properties: JANUSGRAPH_HOST not set");
            return;
        }

        let tx = create_test_transaction();
        let initial_properties = vec![
            (
                "name".to_string(),
                PropertyValue::StringValue("test".to_string()),
            ),
            (
                "status".to_string(),
                PropertyValue::StringValue("initial".to_string()),
            ),
        ];
        let vertex = tx
            .create_vertex("test_v".to_string(), initial_properties)
            .unwrap();

        let new_properties = vec![
            (
                "name".to_string(),
                PropertyValue::StringValue("test_updated".to_string()),
            ),
            (
                "new_prop".to_string(),
                PropertyValue::StringValue("added".to_string()),
            ),
        ];
        let updated_vertex = tx.update_vertex(vertex.id.clone(), new_properties).unwrap();

        assert_eq!(updated_vertex.properties.len(), 2);
        let updated_name = updated_vertex
            .properties
            .iter()
            .find(|(k, _)| k == "name")
            .unwrap()
            .1
            .clone();
        let new_prop = updated_vertex
            .properties
            .iter()
            .find(|(k, _)| k == "new_prop")
            .unwrap()
            .1
            .clone();
        assert_eq!(
            updated_name,
            PropertyValue::StringValue("test_updated".to_string())
        );
        assert_eq!(new_prop, PropertyValue::StringValue("added".to_string()));
        assert!(updated_vertex.properties.iter().any(|(k, _)| k == "status"));

        tx.delete_vertex(vertex.id, true).unwrap();
    }

    #[test]
    fn test_update_edge_replaces_properties() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_update_edge_replaces_properties: JANUSGRAPH_HOST not set");
            return;
        }

        let tx = create_test_transaction();
        let v1 = tx.create_vertex("person".to_string(), vec![]).unwrap();
        let v2 = tx.create_vertex("person".to_string(), vec![]).unwrap();

        let initial_properties = vec![
            ("weight".to_string(), PropertyValue::Float64(1.0)),
            (
                "type".to_string(),
                PropertyValue::StringValue("original".to_string()),
            ),
        ];
        let edge = tx
            .create_edge(
                "rel".to_string(),
                v1.id.clone(),
                v2.id.clone(),
                initial_properties,
            )
            .unwrap();

        // Replace properties
        let new_properties = vec![
            ("weight".to_string(), PropertyValue::Float64(2.0)),
            (
                "notes".to_string(),
                PropertyValue::StringValue("replaced".to_string()),
            ),
        ];
        let updated_edge = tx.update_edge(edge.id.clone(), new_properties).unwrap();

        assert_eq!(updated_edge.properties.len(), 2);
        let updated_weight = updated_edge
            .properties
            .iter()
            .find(|(k, _)| k == "weight")
            .unwrap()
            .1
            .clone();
        let new_prop = updated_edge
            .properties
            .iter()
            .find(|(k, _)| k == "notes")
            .unwrap()
            .1
            .clone();
        assert_eq!(updated_weight, PropertyValue::Float64(2.0));
        assert_eq!(new_prop, PropertyValue::StringValue("replaced".to_string()));
        assert!(updated_edge.properties.iter().any(|(k, _)| k == "type"));

        tx.delete_vertex(v1.id, true).unwrap();
        tx.delete_vertex(v2.id, true).unwrap();
    }

    #[test]
    fn test_transaction_commit() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_transaction_commit: JANUSGRAPH_HOST not set");
            return;
        }

        let tx = create_test_transaction();
        let result = tx.commit();
        assert!(result.is_ok());
    }

    #[test]
    fn test_transaction_rollback() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_transaction_rollback: JANUSGRAPH_HOST not set");
            return;
        }

        let tx = create_test_transaction();
        let result = tx.rollback();
        assert!(result.is_ok());
    }

    #[test]
    fn test_unsupported_upsert_operations() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_unsupported_upsert_operations: JANUSGRAPH_HOST not set");
            return;
        }
        let tx = create_test_transaction();

        let v1 = tx.create_vertex("person".to_string(), vec![]).unwrap();

        let upsert_vertex_result = tx.upsert_vertex(None, "person".to_string(), vec![]);
        assert!(matches!(
            upsert_vertex_result,
            Err(GraphError::UnsupportedOperation(_))
        ));

        let upsert_edge_result = tx.upsert_edge(
            None,
            "knows".to_string(),
            v1.id.clone(),
            v1.id.clone(),
            vec![],
        );
        assert!(matches!(
            upsert_edge_result,
            Err(GraphError::UnsupportedOperation(_))
        ));

        tx.commit().unwrap();
    }
}
