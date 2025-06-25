use crate::{conversions, helpers, Transaction};
use golem_graph::golem::graph::{
    errors::GraphError,
    transactions::{EdgeSpec, GuestTransaction, VertexSpec},
    types::{Direction, Edge, ElementId, FilterCondition, PropertyMap, SortSpec, Vertex},
};
use serde_json::json;

impl GuestTransaction for Transaction {
    fn commit(&self) -> Result<(), GraphError> {
        self.api.commit_transaction(&self.transaction_id)
    }

    fn rollback(&self) -> Result<(), GraphError> {
        self.api.rollback_transaction(&self.transaction_id)
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
        additional_labels: Vec<String>,
        properties: PropertyMap,
    ) -> Result<Vertex, GraphError> {
        if !additional_labels.is_empty() {
            return Err(GraphError::UnsupportedOperation(
                "ArangoDB does not support multiple labels per vertex. Use vertex collections instead."
                    .to_string(),
            ));
        }

        let props = conversions::to_arango_properties(properties)?;

        let query = json!({
            "query": "INSERT @props INTO @@collection OPTIONS { ignoreErrors: false } RETURN NEW",
            "bindVars": {
                "props": props,
                "@collection": vertex_type
            }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query)?;

        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;
        let vertex_doc = result_array
            .first()
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                GraphError::InternalError("Missing vertex document in response".to_string())
            })?;

        helpers::parse_vertex_from_document(vertex_doc, &vertex_type)
    }

    fn get_vertex(&self, id: ElementId) -> Result<Option<Vertex>, GraphError> {
        let key = helpers::element_id_to_key(&id)?;
        let collection = if let ElementId::StringValue(s) = &id {
            s.split('/').next().unwrap_or_default()
        } else {
            ""
        };

        if collection.is_empty() {
            return Err(GraphError::InvalidQuery(
                "ElementId for get_vertex must be a full _id (e.g., 'collection/key')".to_string(),
            ));
        }

        let query = json!({
            "query": "RETURN DOCUMENT(@@collection, @key)",
            "bindVars": {
                "@collection": collection,
                "key": key
            }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query)?;

        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;

        if let Some(vertex_doc) = result_array.first().and_then(|v| v.as_object()) {
            if vertex_doc.is_empty() || result_array.first().unwrap().is_null() {
                return Ok(None);
            }
            let vertex = helpers::parse_vertex_from_document(vertex_doc, collection)?;
            Ok(Some(vertex))
        } else {
            Ok(None)
        }
    }

    fn update_vertex(&self, id: ElementId, properties: PropertyMap) -> Result<Vertex, GraphError> {
        let key = helpers::element_id_to_key(&id)?;
        let collection = helpers::collection_from_element_id(&id)?;

        let props = conversions::to_arango_properties(properties)?;

        let query = json!({
            "query": "REPLACE @key WITH @props IN @@collection RETURN NEW",
            "bindVars": {
                "key": key,
                "props": props,
                "@collection": collection
            }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query)?;
        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;
        let vertex_doc = result_array
            .first()
            .and_then(|v| v.as_object())
            .ok_or_else(|| GraphError::ElementNotFound(id.clone()))?;

        helpers::parse_vertex_from_document(vertex_doc, collection)
    }

    fn update_vertex_properties(
        &self,
        id: ElementId,
        updates: PropertyMap,
    ) -> Result<Vertex, GraphError> {
        let key = helpers::element_id_to_key(&id)?;
        let collection = if let ElementId::StringValue(s) = &id {
            s.split('/').next().unwrap_or_default()
        } else {
            ""
        };

        if collection.is_empty() {
            return Err(GraphError::InvalidQuery(
                "ElementId for update_vertex_properties must be a full _id (e.g., 'collection/key')".to_string(),
            ));
        }

        let props = conversions::to_arango_properties(updates)?;

        let query = json!({
            "query": "UPDATE @key WITH @props IN @@collection OPTIONS { keepNull: false, mergeObjects: true } RETURN NEW",
            "bindVars": {
                "key": key,
                "props": props,
                "@collection": collection
            }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query)?;
        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;
        let vertex_doc = result_array
            .first()
            .and_then(|v| v.as_object())
            .ok_or_else(|| GraphError::ElementNotFound(id.clone()))?;

        helpers::parse_vertex_from_document(vertex_doc, collection)
    }

    fn delete_vertex(&self, id: ElementId, delete_edges: bool) -> Result<(), GraphError> {
        let key = helpers::element_id_to_key(&id)?;
        let collection = if let ElementId::StringValue(s) = &id {
            s.split('/').next().unwrap_or_default()
        } else {
            ""
        };

        if collection.is_empty() {
            return Err(GraphError::InvalidQuery(
                "ElementId for delete_vertex must be a full _id (e.g., 'collection/key')"
                    .to_string(),
            ));
        }

        if delete_edges {
            let vertex_id = helpers::element_id_to_string(&id);

            let collections = self.api.list_collections().unwrap_or_default();
            let edge_collections: Vec<_> = collections
                .iter()
                .filter(|c| {
                    matches!(
                        c.container_type,
                        golem_graph::golem::graph::schema::ContainerType::EdgeContainer
                    )
                })
                .map(|c| c.name.clone())
                .collect();

            for edge_collection in edge_collections {
                let delete_edges_query = json!({
                    "query": "FOR e IN @@collection FILTER e._from == @vertex_id OR e._to == @vertex_id REMOVE e IN @@collection",
                    "bindVars": {
                        "vertex_id": vertex_id,
                        "@collection": edge_collection
                    }
                });
                let _ = self
                    .api
                    .execute_in_transaction(&self.transaction_id, delete_edges_query);
            }
        }

        let simple_query = json!({
            "query": "REMOVE @key IN @@collection",
            "bindVars": {
                "key": key,
                "@collection": collection
            }
        });

        self.api
            .execute_in_transaction(&self.transaction_id, simple_query)?;
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
        let collection = vertex_type.ok_or_else(|| {
            GraphError::InvalidQuery("vertex_type must be provided for find_vertices".to_string())
        })?;

        let mut query_parts = vec![format!("FOR v IN @@collection")];
        let mut bind_vars = serde_json::Map::new();
        bind_vars.insert("@collection".to_string(), json!(collection.clone()));

        let where_clause = golem_graph::query_utils::build_where_clause(
            &filters,
            "v",
            &mut bind_vars,
            &aql_syntax(),
            conversions::to_arango_value,
        )?;
        if !where_clause.is_empty() {
            query_parts.push(where_clause);
        }

        let sort_clause = golem_graph::query_utils::build_sort_clause(&sort, "v");
        if !sort_clause.is_empty() {
            query_parts.push(sort_clause);
        }

        let limit_val = limit.unwrap_or(100); // Default limit
        let offset_val = offset.unwrap_or(0);
        query_parts.push(format!("LIMIT {}, {}", offset_val, limit_val));
        query_parts.push("RETURN v".to_string());

        let full_query = query_parts.join(" ");

        let query_json = json!({
            "query": full_query,
            "bindVars": bind_vars
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query_json)?;

        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;

        let mut vertices = vec![];
        for val in result_array {
            if let Some(doc) = val.as_object() {
                let vertex = helpers::parse_vertex_from_document(doc, &collection)?;
                vertices.push(vertex);
            }
        }

        Ok(vertices)
    }

    fn create_edge(
        &self,
        edge_type: String,
        from_vertex: ElementId,
        to_vertex: ElementId,
        properties: PropertyMap,
    ) -> Result<Edge, GraphError> {
        let props = conversions::to_arango_properties(properties)?;
        let from_id = helpers::element_id_to_string(&from_vertex);
        let to_id = helpers::element_id_to_string(&to_vertex);

        let query = json!({
            "query": "INSERT MERGE({ _from: @from, _to: @to }, @props) INTO @@collection RETURN NEW",
            "bindVars": {
                "from": from_id,
                "to": to_id,
                "props": props,
                "@collection": edge_type
            }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query)?;
        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;
        let edge_doc = result_array
            .first()
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                GraphError::InternalError("Missing edge document in response".to_string())
            })?;

        helpers::parse_edge_from_document(edge_doc, &edge_type)
    }

    fn get_edge(&self, id: ElementId) -> Result<Option<Edge>, GraphError> {
        let key = helpers::element_id_to_key(&id)?;
        let collection = if let ElementId::StringValue(s) = &id {
            s.split('/').next().unwrap_or_default()
        } else {
            ""
        };

        if collection.is_empty() {
            return Err(GraphError::InvalidQuery(
                "ElementId for get_edge must be a full _id (e.g., 'collection/key')".to_string(),
            ));
        }

        let query = json!({
            "query": "RETURN DOCUMENT(@@collection, @key)",
            "bindVars": {
                "@collection": collection,
                "key": key
            }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query)?;
        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;

        if let Some(edge_doc) = result_array.first().and_then(|v| v.as_object()) {
            if edge_doc.is_empty() || result_array.first().unwrap().is_null() {
                return Ok(None);
            }
            let edge = helpers::parse_edge_from_document(edge_doc, collection)?;
            Ok(Some(edge))
        } else {
            Ok(None)
        }
    }

    fn update_edge(&self, id: ElementId, properties: PropertyMap) -> Result<Edge, GraphError> {
        let key = helpers::element_id_to_key(&id)?;
        let collection = helpers::collection_from_element_id(&id)?;

        // First getting the current edge to preserve _from and _to
        let current_edge = self
            .get_edge(id.clone())?
            .ok_or_else(|| GraphError::ElementNotFound(id.clone()))?;

        let mut props = conversions::to_arango_properties(properties)?;
        // Preserving _from and _to for edge replacement
        props.insert(
            "_from".to_string(),
            json!(helpers::element_id_to_string(&current_edge.from_vertex)),
        );
        props.insert(
            "_to".to_string(),
            json!(helpers::element_id_to_string(&current_edge.to_vertex)),
        );

        let query = json!({
            "query": "REPLACE @key WITH @props IN @@collection RETURN NEW",
            "bindVars": {
                "key": key,
                "props": props,
                "@collection": collection,
            }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query)?;

        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;

        let edge_doc = result_array
            .first()
            .and_then(|v| v.as_object())
            .ok_or_else(|| GraphError::ElementNotFound(id.clone()))?;

        helpers::parse_edge_from_document(edge_doc, collection)
    }

    fn update_edge_properties(
        &self,
        id: ElementId,
        updates: PropertyMap,
    ) -> Result<Edge, GraphError> {
        let key = helpers::element_id_to_key(&id)?;
        let collection = if let ElementId::StringValue(s) = &id {
            s.split('/').next().unwrap_or_default()
        } else {
            ""
        };

        if collection.is_empty() {
            return Err(GraphError::InvalidQuery(
                "ElementId for update_edge_properties must be a full _id (e.g., 'collection/key')"
                    .to_string(),
            ));
        }

        let props = conversions::to_arango_properties(updates)?;

        let query = json!({
            "query": "UPDATE @key WITH @props IN @@collection OPTIONS { keepNull: false, mergeObjects: true } RETURN NEW",
            "bindVars": {
                "key": key,
                "props": props,
                "@collection": collection
            }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query)?;
        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;
        let edge_doc = result_array
            .first()
            .and_then(|v| v.as_object())
            .ok_or_else(|| GraphError::ElementNotFound(id.clone()))?;

        helpers::parse_edge_from_document(edge_doc, collection)
    }

    fn delete_edge(&self, id: ElementId) -> Result<(), GraphError> {
        let key = helpers::element_id_to_key(&id)?;
        let collection = if let ElementId::StringValue(s) = &id {
            s.split('/').next().unwrap_or_default()
        } else {
            ""
        };

        if collection.is_empty() {
            return Err(GraphError::InvalidQuery(
                "ElementId for delete_edge must be a full _id (e.g., 'collection/key')".to_string(),
            ));
        }

        let query = json!({
            "query": "REMOVE @key IN @@collection",
            "bindVars": {
                "key": key,
                "@collection": collection
            }
        });

        self.api
            .execute_in_transaction(&self.transaction_id, query)?;
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
        let collection = edge_types.and_then(|mut et| et.pop()).ok_or_else(|| {
            GraphError::InvalidQuery("An edge_type must be provided for find_edges".to_string())
        })?;

        let mut query_parts = vec![format!("FOR e IN @@collection")];
        let mut bind_vars = serde_json::Map::new();
        bind_vars.insert("@collection".to_string(), json!(collection.clone()));

        let where_clause = golem_graph::query_utils::build_where_clause(
            &filters,
            "e",
            &mut bind_vars,
            &aql_syntax(),
            conversions::to_arango_value,
        )?;
        if !where_clause.is_empty() {
            query_parts.push(where_clause);
        }

        let sort_clause = golem_graph::query_utils::build_sort_clause(&sort, "e");
        if !sort_clause.is_empty() {
            query_parts.push(sort_clause);
        }

        let limit_val = limit.unwrap_or(100);
        let offset_val = offset.unwrap_or(0);
        query_parts.push(format!("LIMIT {}, {}", offset_val, limit_val));
        query_parts.push("RETURN e".to_string());

        let full_query = query_parts.join(" ");

        let query_json = json!({
            "query": full_query,
            "bindVars": bind_vars
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query_json)?;

        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;

        let mut edges = vec![];
        for val in result_array {
            if let Some(doc) = val.as_object() {
                let edge = helpers::parse_edge_from_document(doc, &collection)?;
                edges.push(edge);
            }
        }

        Ok(edges)
    }

    fn get_adjacent_vertices(
        &self,
        vertex_id: ElementId,
        direction: Direction,
        edge_types: Option<Vec<String>>,
        _limit: Option<u32>,
    ) -> Result<Vec<Vertex>, GraphError> {
        let start_node = helpers::element_id_to_string(&vertex_id);
        let dir_str = match direction {
            Direction::Outgoing => "OUTBOUND",
            Direction::Incoming => "INBOUND",
            Direction::Both => "ANY",
        };

        let collections = edge_types.unwrap_or_default().join(", ");

        let query = json!({
            "query": format!(
                "FOR v IN 1..1 {} @start_node {} RETURN v",
                dir_str,
                collections
            ),
            "bindVars": {
                "start_node": start_node,
            }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query)?;
        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;

        let mut vertices = vec![];
        for val in result_array {
            if let Some(doc) = val.as_object() {
                let collection = doc
                    .get("_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.split('/').next())
                    .unwrap_or_default();
                let vertex = helpers::parse_vertex_from_document(doc, collection)?;
                vertices.push(vertex);
            }
        }

        Ok(vertices)
    }

    fn get_connected_edges(
        &self,
        vertex_id: ElementId,
        direction: Direction,
        edge_types: Option<Vec<String>>,
        _limit: Option<u32>,
    ) -> Result<Vec<Edge>, GraphError> {
        let start_node = helpers::element_id_to_string(&vertex_id);
        let dir_str = match direction {
            Direction::Outgoing => "OUTBOUND",
            Direction::Incoming => "INBOUND",
            Direction::Both => "ANY",
        };

        let collections = edge_types.unwrap_or_default().join(", ");

        let query = json!({
            "query": format!(
                "FOR v, e IN 1..1 {} @start_node {} RETURN e",
                dir_str,
                collections
            ),
            "bindVars": {
                "start_node": start_node,
            }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query)?;
        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;

        let mut edges = vec![];
        for val in result_array {
            if let Some(doc) = val.as_object() {
                let collection = doc
                    .get("_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.split('/').next())
                    .unwrap_or_default();
                let edge = helpers::parse_edge_from_document(doc, collection)?;
                edges.push(edge);
            }
        }

        Ok(edges)
    }

    fn create_vertices(&self, vertices: Vec<VertexSpec>) -> Result<Vec<Vertex>, GraphError> {
        let mut created_vertices = vec![];
        for vertex_spec in vertices {
            let vertex = self.create_vertex_with_labels(
                vertex_spec.vertex_type,
                vertex_spec.additional_labels.unwrap_or_default(),
                vertex_spec.properties,
            )?;
            created_vertices.push(vertex);
        }
        Ok(created_vertices)
    }

    fn create_edges(&self, edges: Vec<EdgeSpec>) -> Result<Vec<Edge>, GraphError> {
        let mut created_edges = vec![];
        for edge_spec in edges {
            let edge = self.create_edge(
                edge_spec.edge_type,
                edge_spec.from_vertex,
                edge_spec.to_vertex,
                edge_spec.properties,
            )?;
            created_edges.push(edge);
        }
        Ok(created_edges)
    }

    fn upsert_vertex(
        &self,
        id: Option<ElementId>,
        vertex_type: String,
        properties: PropertyMap,
    ) -> Result<Vertex, GraphError> {
        let props = conversions::to_arango_properties(properties)?;
        let search = if let Some(i) = id.clone() {
            let key = helpers::element_id_to_key(&i)?;
            json!({ "_key": key })
        } else {
            return Err(GraphError::UnsupportedOperation(
                "upsert_vertex without an ID requires key properties, which is not yet supported."
                    .to_string(),
            ));
        };

        let query = json!({
            "query": "UPSERT @search INSERT @props UPDATE @props IN @@collection RETURN NEW",
            "bindVars": {
                "search": search,
                "props": props,
                "@collection": vertex_type
            }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query)?;
        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;
        let vertex_doc = result_array
            .first()
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                GraphError::InternalError("Missing vertex document in upsert response".to_string())
            })?;

        helpers::parse_vertex_from_document(vertex_doc, &vertex_type)
    }

    fn upsert_edge(
        &self,
        id: Option<ElementId>,
        edge_type: String,
        from_vertex: ElementId,
        to_vertex: ElementId,
        properties: PropertyMap,
    ) -> Result<Edge, GraphError> {
        let mut props = conversions::to_arango_properties(properties)?;
        props.insert(
            "_from".to_string(),
            json!(helpers::element_id_to_string(&from_vertex)),
        );
        props.insert(
            "_to".to_string(),
            json!(helpers::element_id_to_string(&to_vertex)),
        );

        let search = if let Some(i) = id {
            let key = helpers::element_id_to_key(&i)?;
            json!({ "_key": key })
        } else {
            return Err(GraphError::UnsupportedOperation(
                "upsert_edge without an ID requires key properties, which is not yet supported."
                    .to_string(),
            ));
        };

        let query = json!({
            "query": "UPSERT @search INSERT @props UPDATE @props IN @@collection RETURN NEW",
            "bindVars": {
                "search": search,
                "props": props,
                "@collection": edge_type
            }
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query)?;
        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL response".to_string())
        })?;
        let edge_doc = result_array
            .first()
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                GraphError::InternalError("Missing edge document in upsert response".to_string())
            })?;

        helpers::parse_edge_from_document(edge_doc, &edge_type)
    }

    fn is_active(&self) -> bool {
        self.api
            .get_transaction_status(&self.transaction_id)
            .map(|status| status == "running")
            .unwrap_or(false)
    }
}

fn aql_syntax() -> golem_graph::query_utils::QuerySyntax {
    golem_graph::query_utils::QuerySyntax {
        equal: "==",
        not_equal: "!=",
        less_than: "<",
        less_than_or_equal: "<=",
        greater_than: ">",
        greater_than_or_equal: ">=",
        contains: "CONTAINS",
        starts_with: "STARTS_WITH",
        ends_with: "ENDS_WITH",
        regex_match: "=~",
        param_prefix: "@",
    }
}
