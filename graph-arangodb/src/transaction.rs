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
            "query": "INSERT @props INTO @@collection RETURN NEW",
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

        let _query = if delete_edges {
            json!({
                "query": "FOR v, e IN 1..1 ANY @vertex_id GRAPH @graph_name REMOVE e IN @@edge_collection REMOVE v IN @@vertex_collection",
                "bindVars": {
                    "vertex_id": helpers::element_id_to_string(&id), // This assumes a graph is defined. This is complex.
                    // A simpler, graph-agnostic way is needed.
                    // For now, let's just delete the vertex and let ArangoDB handle edge deletion if configured.
                    // A better implementation requires graph name.
                    "key": key,
                    "@collection": collection
                }
            })
        } else {
            json!({
                "query": "REMOVE @key IN @@collection",
                "bindVars": {
                    "key": key,
                    "@collection": collection
                }
            })
        };
        let simple_query = json!({
            "query": "REMOVE @key IN @@collection",
            "bindVars": {
                "key": key,
                "@collection": collection
            }
        });

        // The logic for deleting edges is complex and often depends on a named graph in ArangoDB.
        // For a generic implementation, we will only delete the vertex. The user is expected
        // to handle edge deletion if `delete_edges` is true.
        if delete_edges {
            return Err(GraphError::UnsupportedOperation(
                "delete_edges=true is not supported yet. Please delete edges manually.".to_string(),
            ));
        }

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
            "query": "INSERT { _from: @from, _to: @to, ...@props } INTO @@collection RETURN NEW",
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

        let props = conversions::to_arango_properties(properties)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Transaction;
    use golem_graph::golem::graph::errors::GraphError;
    use golem_graph::golem::graph::types::PropertyValue;
    use std::env;
    use std::sync::Arc;

    fn create_test_transaction() -> Transaction {
        let host = env::var("ARANGO_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port: u16 = env::var("ARANGO_PORT")
            .unwrap_or_else(|_| "8529".to_string())
            .parse()
            .expect("Invalid ARANGO_PORT");
        let user = env::var("ARANGO_USER").unwrap_or_else(|_| "root".to_string());
        let password = env::var("ARANGO_PASSWORD").unwrap_or_else(|_| "".to_string());
        let database = env::var("ARANGO_DATABASE").unwrap_or_else(|_| "test".to_string());

        let api = crate::client::ArangoDbApi::new(&host, port, &user, &password, &database);

        let tx_id = api
            .begin_transaction(false)
            .expect("Failed to begin ArangoDB transaction");
        Transaction::new(Arc::new(api), tx_id)
    }

    #[test]
    fn test_create_and_get_vertex() {
        if env::var("ARANGO_HOST").is_err() {
            println!("Skipping test_create_and_get_vertex: ARANGO_HOST not set");
            return;
        }

        let tx = create_test_transaction();
        let vertex_type = "person".to_string();
        let props = vec![(
            "name".to_string(),
            PropertyValue::StringValue("Alice".to_string()),
        )];

        let created = tx
            .create_vertex(vertex_type.clone(), props.clone())
            .expect("create_vertex failed");
        assert_eq!(created.vertex_type, vertex_type);

        let fetched = tx
            .get_vertex(created.id.clone())
            .expect("get_vertex error")
            .expect("vertex not found");
        assert_eq!(fetched.id, created.id);
        assert_eq!(
            fetched.properties[0].1,
            PropertyValue::StringValue("Alice".to_string())
        );

        tx.delete_vertex(created.id, true)
            .expect("delete_vertex failed");
        tx.commit().unwrap();
    }

    #[test]
    fn test_create_and_delete_edge() {
        if env::var("ARANGO_HOST").is_err() {
            println!("Skipping test_create_and_delete_edge: ARANGO_HOST not set");
            return;
        }

        let tx = create_test_transaction();

        let v1 = tx.create_vertex("person".to_string(), vec![]).unwrap();
        let v2 = tx.create_vertex("person".to_string(), vec![]).unwrap();

        let edge = tx
            .create_edge("knows".to_string(), v1.id.clone(), v2.id.clone(), vec![])
            .expect("create_edge failed");
        assert_eq!(edge.edge_type, "knows");

        tx.delete_edge(edge.id.clone()).unwrap();
        let got = tx.get_edge(edge.id).unwrap();
        assert!(got.is_none());

        tx.delete_vertex(v1.id, true).unwrap();
        tx.delete_vertex(v2.id, true).unwrap();
        tx.commit().unwrap();
    }

    #[test]
    fn test_update_vertex_properties() {
        if env::var("ARANGO_HOST").is_err() {
            println!("Skipping test_update_vertex_properties: ARANGO_HOST not set");
            return;
        }

        let tx = create_test_transaction();
        let vt = "character".to_string();
        let init_props = vec![(
            "name".to_string(),
            PropertyValue::StringValue("Gandalf".to_string()),
        )];

        let created = tx.create_vertex(vt.clone(), init_props).unwrap();

        let updates = vec![(
            "name".to_string(),
            PropertyValue::StringValue("Gandalf the White".to_string()),
        )];
        let updated = tx
            .update_vertex_properties(created.id.clone(), updates)
            .expect("update_vertex_properties failed");

        let name = &updated
            .properties
            .iter()
            .find(|(k, _)| k == "name")
            .unwrap()
            .1;
        assert_eq!(
            name,
            &PropertyValue::StringValue("Gandalf the White".to_string())
        );

        // Cleanup
        tx.delete_vertex(created.id, true).unwrap();
        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_commit_and_rollback() {
        if env::var("ARANGO_HOST").is_err() {
            println!("Skipping test_transaction_commit_and_rollback: ARANGO_HOST not set");
            return;
        }

        let tx = create_test_transaction();
        assert!(tx.commit().is_ok());

        let tx2 = create_test_transaction();
        assert!(tx2.rollback().is_ok());
    }

    #[test]
    fn test_unsupported_upsert_operations() {
        if env::var("ARANGO_HOST").is_err() {
            println!("Skipping test_unsupported_upsert_operations: ARANGO_HOST not set");
            return;
        }

        let tx = create_test_transaction();
        let v = tx.create_vertex("person".to_string(), vec![]).unwrap();

        let u1 = tx.upsert_vertex(None, "person".to_string(), vec![]);
        assert!(matches!(u1, Err(GraphError::UnsupportedOperation(_))));

        let u2 = tx.upsert_edge(
            None,
            "knows".to_string(),
            v.id.clone(),
            v.id.clone(),
            vec![],
        );
        assert!(matches!(u2, Err(GraphError::UnsupportedOperation(_))));

        tx.commit().unwrap();
    }

    #[test]
    fn test_update_edge_properties_and_replace() {
        if env::var("ARANGO_HOST").is_err() {
            println!("Skipping test_update_edge_properties_and_replace");
            return;
        }
        let tx = create_test_transaction();

        let v1 = tx.create_vertex("person".to_string(), vec![]).unwrap();
        let v2 = tx.create_vertex("person".to_string(), vec![]).unwrap();

        let initial_props = vec![("weight".to_string(), PropertyValue::Float64(1.0))];
        let edge = tx
            .create_edge(
                "knows".to_string(),
                v1.id.clone(),
                v2.id.clone(),
                initial_props,
            )
            .unwrap();

        let merged = tx
            .update_edge_properties(
                edge.id.clone(),
                vec![("weight".to_string(), PropertyValue::Float64(2.0))],
            )
            .unwrap();
        assert_eq!(
            merged
                .properties
                .iter()
                .find(|(k, _)| k == "weight")
                .unwrap()
                .1,
            PropertyValue::Float64(2.0)
        );

        let replaced = tx
            .update_edge(
                edge.id.clone(),
                vec![(
                    "strength".to_string(),
                    PropertyValue::StringValue("high".to_string()),
                )],
            )
            .unwrap();
        assert_eq!(replaced.properties.len(), 1);
        assert_eq!(
            replaced.properties[0].1,
            PropertyValue::StringValue("high".to_string())
        );
        assert!(replaced.properties.iter().all(|(k, _)| k == "strength"));

        tx.delete_vertex(v1.id, true).unwrap();
        tx.delete_vertex(v2.id, true).unwrap();
        tx.commit().unwrap();
    }

    #[test]
    fn test_update_vertex_and_replace() {
        if env::var("ARANGO_HOST").is_err() {
            return;
        }
        let tx = create_test_transaction();

        let v = tx
            .create_vertex(
                "item".to_string(),
                vec![
                    ("a".to_string(), PropertyValue::StringValue("1".to_string())),
                    ("b".to_string(), PropertyValue::StringValue("2".to_string())),
                ],
            )
            .unwrap();

        let merged = tx
            .update_vertex_properties(
                v.id.clone(),
                vec![("b".to_string(), PropertyValue::StringValue("3".to_string()))],
            )
            .unwrap();
        assert_eq!(
            merged.properties.iter().find(|(k, _)| k == "b").unwrap().1,
            PropertyValue::StringValue("3".to_string())
        );
        assert!(merged.properties.iter().any(|(k, _)| k == "a"));

        let replaced = tx
            .update_vertex(
                v.id.clone(),
                vec![("c".to_string(), PropertyValue::Int64(42))],
            )
            .unwrap();
        assert_eq!(replaced.properties.len(), 1);
        assert_eq!(replaced.properties[0].1, PropertyValue::Int64(42));

        tx.delete_vertex(v.id, true).unwrap();
        tx.commit().unwrap();
    }

    #[test]
    fn test_find_vertices_and_edges() {
        if env::var("ARANGO_HOST").is_err() {
            return;
        }
        let tx = create_test_transaction();
        let v1 = tx
            .create_vertex(
                "person".to_string(),
                vec![(
                    "name".to_string(),
                    PropertyValue::StringValue("X".to_string()),
                )],
            )
            .unwrap();
        let v2 = tx
            .create_vertex(
                "person".to_string(),
                vec![(
                    "name".to_string(),
                    PropertyValue::StringValue("Y".to_string()),
                )],
            )
            .unwrap();

        let found: Vec<_> = tx
            .find_vertices(Some("person".to_string()), None, None, None, None)
            .unwrap();
        assert!(found.iter().any(|vx| vx.id == v1.id));
        assert!(found.iter().any(|vx| vx.id == v2.id));

        let e = tx
            .create_edge("likes".to_string(), v1.id.clone(), v2.id.clone(), vec![])
            .unwrap();
        let found_e = tx
            .find_edges(Some(vec!["likes".to_string()]), None, None, None, None)
            .unwrap();
        assert!(found_e.iter().any(|ed| ed.id == e.id));

        tx.delete_edge(e.id.clone()).unwrap();
        tx.delete_vertex(v1.id, true).unwrap();
        tx.delete_vertex(v2.id, true).unwrap();
        tx.commit().unwrap();
    }

    #[test]
    fn test_get_adjacent_and_connected() {
        if env::var("ARANGO_HOST").is_err() {
            return;
        }
        let tx = create_test_transaction();
        let v1 = tx.create_vertex("person".to_string(), vec![]).unwrap();
        let v2 = tx.create_vertex("person".to_string(), vec![]).unwrap();
        let v3 = tx.create_vertex("person".to_string(), vec![]).unwrap();
        // v1->v2 and v1->v3
        let _e1 = tx
            .create_edge("knows".to_string(), v1.id.clone(), v2.id.clone(), vec![])
            .unwrap();
        let _e2 = tx
            .create_edge("knows".to_string(), v1.id.clone(), v3.id.clone(), vec![])
            .unwrap();

        let out = tx
            .get_adjacent_vertices(
                v1.id.clone(),
                Direction::Outgoing,
                Some(vec!["knows".to_string()]),
                None,
            )
            .unwrap();
        assert_eq!(out.len(), 2);
        let inbound = tx
            .get_adjacent_vertices(
                v2.id.clone(),
                Direction::Incoming,
                Some(vec!["knows".to_string()]),
                None,
            )
            .unwrap();
        assert_eq!(inbound.len(), 1);
        assert_eq!(inbound[0].id, v1.id);

        let ces = tx
            .get_connected_edges(
                v1.id.clone(),
                Direction::Outgoing,
                Some(vec!["knows".to_string()]),
                None,
            )
            .unwrap();
        assert_eq!(ces.len(), 2);

        tx.delete_vertex(v1.id, true).unwrap();
        tx.delete_vertex(v2.id, true).unwrap();
        tx.delete_vertex(v3.id, true).unwrap();
        tx.commit().unwrap();
    }

    #[test]
    fn test_bulk_create_vertices_and_edges() {
        if env::var("ARANGO_HOST").is_err() {
            return;
        }
        let tx = create_test_transaction();
        let specs = vec![
            golem_graph::golem::graph::transactions::VertexSpec {
                vertex_type: "t".to_string(),
                additional_labels: None,
                properties: vec![("k".to_string(), PropertyValue::StringValue("v".to_string()))],
            };
            3
        ];
        let verts = tx.create_vertices(specs.clone()).unwrap();
        assert_eq!(verts.len(), 3);

        // Bulk edges between 0->1,1->2
        let specs_e = vec![
            golem_graph::golem::graph::transactions::EdgeSpec {
                edge_type: "rel".to_string(),
                from_vertex: verts[0].id.clone(),
                to_vertex: verts[1].id.clone(),
                properties: vec![],
            },
            golem_graph::golem::graph::transactions::EdgeSpec {
                edge_type: "rel".to_string(),
                from_vertex: verts[1].id.clone(),
                to_vertex: verts[2].id.clone(),
                properties: vec![],
            },
        ];
        let edges = tx.create_edges(specs_e.clone()).unwrap();
        assert_eq!(edges.len(), 2);

        for e in edges {
            tx.delete_edge(e.id).unwrap();
        }
        for v in verts {
            tx.delete_vertex(v.id, true).unwrap();
        }
        tx.commit().unwrap();
    }
}
