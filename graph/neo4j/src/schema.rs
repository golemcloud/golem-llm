use crate::helpers::{config_from_env, map_neo4j_type_to_wit};
use crate::{GraphNeo4jComponent, SchemaManager};
use golem_graph::durability::ExtendedGuest;
use golem_graph::golem::graph::{
    errors::GraphError,
    schema::{
        Guest as SchemaGuest, GuestSchemaManager, IndexDefinition, IndexType, PropertyDefinition,
        PropertyType, SchemaManager as SchemaManagerResource, VertexLabelSchema,
    },
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

impl SchemaGuest for GraphNeo4jComponent {
    type SchemaManager = SchemaManager;

    fn get_schema_manager() -> Result<SchemaManagerResource, GraphError> {
        let config = config_from_env()?;
        let graph = GraphNeo4jComponent::connect_internal(&config)?;
        let manager = SchemaManager {
            graph: Arc::new(graph),
        };
        Ok(SchemaManagerResource::new(manager))
    }
}

impl GuestSchemaManager for SchemaManager {
    fn define_vertex_label(
        &self,
        schema: golem_graph::golem::graph::schema::VertexLabelSchema,
    ) -> Result<(), GraphError> {
        for prop in schema.properties {
            if prop.required {
                let q = format!(
                    "CREATE CONSTRAINT constraint_required_{label}_{name} \
                     IF NOT EXISTS FOR (n:{label}) REQUIRE n.{name} IS NOT NULL",
                    label = schema.label,
                    name = prop.name
                );
                let tx = self.graph.begin_transaction()?;
                // run and swallow the EE‐only error
                match tx.api.execute_in_transaction(
                    &tx.transaction_url,
                    json!({ "statements": [ { "statement": q } ] }),
                ) {
                    Err(e) => {
                        let msg = e.to_string();
                        if msg.contains("Enterprise Edition")
                            || msg.contains("ConstraintCreationFailed")
                        {
                            println!("[WARN] Skipping property existence constraint: requires Neo4j Enterprise Edition. Error: {}", msg);
                            tx.commit()?;
                        } else {
                            return Err(e);
                        }
                    }
                    Ok(_) => tx.commit()?,
                }
            }

            if prop.unique {
                let q = format!(
                    "CREATE CONSTRAINT constraint_unique_{label}_{name} \
                     IF NOT EXISTS FOR (n:{label}) REQUIRE n.{name} IS UNIQUE",
                    label = schema.label,
                    name = prop.name
                );
                let tx = self.graph.begin_transaction()?;
                // unique constraints work on CE
                tx.api.execute_in_transaction(
                    &tx.transaction_url,
                    json!({ "statements": [ { "statement": q } ] }),
                )?;
                tx.commit()?;
            }
        }

        Ok(())
    }

    fn define_edge_label(
        &self,
        schema: golem_graph::golem::graph::schema::EdgeLabelSchema,
    ) -> Result<(), GraphError> {
        let tx = self.graph.begin_transaction()?;
        let mut statements = Vec::new();

        for prop in schema.properties {
            if prop.required {
                let constraint_name =
                    format!("constraint_rel_required_{}_{}", &schema.label, &prop.name);
                let query = format!(
                    "CREATE CONSTRAINT {} IF NOT EXISTS FOR ()-[r:{}]-() REQUIRE r.{} IS NOT NULL",
                    constraint_name, &schema.label, &prop.name
                );
                statements.push(json!({ "statement": query, "parameters": {} }));
            }
            if prop.unique {
                // Neo4j does not support uniqueness constraints on relationship properties.
                // We will silently ignore this for now.
            }
        }

        if statements.is_empty() {
            return tx.commit();
        }

        let statements_payload = json!({ "statements": statements });
        tx.api
            .execute_in_transaction(&tx.transaction_url, statements_payload)?;

        tx.commit()
    }

    fn get_vertex_label_schema(
        &self,
        label: String,
    ) -> Result<Option<VertexLabelSchema>, GraphError> {
        let tx = self.graph.begin_transaction()?;

        // Fetch node‐property metadata
        let props_query =
            "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes, mandatory \
             WHERE $label IN nodeLabels \
             RETURN propertyName, propertyTypes, mandatory";
        let props_stmt = json!({
            "statement": props_query,
            "parameters": { "label": &label }
        });
        let props_resp = tx
            .api
            .execute_in_transaction(&tx.transaction_url, json!({ "statements": [props_stmt] }))?;

        // Fetch uniqueness constraints
        let cons_query = "SHOW CONSTRAINTS YIELD name, type, properties, labelsOrTypes \
             WHERE type = 'UNIQUENESS' AND $label IN labelsOrTypes \
             RETURN properties";
        let cons_stmt = json!({
            "statement": cons_query,
            "parameters": { "label": &label }
        });
        let cons_resp = tx
            .api
            .execute_in_transaction(&tx.transaction_url, json!({ "statements": [cons_stmt] }))?;

        tx.commit()?;

        // Parse properties
        let props_block = props_resp["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| GraphError::InternalError("Invalid property schema response".into()))?;
        let props_data = props_block["data"]
            .as_array()
            .ok_or_else(|| GraphError::InternalError("Missing property schema data".into()))?;

        if props_data.is_empty() {
            return Ok(None);
        }

        #[derive(serde::Deserialize)]
        struct Info {
            property_name: String,
            property_types: Vec<String>,
            mandatory: bool,
        }

        let mut defs: HashMap<String, PropertyDefinition> = HashMap::new();
        for row_item in props_data {
            if let Some(row_val) = row_item.get("row") {
                if let Ok(row) = serde_json::from_value::<Vec<Value>>(row_val.clone()) {
                    if row.len() >= 3 {
                        let info = Info {
                            property_name: row[0].as_str().unwrap_or("").to_string(),
                            property_types: serde_json::from_value(row[1].clone())
                                .unwrap_or_default(),
                            mandatory: row[2].as_bool().unwrap_or(false),
                        };
                        defs.insert(
                            info.property_name.clone(),
                            PropertyDefinition {
                                name: info.property_name.clone(),
                                property_type: info
                                    .property_types
                                    .first()
                                    .map(|s| map_neo4j_type_to_wit(s))
                                    .unwrap_or(PropertyType::StringType),
                                required: info.mandatory,
                                unique: false, // will flip next
                                default_value: None,
                            },
                        );
                    }
                }
            }
        }

        // Parse uniqueness constraints
        let cons_block = cons_resp["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| {
                GraphError::InternalError("Invalid constraint schema response".into())
            })?;
        let cons_data = cons_block["data"]
            .as_array()
            .ok_or_else(|| GraphError::InternalError("Missing constraint data".into()))?;

        for row_item in cons_data {
            if let Some(row_val) = row_item.get("row") {
                if let Ok(row) = serde_json::from_value::<Vec<Value>>(row_val.clone()) {
                    if let Some(list_val) = row.first() {
                        if let Ok(list) = serde_json::from_value::<Vec<String>>(list_val.clone()) {
                            for prop_name in list {
                                if let Some(d) = defs.get_mut(&prop_name) {
                                    d.unique = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Ensure any unique property is also required
        for def in defs.values_mut() {
            if def.unique {
                def.required = true;
            }
        }

        let props = defs.into_values().collect();
        Ok(Some(VertexLabelSchema {
            label,
            properties: props,
            container: None,
        }))
    }

    fn get_edge_label_schema(
        &self,
        label: String,
    ) -> Result<Option<golem_graph::golem::graph::schema::EdgeLabelSchema>, GraphError> {
        let tx = self.graph.begin_transaction()?;

        let props_query = "CALL db.schema.relTypeProperties() YIELD relType, propertyName, propertyTypes, mandatory WHERE relType = $label RETURN propertyName, propertyTypes, mandatory";
        let props_statement = json!({
            "statement": props_query,
            "parameters": { "label": &label }
        });
        let props_response = tx.api.execute_in_transaction(
            &tx.transaction_url,
            json!({ "statements": [props_statement] }),
        )?;

        tx.commit()?;

        let props_result = props_response["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| {
                GraphError::InternalError("Invalid property schema response for edge".to_string())
            })?;
        let props_data = props_result["data"].as_array().ok_or_else(|| {
            GraphError::InternalError("Missing property schema data for edge".to_string())
        })?;

        if props_data.is_empty() {
            return Ok(None);
        }

        #[derive(serde::Deserialize)]
        struct Neo4jPropertyInfo {
            property_name: String,
            property_types: Vec<String>,
            mandatory: bool,
        }

        let mut property_definitions = Vec::new();
        for item in props_data {
            if let Some(row_val) = item.get("row") {
                if let Ok(row) = serde_json::from_value::<Vec<Value>>(row_val.clone()) {
                    if row.len() >= 3 {
                        let info = Neo4jPropertyInfo {
                            property_name: row[0].as_str().unwrap_or("").to_string(),
                            property_types: serde_json::from_value(row[1].clone())
                                .unwrap_or_default(),
                            mandatory: row[2].as_bool().unwrap_or(false),
                        };

                        if !info.property_name.is_empty() {
                            property_definitions.push(PropertyDefinition {
                                name: info.property_name,
                                property_type: info
                                    .property_types
                                    .first()
                                    .map(|s| map_neo4j_type_to_wit(s))
                                    .unwrap_or(PropertyType::StringType),
                                required: info.mandatory,
                                unique: false, // Not supported for relationships in Neo4j
                                default_value: None,
                            });
                        }
                    }
                }
            }
        }

        Ok(Some(golem_graph::golem::graph::schema::EdgeLabelSchema {
            label,
            properties: property_definitions,
            from_labels: None, // Neo4j does not enforce this at the schema level
            to_labels: None,   // Neo4j does not enforce this at the schema level
            container: None,
        }))
    }

    fn list_vertex_labels(&self) -> Result<Vec<String>, GraphError> {
        let tx = self.graph.begin_transaction()?;
        let result = tx.execute_schema_query_and_extract_string_list(
            "CALL db.labels() YIELD label RETURN label",
        );
        tx.commit()?;
        result
    }

    fn list_edge_labels(&self) -> Result<Vec<String>, GraphError> {
        let tx = self.graph.begin_transaction()?;
        let result = tx.execute_schema_query_and_extract_string_list(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType",
        );
        tx.commit()?;
        result
    }

    fn create_index(
        &self,
        index: golem_graph::golem::graph::schema::IndexDefinition,
    ) -> Result<(), GraphError> {
        let tx = self.graph.begin_transaction()?;

        let index_type_str = match index.index_type {
            IndexType::Range => "RANGE",
            IndexType::Text => "TEXT",
            IndexType::Geospatial => "POINT",
            IndexType::Exact => {
                return Err(GraphError::UnsupportedOperation(
                    "Neo4j does not have a separate 'Exact' index type; use RANGE or TEXT."
                        .to_string(),
                ))
            }
        };

        let properties_str = index.properties.join(", ");

        let query = format!(
            "CREATE {} INDEX {} IF NOT EXISTS FOR (n:{}) ON (n.{})",
            index_type_str, index.name, index.label, properties_str
        );

        let statement = json!({ "statement": query, "parameters": {} });
        let statements = json!({ "statements": [statement] });
        tx.api
            .execute_in_transaction(&tx.transaction_url, statements)?;
        tx.commit()
    }

    fn drop_index(&self, name: String) -> Result<(), GraphError> {
        let tx = self.graph.begin_transaction()?;
        let query = format!("DROP INDEX {} IF EXISTS", name);
        let statement = json!({ "statement": query, "parameters": {} });
        let statements = json!({ "statements": [statement] });
        tx.api
            .execute_in_transaction(&tx.transaction_url, statements)?;
        tx.commit()
    }

    fn list_indexes(
        &self,
    ) -> Result<Vec<golem_graph::golem::graph::schema::IndexDefinition>, GraphError> {
        let tx = self.graph.begin_transaction()?;
        let query = "SHOW INDEXES";
        let statement = json!({ "statement": query, "parameters": {} });
        let statements = json!({ "statements": [statement] });
        let response = tx
            .api
            .execute_in_transaction(&tx.transaction_url, statements)?;

        tx.commit()?;

        let result = response["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| {
                GraphError::InternalError("Invalid response for list_indexes".to_string())
            })?;

        if let Some(errors) = result["errors"].as_array() {
            if !errors.is_empty() {
                return Err(GraphError::InvalidQuery(errors[0].to_string()));
            }
        }

        let empty_vec = vec![];
        let data = result["data"].as_array().unwrap_or(&empty_vec);
        let mut indexes = Vec::new();

        for item in data {
            if let Some(row) = item["row"].as_array() {
                if row.len() >= 8 {
                    let name = row[1].as_str().unwrap_or_default().to_string();
                    let index_type_str = row[4].as_str().unwrap_or_default().to_lowercase();
                    let label = row[6]
                        .as_array()
                        .and_then(|arr| arr.first())
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let properties: Vec<String> = row[7]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .map(|v| v.as_str().unwrap_or_default().to_string())
                                .collect()
                        })
                        .unwrap_or_default();
                    let unique = row[9].is_string();

                    let index_type = match index_type_str.as_str() {
                        "range" => IndexType::Range,
                        "text" => IndexType::Text,
                        "point" => IndexType::Geospatial,
                        _ => continue,
                    };

                    indexes.push(IndexDefinition {
                        name,
                        label,
                        properties,
                        index_type,
                        unique,
                        container: None,
                    });
                }
            }
        }
        Ok(indexes)
    }

    fn get_index(
        &self,
        _name: String,
    ) -> Result<Option<golem_graph::golem::graph::schema::IndexDefinition>, GraphError> {
        Err(GraphError::UnsupportedOperation(
            "get_index is not supported by the Neo4j provider yet.".to_string(),
        ))
    }

    fn define_edge_type(
        &self,
        _definition: golem_graph::golem::graph::schema::EdgeTypeDefinition,
    ) -> Result<(), GraphError> {
        Err(GraphError::UnsupportedOperation(
            "define_edge_type is not supported by the Neo4j provider".to_string(),
        ))
    }

    fn list_edge_types(
        &self,
    ) -> Result<Vec<golem_graph::golem::graph::schema::EdgeTypeDefinition>, GraphError> {
        Err(GraphError::UnsupportedOperation(
            "list_edge_types is not supported by the Neo4j provider".to_string(),
        ))
    }

    fn create_container(
        &self,
        _name: String,
        _container_type: golem_graph::golem::graph::schema::ContainerType,
    ) -> Result<(), GraphError> {
        Err(GraphError::UnsupportedOperation(
            "create_container is not supported by the Neo4j provider".to_string(),
        ))
    }

    fn list_containers(
        &self,
    ) -> Result<Vec<golem_graph::golem::graph::schema::ContainerInfo>, GraphError> {
        Ok(vec![])
    }
}
