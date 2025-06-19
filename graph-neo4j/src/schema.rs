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
        let tx = self.graph.begin_transaction()?;
        let mut statements = Vec::new();

        for prop in schema.properties {
            if prop.required {
                let constraint_name =
                    format!("constraint_required_{}_{}", &schema.label, &prop.name);
                let query = format!(
                    "CREATE CONSTRAINT {} IF NOT EXISTS FOR (n:{}) REQUIRE n.{} IS NOT NULL",
                    constraint_name, &schema.label, &prop.name
                );
                statements.push(json!({ "statement": query, "parameters": {} }));
            }
            if prop.unique {
                let constraint_name = format!("constraint_unique_{}_{}", &schema.label, &prop.name);
                let query = format!(
                    "CREATE CONSTRAINT {} IF NOT EXISTS FOR (n:{}) REQUIRE n.{} IS UNIQUE",
                    constraint_name, &schema.label, &prop.name
                );
                statements.push(json!({ "statement": query, "parameters": {} }));
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

        let props_query = "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes, mandatory WHERE $label IN nodeLabels RETURN propertyName, propertyTypes, mandatory";
        let props_statement = json!({
            "statement": props_query,
            "parameters": { "label": &label }
        });
        let props_response = tx.api.execute_in_transaction(
            &tx.transaction_url,
            json!({ "statements": [props_statement] }),
        )?;

        let constraints_query = "SHOW CONSTRAINTS YIELD name, type, properties, labelsOrTypes WHERE type = 'UNIQUENESS' AND $label IN labelsOrTypes RETURN properties";
        let constraints_statement = json!({
            "statement": constraints_query,
            "parameters": { "label": &label }
        });
        let constraints_response = tx.api.execute_in_transaction(
            &tx.transaction_url,
            json!({ "statements": [constraints_statement] }),
        )?;

        tx.commit()?;

        let props_result = props_response["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| {
                GraphError::InternalError("Invalid property schema response".to_string())
            })?;
        let props_data = props_result["data"]
            .as_array()
            .ok_or_else(|| GraphError::InternalError("Missing property schema data".to_string()))?;

        if props_data.is_empty() {
            return Ok(None);
        }

        #[derive(serde::Deserialize)]
        struct Neo4jPropertyInfo {
            property_name: String,
            property_types: Vec<String>,
            mandatory: bool,
        }

        let mut property_definitions: HashMap<String, PropertyDefinition> = HashMap::new();
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
                            property_definitions.insert(
                                info.property_name.clone(),
                                PropertyDefinition {
                                    name: info.property_name,
                                    property_type: info
                                        .property_types
                                        .first()
                                        .map(|s| map_neo4j_type_to_wit(s))
                                        .unwrap_or(PropertyType::StringType),
                                    required: info.mandatory,
                                    unique: false, // will set this in the next step
                                    default_value: None,
                                },
                            );
                        }
                    }
                }
            }
        }

        let constraints_result = constraints_response["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| {
                GraphError::InternalError("Invalid constraint schema response".to_string())
            })?;
        let constraints_data = constraints_result["data"].as_array().ok_or_else(|| {
            GraphError::InternalError("Missing constraint schema data".to_string())
        })?;

        for item in constraints_data {
            if let Some(row_val) = item.get("row") {
                if let Ok(row) = serde_json::from_value::<Vec<Value>>(row_val.clone()) {
                    if let Some(prop_list_val) = row.first() {
                        if let Ok(prop_list) =
                            serde_json::from_value::<Vec<String>>(prop_list_val.clone())
                        {
                            for prop_name in prop_list {
                                if let Some(prop_def) = property_definitions.get_mut(&prop_name) {
                                    prop_def.unique = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(Some(VertexLabelSchema {
            label,
            properties: property_definitions.into_values().collect(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use golem_graph::golem::graph::{
        connection::ConnectionConfig,
        schema::{IndexDefinition, IndexType, PropertyDefinition, PropertyType, VertexLabelSchema},
    };
    use std::env;

    fn create_test_schema_manager() -> SchemaManager {
        let host = env::var("NEO4J_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port = env::var("NEO4J_PORT")
            .unwrap_or_else(|_| "7474".to_string())
            .parse()
            .unwrap();
        let user = env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string());
        let password = env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".to_string());
        let database = env::var("NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".to_string());

        let config = ConnectionConfig {
            hosts: vec![host],
            port: Some(port),
            username: Some(user),
            password: Some(password),
            database_name: Some(database),
            timeout_seconds: None,
            max_connections: None,
            provider_config: vec![],
        };

        let graph = GraphNeo4jComponent::connect_internal(&config).unwrap();
        SchemaManager {
            graph: Arc::new(graph),
        }
    }

    #[test]
    fn test_create_and_drop_index() {
        if env::var("NEO4J_HOST").is_err() {
            println!("Skipping test_create_and_drop_index: NEO4J_HOST not set");
            return;
        }

        let manager = create_test_schema_manager();
        let index_name = "test_index_for_person_name".to_string();
        let index_def = IndexDefinition {
            name: index_name.clone(),
            label: "Person".to_string(),
            properties: vec!["name".to_string()],
            index_type: IndexType::Range,
            unique: false,
            container: None,
        };

        manager.create_index(index_def.clone()).unwrap();

        let indexes = manager.list_indexes().unwrap();
        assert!(indexes.iter().any(|i| i.name == index_name));

        manager.drop_index(index_name.clone()).unwrap();

        let indexes_after_drop = manager.list_indexes().unwrap();
        assert!(!indexes_after_drop.iter().any(|i| i.name == index_name));
    }

    #[test]
    fn test_define_and_get_vertex_label() {
        if env::var("NEO4J_HOST").is_err() {
            println!("Skipping test_define_and_get_vertex_label: NEO4J_HOST not set");
            return;
        }

        let manager = create_test_schema_manager();
        let label = "TestLabel".to_string();
        let schema = VertexLabelSchema {
            label: label.clone(),
            properties: vec![
                PropertyDefinition {
                    name: "id".to_string(),
                    property_type: PropertyType::StringType,
                    required: true,
                    unique: true,
                    default_value: None,
                },
                PropertyDefinition {
                    name: "score".to_string(),
                    property_type: PropertyType::Float64,
                    required: false,
                    unique: false,
                    default_value: None,
                },
            ],
            container: None,
        };

        manager.define_vertex_label(schema).unwrap();

        let retrieved_schema = manager
            .get_vertex_label_schema(label.clone())
            .unwrap()
            .unwrap();
        assert_eq!(retrieved_schema.label, label);
        assert_eq!(retrieved_schema.properties.len(), 2);

        let id_prop = retrieved_schema
            .properties
            .iter()
            .find(|p| p.name == "id")
            .unwrap();
        assert!(id_prop.required);
        assert!(id_prop.unique);

        let tx = manager.graph.begin_transaction().unwrap();
        let drop_required_query = format!("DROP CONSTRAINT constraint_required_{}_id", label);
        let drop_unique_query = format!("DROP CONSTRAINT constraint_unique_{}_id", label);
        tx.api
            .execute_in_transaction(
                &tx.transaction_url,
                json!({ "statements": [
                    { "statement": drop_required_query },
                    { "statement": drop_unique_query }
                ]}),
            )
            .unwrap();
        tx.commit().unwrap();
    }

    #[test]
    fn test_unsupported_get_index() {
        if env::var("NEO4J_HOST").is_err() {
            println!("Skipping test_unsupported_get_index: NEO4J_HOST not set");
            return;
        }

        let manager = create_test_schema_manager();
        let result = manager.get_index("any_index".to_string());
        assert!(matches!(result, Err(GraphError::UnsupportedOperation(_))));
    }
}
