use crate::{helpers, GraphJanusGraphComponent, SchemaManager};
use golem_graph::durability::ExtendedGuest;
use golem_graph::golem::graph::{
    errors::GraphError,
    schema::{
        ContainerInfo, EdgeLabelSchema, EdgeTypeDefinition, Guest as SchemaGuest,
        GuestSchemaManager, IndexDefinition, IndexType, SchemaManager as SchemaManagerResource,
        VertexLabelSchema,
    },
};
use serde_json::Value;
use std::sync::Arc;

impl SchemaGuest for GraphJanusGraphComponent {
    type SchemaManager = SchemaManager;

    fn get_schema_manager() -> Result<SchemaManagerResource, GraphError> {
        let config = helpers::config_from_env()?;
        let graph = crate::GraphJanusGraphComponent::connect_internal(&config)?;
        let manager = SchemaManager {
            graph: Arc::new(graph),
        };
        Ok(SchemaManagerResource::new(manager))
    }
}

impl GuestSchemaManager for SchemaManager {
    fn define_vertex_label(&self, schema: VertexLabelSchema) -> Result<(), GraphError> {
        let mut script = String::new();

        for prop in &schema.properties {
            let prop_type_class = SchemaManager::map_wit_type_to_janus_class(&prop.property_type);
            script.push_str(&format!(
                "if (mgmt.getPropertyKey('{}') == null) {{ mgmt.makePropertyKey('{}').dataType({}).make() }};",
                prop.name, prop.name, prop_type_class
            ));
        }

        script.push_str(&format!(
            "if (mgmt.getVertexLabel('{}') == null) {{ mgmt.makeVertexLabel('{}').make() }};",
            schema.label, schema.label
        ));

        self.execute_management_query(&script)?;
        Ok(())
    }

    fn define_edge_label(&self, schema: EdgeLabelSchema) -> Result<(), GraphError> {
        let mut script = String::new();

        for prop in &schema.properties {
            let prop_type_class = SchemaManager::map_wit_type_to_janus_class(&prop.property_type);
            script.push_str(&format!(
                "if (mgmt.getPropertyKey('{}') == null) {{ mgmt.makePropertyKey('{}').dataType({}).make() }};",
                prop.name, prop.name, prop_type_class
            ));
        }

        script.push_str(&format!(
            "if (mgmt.getEdgeLabel('{}') == null) {{ mgmt.makeEdgeLabel('{}').make() }};",
            schema.label, schema.label
        ));

        self.execute_management_query(&script)?;
        Ok(())
    }

    fn get_vertex_label_schema(
        &self,
        label: String,
    ) -> Result<Option<VertexLabelSchema>, GraphError> {
        let script = format!("mgmt.getVertexLabel('{}') != null", label);
        let result = self.execute_management_query(&script)?;
        let exists = result
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if exists {
            Ok(Some(VertexLabelSchema {
                label,
                properties: vec![],
                container: None,
            }))
        } else {
            Ok(None)
        }
    }

    fn get_edge_label_schema(&self, label: String) -> Result<Option<EdgeLabelSchema>, GraphError> {
        let script = format!("mgmt.getEdgeLabel('{}') != null", label);
        let result = self.execute_management_query(&script)?;
        let exists = result
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if exists {
            Ok(Some(EdgeLabelSchema {
                label,
                properties: vec![],
                from_labels: None,
                to_labels: None,
                container: None,
            }))
        } else {
            Ok(None)
        }
    }

    fn list_vertex_labels(&self) -> Result<Vec<String>, GraphError> {
        let script = "mgmt.getVertexLabels().collect{ it.name() }";
        let result = self.execute_management_query(script)?;
        self.parse_string_list_from_result(result)
    }

    fn list_edge_labels(&self) -> Result<Vec<String>, GraphError> {
        let script = "mgmt.getEdgeLabels().collect{ it.name() }";
        let result = self.execute_management_query(script)?;
        self.parse_string_list_from_result(result)
    }

    fn create_index(&self, index: IndexDefinition) -> Result<(), GraphError> {
        let mut script_parts = Vec::new();

        for prop_name in &index.properties {
            script_parts.push(format!(
                "if (mgmt.getPropertyKey('{}') == null) throw new IllegalArgumentException('Property key {} not found');",
                prop_name, prop_name
            ));
        }

        let container_name = index.container.as_deref().unwrap_or_default();

        script_parts.push(format!(
            "def label = mgmt.getVertexLabel('{}'); def elementClass = Vertex.class;",
            container_name
        ));
        script_parts.push(format!(
            "if (label == null) {{ label = mgmt.getEdgeLabel('{}'); elementClass = Edge.class; }}",
            container_name
        ));
        script_parts.push(format!(
            "if (label == null) throw new IllegalArgumentException('Label {} not found');",
            container_name
        ));

        let mut index_builder = format!("mgmt.buildIndex('{}', elementClass)", index.name);
        for prop_name in &index.properties {
            index_builder.push_str(&format!(".addKey(mgmt.getPropertyKey('{}'))", prop_name));
        }

        if index.unique {
            index_builder.push_str(".unique()");
        }

        index_builder.push_str(".indexOnly(label).buildCompositeIndex()");
        script_parts.push(index_builder);

        let script = script_parts.join(" ");
        self.execute_management_query(&script)?;

        Ok(())
    }

    fn drop_index(&self, name: String) -> Result<(), GraphError> {
        // Dropping an index in JanusGraph is a multi-step async process.
        // A simple synchronous version is not readily available.
        // We can, however, disable it. For now, we return unsupported.
        let _ = name;
        Err(GraphError::UnsupportedOperation(
            "Dropping an index is not supported in this version.".to_string(),
        ))
    }

    fn list_indexes(&self) -> Result<Vec<IndexDefinition>, GraphError> {
        let script = "
            def results = [];
            mgmt.getGraphIndexes(Vertex.class).each { index ->
                def backingIndex = index.getBackingIndex();
                def properties = index.getFieldKeys().collect{ it.name() };
                results.add([
                    'name': index.name(),
                    'unique': index.isUnique(),
                    'label': backingIndex.split(':')[0],
                    'properties': properties
                ]);
            };
            mgmt.getGraphIndexes(Edge.class).each { index ->
                def backingIndex = index.getBackingIndex();
                def properties = index.getFieldKeys().collect{ it.name() };
                results.add([
                    'name': index.name(),
                    'unique': index.isUnique(),
                    'label': backingIndex.split(':')[0],
                    'properties': properties
                ]);
            };
            results
        ";

        let result = self.execute_management_query(script)?;
        self.parse_index_list_from_result(result)
    }

    fn get_index(&self, name: String) -> Result<Option<IndexDefinition>, GraphError> {
        let indexes = self.list_indexes()?;
        Ok(indexes.into_iter().find(|i| i.name == name))
    }

    fn define_edge_type(&self, definition: EdgeTypeDefinition) -> Result<(), GraphError> {
        let mut script_parts = Vec::new();
        for from_label in &definition.from_collections {
            for to_label in &definition.to_collections {
                script_parts.push(format!(
                    "
                    def edgeLabel = mgmt.getEdgeLabel('{}');
                    def fromLabel = mgmt.getVertexLabel('{}');
                    def toLabel = mgmt.getVertexLabel('{}');
                    if (edgeLabel != null && fromLabel != null && toLabel != null) {{
                        mgmt.addConnection(edgeLabel, fromLabel, toLabel);
                    }}
                    ",
                    definition.collection, from_label, to_label
                ));
            }
        }

        self.execute_management_query(&script_parts.join("\n"))?;
        Ok(())
    }

    fn list_edge_types(&self) -> Result<Vec<EdgeTypeDefinition>, GraphError> {
        Err(GraphError::UnsupportedOperation(
            "Schema management is not supported in this version.".to_string(),
        ))
    }

    fn create_container(
        &self,
        _name: String,
        _container_type: golem_graph::golem::graph::schema::ContainerType,
    ) -> Result<(), GraphError> {
        Err(GraphError::UnsupportedOperation(
            "Schema management is not supported in this version.".to_string(),
        ))
    }

    fn list_containers(&self) -> Result<Vec<ContainerInfo>, GraphError> {
        Err(GraphError::UnsupportedOperation(
            "Schema management is not supported in this version.".to_string(),
        ))
    }
}

impl SchemaManager {
    fn execute_management_query(&self, script: &str) -> Result<Value, GraphError> {
        let full_script = format!(
            "mgmt = graph.openManagement(); def result = {{ {} }}.call(); mgmt.commit(); result",
            script
        );
        let response = self.graph.api.execute(&full_script, None)?;
        Ok(response["result"]["data"].clone())
    }

    fn parse_string_list_from_result(&self, result: Value) -> Result<Vec<String>, GraphError> {
        result
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|inner| inner.as_array())
            .ok_or_else(|| {
                GraphError::InternalError("Failed to parse string list from Gremlin".to_string())
            })?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(String::from)
                    .ok_or_else(|| GraphError::InternalError("Expected string in list".to_string()))
            })
            .collect()
    }

    fn parse_index_list_from_result(
        &self,
        result: Value,
    ) -> Result<Vec<IndexDefinition>, GraphError> {
        let mut indexes = Vec::new();
        if let Some(arr) = result.as_array() {
            for item in arr {
                if let Some(map) = item.as_object() {
                    let name = map
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let unique = map
                        .get("unique")
                        .and_then(|v| v.as_bool())
                        .unwrap_or_default();
                    let label = map
                        .get("label")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let properties = map
                        .get("properties")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default();

                    indexes.push(IndexDefinition {
                        name,
                        label: label.clone(),
                        container: Some(label),
                        properties,
                        unique,
                        index_type: IndexType::Exact,
                    });
                }
            }
        }
        Ok(indexes)
    }

    fn map_wit_type_to_janus_class(
        prop_type: &golem_graph::golem::graph::schema::PropertyType,
    ) -> &'static str {
        use golem_graph::golem::graph::schema::PropertyType;
        match prop_type {
            PropertyType::StringType => "String.class",
            PropertyType::Int64 => "Long.class",
            PropertyType::Float64 => "Double.class",
            PropertyType::Boolean => "Boolean.class",
            PropertyType::Datetime => "Date.class",
            _ => "Object.class",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::JanusGraphApi;
    use golem_graph::golem::graph::schema::{
        EdgeLabelSchema, GuestSchemaManager, IndexDefinition, IndexType, PropertyDefinition,
        PropertyType, VertexLabelSchema,
    };
    use std::env;
    use uuid::Uuid;

    fn create_test_schema_manager() -> SchemaManager {
        let host = env::var("JANUSGRAPH_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port = env::var("JANUSGRAPH_PORT")
            .unwrap_or_else(|_| "8182".to_string())
            .parse()
            .unwrap();

        let api = JanusGraphApi::new(&host, port, None, None).unwrap();
        let graph = crate::Graph { api: Arc::new(api) };
        SchemaManager {
            graph: Arc::new(graph),
        }
    }

    #[test]
    fn test_define_and_get_vertex_label() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_define_and_get_vertex_label: JANUSGRAPH_HOST not set");
            return;
        }

        let manager = create_test_schema_manager();
        let label_name = "test_vertex_label_".to_string() + &Uuid::new_v4().to_string();
        let schema = VertexLabelSchema {
            label: label_name.clone(),
            properties: vec![PropertyDefinition {
                name: "test_prop".to_string(),
                property_type: PropertyType::StringType,
                required: false,
                unique: false,
                default_value: None,
            }],
            container: None,
        };

        manager.define_vertex_label(schema).unwrap();
        let fetched_schema = manager.get_vertex_label_schema(label_name).unwrap();
        assert!(fetched_schema.is_some());
    }

    #[test]
    fn test_define_and_get_edge_label() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_define_and_get_edge_label: JANUSGRAPH_HOST not set");
            return;
        }

        let manager = create_test_schema_manager();
        let label_name = "test_edge_label_".to_string() + &Uuid::new_v4().to_string();
        let schema = EdgeLabelSchema {
            label: label_name.clone(),
            properties: vec![PropertyDefinition {
                name: "edge_prop".to_string(),
                property_type: PropertyType::StringType,
                required: false,
                unique: false,
                default_value: None,
            }],
            from_labels: None,
            to_labels: None,
            container: None,
        };

        manager.define_edge_label(schema).unwrap();
        let fetched_schema = manager.get_edge_label_schema(label_name).unwrap();
        assert!(fetched_schema.is_some());
    }

    #[test]
    fn test_create_and_list_vertex_index() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_create_and_list_vertex_index: JANUSGRAPH_HOST not set");
            return;
        }

        let manager = create_test_schema_manager();
        let vertex_label = "indexed_vertex_".to_string() + &Uuid::new_v4().to_string();
        let prop_name = "indexed_prop".to_string();
        let index_name = "v_index_".to_string() + &Uuid::new_v4().to_string();

        let vertex_schema = VertexLabelSchema {
            label: vertex_label.clone(),
            properties: vec![PropertyDefinition {
                name: prop_name.clone(),
                property_type: PropertyType::StringType,
                required: false,
                unique: false,
                default_value: None,
            }],
            container: None,
        };
        manager.define_vertex_label(vertex_schema).unwrap();

        let index_def = IndexDefinition {
            name: index_name.clone(),
            label: vertex_label.clone(),
            container: Some(vertex_label),
            properties: vec![prop_name],
            unique: false,
            index_type: IndexType::Exact,
        };
        manager.create_index(index_def).unwrap();

        // It can take some time for the index to be available in JanusGraph
        std::thread::sleep(std::time::Duration::from_secs(2));

        let indexes = manager.list_indexes().unwrap();
        assert!(
            indexes.iter().any(|i| i.name == index_name),
            "Index not found"
        );
    }

    #[test]
    fn test_list_labels() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_list_labels: JANUSGRAPH_HOST not set");
            return;
        }

        let manager = create_test_schema_manager();
        let vertex_label = "list_v_label_".to_string() + &Uuid::new_v4().to_string();
        let edge_label = "list_e_label_".to_string() + &Uuid::new_v4().to_string();

        manager
            .define_vertex_label(VertexLabelSchema {
                label: vertex_label.clone(),
                properties: vec![],
                container: None,
            })
            .unwrap();
        manager
            .define_edge_label(EdgeLabelSchema {
                label: edge_label.clone(),
                properties: vec![],
                from_labels: None,
                to_labels: None,
                container: None,
            })
            .unwrap();

        let vertex_labels = manager.list_vertex_labels().unwrap();
        let edge_labels = manager.list_edge_labels().unwrap();

        assert!(vertex_labels.contains(&vertex_label));
        assert!(edge_labels.contains(&edge_label));
    }

    #[test]
    fn test_get_and_drop_index() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_get_and_drop_index: JANUSGRAPH_HOST not set");
            return;
        }

        let manager = create_test_schema_manager();
        let vertex_label = "gdi_v_".to_string() + &Uuid::new_v4().to_string();
        let prop_name = "gdi_p".to_string();
        let index_name = "gdi_i_".to_string() + &Uuid::new_v4().to_string();

        let vertex_schema = VertexLabelSchema {
            label: vertex_label.clone(),
            properties: vec![PropertyDefinition {
                name: prop_name.clone(),
                property_type: PropertyType::StringType,
                required: false,
                unique: false,
                default_value: None,
            }],
            container: None,
        };
        manager.define_vertex_label(vertex_schema).unwrap();

        let index_def = IndexDefinition {
            name: index_name.clone(),
            label: vertex_label.clone(),
            container: Some(vertex_label),
            properties: vec![prop_name],
            unique: false,
            index_type: IndexType::Exact,
        };
        manager.create_index(index_def.clone()).unwrap();

        std::thread::sleep(std::time::Duration::from_secs(2));

        let fetched_index = manager.get_index(index_name.clone()).unwrap();
        assert!(fetched_index.is_some());
        assert_eq!(fetched_index.unwrap().name, index_name);

        let drop_result = manager.drop_index(index_name);
        assert!(matches!(
            drop_result,
            Err(GraphError::UnsupportedOperation(_))
        ));
    }

    #[test]
    fn test_unsupported_list_edge_types() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_unsupported_list_edge_types: JANUSGRAPH_HOST not set");
            return;
        }

        let manager = create_test_schema_manager();
        let result = manager.list_edge_types();
        assert!(matches!(result, Err(GraphError::UnsupportedOperation(_))));
    }

    #[test]
    fn test_unsupported_get_index() {
        if env::var("JANUSGRAPH_HOST").is_err() {
            println!("Skipping test_unsupported_get_index: JANUSGRAPH_HOST not set");
            return;
        }

        let manager = create_test_schema_manager();
        let result = manager.get_index("any_index".to_string());
        assert!(matches!(result, Err(GraphError::UnsupportedOperation(_))));
    }
}
