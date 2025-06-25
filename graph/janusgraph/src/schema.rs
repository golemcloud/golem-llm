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
        // DEBUG: Add unique identifier to confirm this JanusGraph implementation is being called
        eprintln!("DEBUG: JanusGraph schema manager get_schema_manager() called!");

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
        let script = "mgmt.getVertexLabels().collect{ it.name() }";
        let result = self.execute_management_query(script)?;

        let labels = self.parse_string_list_from_result(result)?;
        let exists = labels.contains(&label);

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
        // JanusGraph doesn't have getEdgeLabels() method, so we need to check directly
        let script = format!("mgmt.getEdgeLabel('{}') != null", label);
        let result = self.execute_management_query(&script)?;

        let exists = if let Some(graphson_obj) = result.as_object() {
            if let Some(value_array) = graphson_obj.get("@value").and_then(|v| v.as_array()) {
                value_array
                    .first()
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            } else {
                false
            }
        } else {
            result
                .as_array()
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        };

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
        // JanusGraph doesn't have getEdgeLabels() method, so return empty list or use alternative approach
        // For now, we'll return an error indicating this is not supported
        Err(GraphError::UnsupportedOperation(
            "Listing edge labels is not supported in JanusGraph management API".to_string(),
        ))
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

        index_builder.push_str(".indexOnly(label).buildCompositeIndex();");

        let wrapped_index_builder = format!("try {{ {} }} catch (Exception e) {{ if (!e.message.contains('already been defined')) throw e; }}", index_builder);
        script_parts.push(wrapped_index_builder);

        let script = script_parts.join("; ");
        self.execute_management_query(&script)?;

        Ok(())
    }

    fn drop_index(&self, name: String) -> Result<(), GraphError> {
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
            "
            try {{
                mgmt = graph.openManagement();
                result = {{ {} }}.call();
                mgmt.commit();
                return result;
            }} catch (Exception e) {{
                if (mgmt != null) {{
                    try {{ mgmt.rollback(); }} catch (Exception ignored) {{}}
                }}
                throw e;
            }}
            ",
            script
        );

        let mut last_error = None;
        for _attempt in 0..3 {
            match self.graph.api.execute(&full_script, None) {
                Ok(response) => {
                    let result = response["result"]["data"].clone();
                    return Ok(result);
                }
                Err(e) if e.to_string().contains("transaction is closed") => {
                    last_error = Some(e);
                    std::thread::sleep(std::time::Duration::from_millis(1000));
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            GraphError::InternalError(
                "Schema management transaction failed after retries".to_string(),
            )
        }))
    }

    fn parse_string_list_from_result(&self, result: Value) -> Result<Vec<String>, GraphError> {
        if let Some(graphson_obj) = result.as_object() {
            if let Some(value_array) = graphson_obj.get("@value").and_then(|v| v.as_array()) {
                return value_array
                    .iter()
                    .map(|v| {
                        v.as_str().map(String::from).ok_or_else(|| {
                            GraphError::InternalError("Expected string in list".to_string())
                        })
                    })
                    .collect();
            }
        }

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

        let items = if let Some(graphson_obj) = result.as_object() {
            if let Some(value_array) = graphson_obj.get("@value").and_then(|v| v.as_array()) {
                value_array
            } else {
                return Ok(indexes);
            }
        } else if let Some(arr) = result.as_array() {
            arr
        } else {
            return Ok(indexes);
        };

        for item in items {
            // Handling GraphSON map format: {"@type": "g:Map", "@value": [key1, value1, key2, value2, ...]}
            let map_data = if let Some(graphson_map) = item.as_object() {
                if let Some(map_array) = graphson_map.get("@value").and_then(|v| v.as_array()) {
                    let mut map = std::collections::HashMap::new();
                    let mut i = 0;
                    while i + 1 < map_array.len() {
                        if let Some(key) = map_array[i].as_str() {
                            map.insert(key.to_string(), map_array[i + 1].clone());
                        }
                        i += 2;
                    }
                    map
                } else {
                    continue;
                }
            } else if let Some(map) = item.as_object() {
                map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
            } else {
                continue;
            };

            let name = map_data
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let unique = map_data
                .get("unique")
                .and_then(|v| v.as_bool())
                .unwrap_or_default();
            let label = map_data
                .get("label")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            let properties = map_data
                .get("properties")
                .and_then(|v| {
                    if let Some(graphson_obj) = v.as_object() {
                        graphson_obj
                            .get("@value")
                            .and_then(|v| v.as_array())
                            .map(|props_array| {
                                props_array
                                    .iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect()
                            })
                    } else {
                        v.as_array().map(|props_array| {
                            props_array
                                .iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                    }
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

        Ok(indexes)
    }

    fn map_wit_type_to_janus_class(
        prop_type: &golem_graph::golem::graph::schema::PropertyType,
    ) -> &'static str {
        use golem_graph::golem::graph::schema::PropertyType;
        match prop_type {
            PropertyType::StringType => "String.class",
            PropertyType::Int64 => "Long.class",
            PropertyType::Float64Type => "Double.class",
            PropertyType::Boolean => "Boolean.class",
            PropertyType::Datetime => "Date.class",
            _ => "Object.class",
        }
    }
}
