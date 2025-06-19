use crate::{helpers, GraphArangoDbComponent, SchemaManager};
use golem_graph::{
    durability::ExtendedGuest,
    golem::graph::{
        errors::GraphError,
        schema::{
            ContainerInfo, ContainerType, EdgeLabelSchema, EdgeTypeDefinition,
            Guest as SchemaGuest, GuestSchemaManager, IndexDefinition,
            SchemaManager as SchemaManagerResource, VertexLabelSchema,
        },
    },
};
use std::sync::Arc;

impl SchemaGuest for GraphArangoDbComponent {
    type SchemaManager = SchemaManager;

    fn get_schema_manager() -> Result<golem_graph::golem::graph::schema::SchemaManager, GraphError>
    {
        let config = helpers::config_from_env()?;

        let graph = GraphArangoDbComponent::connect_internal(&config)?;

        let manager = SchemaManager {
            graph: Arc::new(graph),
        };

        Ok(SchemaManagerResource::new(manager))
    }
}

impl GuestSchemaManager for SchemaManager {
    fn define_vertex_label(&self, schema: VertexLabelSchema) -> Result<(), GraphError> {
        self.create_container(schema.label, ContainerType::VertexContainer)
    }

    fn define_edge_label(&self, schema: EdgeLabelSchema) -> Result<(), GraphError> {
        self.create_container(schema.label, ContainerType::EdgeContainer)
    }

    fn get_vertex_label_schema(
        &self,
        _label: String,
    ) -> Result<Option<VertexLabelSchema>, GraphError> {
        Err(GraphError::UnsupportedOperation(
            "get_vertex_label_schema is not yet supported".to_string(),
        ))
    }

    fn get_edge_label_schema(&self, _label: String) -> Result<Option<EdgeLabelSchema>, GraphError> {
        Err(GraphError::UnsupportedOperation(
            "get_edge_label_schema is not yet supported".to_string(),
        ))
    }

    fn list_vertex_labels(&self) -> Result<Vec<String>, GraphError> {
        let all_containers = self.list_containers()?;
        Ok(all_containers
            .into_iter()
            .filter(|c| matches!(c.container_type, ContainerType::VertexContainer))
            .map(|c| c.name)
            .collect())
    }

    fn list_edge_labels(&self) -> Result<Vec<String>, GraphError> {
        let all_containers = self.list_containers()?;
        Ok(all_containers
            .into_iter()
            .filter(|c| matches!(c.container_type, ContainerType::EdgeContainer))
            .map(|c| c.name)
            .collect())
    }

    fn create_index(&self, index: IndexDefinition) -> Result<(), GraphError> {
        self.graph.api.create_index(
            index.label,
            index.properties,
            index.unique,
            index.index_type,
        )
    }

    fn drop_index(&self, name: String) -> Result<(), GraphError> {
        self.graph.api.drop_index(&name)
    }

    fn list_indexes(&self) -> Result<Vec<IndexDefinition>, GraphError> {
        self.graph.api.list_indexes()
    }

    fn get_index(&self, name: String) -> Result<Option<IndexDefinition>, GraphError> {
        self.graph.api.get_index(&name)
    }

    fn define_edge_type(&self, definition: EdgeTypeDefinition) -> Result<(), GraphError> {
        self.graph.api.define_edge_type(definition)
    }

    fn list_edge_types(&self) -> Result<Vec<EdgeTypeDefinition>, GraphError> {
        self.graph.api.list_edge_types()
    }

    fn create_container(
        &self,
        name: String,
        container_type: ContainerType,
    ) -> Result<(), GraphError> {
        self.graph.api.create_collection(&name, container_type)
    }

    fn list_containers(&self) -> Result<Vec<ContainerInfo>, GraphError> {
        self.graph.api.list_collections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use golem_graph::golem::graph::schema::{
        ContainerInfo, ContainerType, EdgeLabelSchema, EdgeTypeDefinition, GuestSchemaManager,
        IndexDefinition, VertexLabelSchema,
    };
    use std::env;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn create_test_schema_manager() -> SchemaManager {
        let config = helpers::config_from_env().expect("config_from_env failed");
        let graph =
            GraphArangoDbComponent::connect_internal(&config).expect("connect_internal failed");
        SchemaManager {
            graph: Arc::new(graph),
        }
    }

    /// Generate a pseudo‐unique suffix based on current time
    fn unique_suffix() -> String {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
            .to_string()
    }

    #[test]
    fn test_define_and_list_vertex_label() {
        if env::var("ARANGO_HOST").is_err() {
            println!("Skipping test_define_and_list_vertex_label");
            return;
        }
        let mgr = create_test_schema_manager();
        let label = format!("vlabel_{}", unique_suffix());
        // define—with container=None
        mgr.define_vertex_label(VertexLabelSchema {
            label: label.clone(),
            properties: vec![],
            container: None,
        })
        .unwrap();
        // list
        let vlabels = mgr.list_vertex_labels().unwrap();
        assert!(vlabels.contains(&label));
    }

    #[test]
    fn test_define_and_list_edge_label() {
        if env::var("ARANGO_HOST").is_err() {
            println!("Skipping test_define_and_list_edge_label");
            return;
        }
        let mgr = create_test_schema_manager();
        let label = format!("elabel_{}", unique_suffix());
        mgr.define_edge_label(EdgeLabelSchema {
            label: label.clone(),
            properties: vec![],
            from_labels: None,
            to_labels: None,
            container: None,
        })
        .unwrap();
        let elabels = mgr.list_edge_labels().unwrap();
        assert!(elabels.contains(&label));
    }

    #[test]
    fn test_container_roundtrip() {
        if env::var("ARANGO_HOST").is_err() {
            println!("Skipping test_container_roundtrip");
            return;
        }
        let mgr = create_test_schema_manager();
        let name = format!("col_{}", unique_suffix());
        mgr.create_container(name.clone(), ContainerType::VertexContainer)
            .unwrap();
        let cols: Vec<ContainerInfo> = mgr.list_containers().unwrap();
        assert!(cols
            .iter()
            .any(|c| c.name == name && c.container_type == ContainerType::VertexContainer));
    }

    #[test]
    fn test_index_lifecycle() {
        if env::var("ARANGO_HOST").is_err() {
            println!("Skipping test_index_lifecycle");
            return;
        }
        let mgr = create_test_schema_manager();
        let col = format!("idxcol_{}", unique_suffix());
        mgr.create_container(col.clone(), ContainerType::VertexContainer)
            .unwrap();

        let idx_name = format!("idx_{}", unique_suffix());
        let idx_def = IndexDefinition {
            name: idx_name.clone(),
            label: col.clone(),
            container: Some(col.clone()),
            properties: vec!["foo".to_string()],
            unique: false,
            index_type: golem_graph::golem::graph::schema::IndexType::Exact,
        };

        mgr.create_index(idx_def.clone()).unwrap();

        let all = mgr.list_indexes().unwrap();
        assert!(all.iter().any(|i| i.name == idx_name));

        let fetched = mgr.get_index(idx_name.clone()).unwrap();
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().name, idx_name);

        mgr.drop_index(idx_name.clone()).unwrap();
        let after = mgr.get_index(idx_name).unwrap();
        assert!(after.is_none());
    }

    #[test]
    fn test_edge_type_and_list() {
        if env::var("ARANGO_HOST").is_err() {
            println!("Skipping test_edge_type_and_list");
            return;
        }
        let mgr = create_test_schema_manager();
        let v1 = format!("V1_{}", unique_suffix());
        let v2 = format!("V2_{}", unique_suffix());
        mgr.create_container(v1.clone(), ContainerType::VertexContainer)
            .unwrap();
        mgr.create_container(v2.clone(), ContainerType::VertexContainer)
            .unwrap();

        let def = EdgeTypeDefinition {
            collection: format!("E_{}", unique_suffix()),
            from_collections: vec![v1.clone()],
            to_collections: vec![v2.clone()],
        };
        mgr.define_edge_type(def.clone()).unwrap();
        let etypes = mgr.list_edge_types().unwrap();
        assert!(etypes.iter().any(|e| e.collection == def.collection));
    }
}
