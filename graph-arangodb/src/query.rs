use crate::{conversions, GraphArangoDbComponent, Transaction};
use golem_graph::golem::graph::{
    errors::GraphError,
    query::{
        Guest as QueryGuest, QueryExecutionResult, QueryOptions, QueryParameters, QueryResult,
    },
};

impl Transaction {
    pub fn execute_query(
        &self,
        query: String,
        parameters: Option<QueryParameters>,
        _options: Option<QueryOptions>,
    ) -> Result<QueryExecutionResult, GraphError> {
        let mut bind_vars = serde_json::Map::new();
        if let Some(p) = parameters {
            for (key, value) in p {
                bind_vars.insert(key, conversions::to_arango_value(value)?);
            }
        }

        let query_json = serde_json::json!({
            "query": query,
            "bindVars": bind_vars,
        });

        let response = self
            .api
            .execute_in_transaction(&self.transaction_id, query_json)?;

        let result_array = response.as_array().ok_or_else(|| {
            GraphError::InternalError("Expected array in AQL query response".to_string())
        })?;

        let query_result_value = if result_array.is_empty() {
            QueryResult::Values(vec![])
        } else {
            let first_item = &result_array[0];
            if first_item.is_object() {
                let obj = first_item.as_object().unwrap();
                if obj.contains_key("_id") && obj.contains_key("_from") && obj.contains_key("_to") {
                    let mut edges = vec![];
                    for item in result_array {
                        if let Some(doc) = item.as_object() {
                            let collection = doc
                                .get("_id")
                                .and_then(|v| v.as_str())
                                .and_then(|s| s.split('/').next())
                                .unwrap_or_default();
                            edges.push(crate::helpers::parse_edge_from_document(doc, collection)?);
                        }
                    }
                    QueryResult::Edges(edges)
                } else if obj.contains_key("_id") {
                    let mut vertices = vec![];
                    for item in result_array {
                        if let Some(doc) = item.as_object() {
                            let collection = doc
                                .get("_id")
                                .and_then(|v| v.as_str())
                                .and_then(|s| s.split('/').next())
                                .unwrap_or_default();
                            vertices
                                .push(crate::helpers::parse_vertex_from_document(doc, collection)?);
                        }
                    }
                    QueryResult::Vertices(vertices)
                } else {
                    let mut maps = vec![];
                    for item in result_array {
                        if let Some(doc) = item.as_object() {
                            let mut map_row = vec![];
                            for (key, value) in doc {
                                map_row.push((
                                    key.clone(),
                                    conversions::from_arango_value(value.clone())?,
                                ));
                            }
                            maps.push(map_row);
                        }
                    }
                    QueryResult::Maps(maps)
                }
            } else {
                let mut values = vec![];
                for item in result_array {
                    values.push(conversions::from_arango_value(item.clone())?);
                }
                QueryResult::Values(values)
            }
        };

        Ok(QueryExecutionResult {
            query_result_value,
            execution_time_ms: None, // ArangoDB response can contain this, but it's an enterprise feature.
            rows_affected: None,
            explanation: None,
            profile_data: None,
        })
    }
}

impl QueryGuest for GraphArangoDbComponent {
    fn execute_query(
        transaction: golem_graph::golem::graph::transactions::TransactionBorrow<'_>,
        query: String,
        parameters: Option<QueryParameters>,
        options: Option<QueryOptions>,
    ) -> Result<QueryExecutionResult, GraphError> {
        let tx: &Transaction = transaction.get();
        tx.execute_query(query, parameters, options)
    }
}

#[cfg(test)]
mod query_tests {
    use super::*;
    use crate::client::ArangoDbApi;
    use golem_graph::golem::graph::transactions::GuestTransaction;
    use golem_graph::golem::graph::types::PropertyValue;
    use golem_graph::golem::graph::{
        errors::GraphError,
        query::{QueryParameters, QueryResult},
    };
    use std::{env, sync::Arc};

    fn create_test_transaction() -> Transaction {
        let host = env::var("ARANGODB_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port: u16 = env::var("ARANGODB_PORT")
            .unwrap_or_else(|_| "8529".to_string())
            .parse()
            .expect("Invalid ARANGODB_PORT");
        let user = env::var("ARANGODB_USER").unwrap_or_else(|_| "root".to_string());
        let pass = env::var("ARANGODB_PASS").unwrap_or_else(|_| "".to_string());
        let db = env::var("ARANGODB_DB").unwrap_or_else(|_| "test_db".to_string());
        let api = ArangoDbApi::new(&host, port, &user, &pass, &db);
        let transaction_id = api.begin_transaction(false).unwrap();
        let api = Arc::new(api);
        Transaction {
            api,
            transaction_id,
        }
    }

    fn setup_test_data(tx: &Transaction) {
        let prop = |k: &str, v| (k.to_string(), v);
        tx.create_vertex(
            "person".into(),
            vec![
                prop("name", PropertyValue::StringValue("marko".into())),
                prop("age", PropertyValue::Int64(29)),
            ],
        )
        .unwrap();
        tx.create_vertex(
            "person".into(),
            vec![
                prop("name", PropertyValue::StringValue("vadas".into())),
                prop("age", PropertyValue::Int64(27)),
            ],
        )
        .unwrap();
        tx.create_vertex(
            "software".into(),
            vec![
                prop("name", PropertyValue::StringValue("lop".into())),
                prop("lang", PropertyValue::StringValue("java".into())),
            ],
        )
        .unwrap();
    }

    fn cleanup_test_data(tx: &Transaction) {
        tx.execute_query("FOR v IN person REMOVE v IN person".to_string(), None, None)
            .unwrap();
        tx.execute_query(
            "FOR v IN software REMOVE v IN software".to_string(),
            None,
            None,
        )
        .unwrap();
        tx.commit().unwrap();
    }

    #[test]
    fn test_simple_value_query() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_simple_value_query: ARANGODB_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        setup_test_data(&tx);

        let result = tx
            .execute_query(
                "FOR v IN person FILTER v.name == 'marko' RETURN v.age".to_string(),
                None,
                None,
            )
            .unwrap();

        match result.query_result_value {
            QueryResult::Values(vals) => {
                assert_eq!(vals.len(), 1);
                assert_eq!(vals[0], PropertyValue::Int64(29));
            }
            _ => panic!("Expected Values result"),
        }

        cleanup_test_data(&tx);
    }

    #[test]
    fn test_map_query_with_params() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_map_query_with_params: ARANGODB_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        setup_test_data(&tx);

        let params: QueryParameters = vec![(
            "person_name".to_string(),
            PropertyValue::StringValue("marko".to_string()),
        )];
        let result = tx
            .execute_query(
                "FOR v IN person FILTER v.name == @person_name RETURN { name: v.name, age: v.age }"
                    .to_string(),
                Some(params),
                None,
            )
            .unwrap();

        match result.query_result_value {
            QueryResult::Maps(maps) => {
                assert_eq!(maps.len(), 1);
                let row = &maps[0];
                let name = row.iter().find(|(k, _)| k == "name").unwrap();
                let age = row.iter().find(|(k, _)| k == "age").unwrap();
                assert_eq!(name.1, PropertyValue::StringValue("marko".into()));
                assert_eq!(age.1, PropertyValue::Int64(29));
            }
            _ => panic!("Expected Maps result"),
        }

        cleanup_test_data(&tx);
    }

    #[test]
    fn test_complex_query() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_complex_query: ARANGODB_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        setup_test_data(&tx);

        let result = tx
            .execute_query(
                "RETURN LENGTH(FOR v IN person RETURN 1)".to_string(),
                None,
                None,
            )
            .unwrap();

        match result.query_result_value {
            QueryResult::Values(vals) => {
                assert_eq!(vals.len(), 1);
                assert_eq!(vals[0], PropertyValue::Int64(2));
            }
            _ => panic!("Expected Values result"),
        }

        cleanup_test_data(&tx);
    }

    #[test]
    fn test_empty_result_query() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_empty_result_query: ARANGODB_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        setup_test_data(&tx);

        let result = tx
            .execute_query(
                "FOR v IN person FILTER v.name == 'non_existent' RETURN v".to_string(),
                None,
                None,
            )
            .unwrap();

        match result.query_result_value {
            QueryResult::Values(vals) => assert!(vals.is_empty()),
            _ => panic!("Expected empty Values result"),
        }

        cleanup_test_data(&tx);
    }

    #[test]
    fn test_invalid_query() {
        if env::var("ARANGODB_HOST").is_err() {
            println!("Skipping test_invalid_query: ARANGODB_HOST not set");
            return;
        }
        let tx = create_test_transaction();

        let res = tx.execute_query("FOR v IN person INVALID".to_string(), None, None);
        assert!(matches!(res, Err(GraphError::InvalidQuery(_))));
    }
}
