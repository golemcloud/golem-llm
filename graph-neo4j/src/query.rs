use crate::conversions;
use crate::{GraphNeo4jComponent, Transaction};
use golem_graph::golem::graph::{
    errors::GraphError,
    query::{Guest as QueryGuest, QueryExecutionResult, QueryOptions, QueryParameters},
};
use serde_json::{json, Map};

impl Transaction {
    pub fn execute_query(
        &self,
        query: String,
        parameters: Option<QueryParameters>,
        _options: Option<QueryOptions>,
    ) -> Result<QueryExecutionResult, GraphError> {
        let mut params = Map::new();
        if let Some(p) = parameters {
            for (key, value) in p {
                params.insert(key, conversions::to_json_value(value)?);
            }
        }

        let statement = json!({
            "statement": query,
            "parameters": params,
        });

        let statements = json!({ "statements": [statement] });
        let response = self
            .api
            .execute_in_transaction(&self.transaction_url, statements)?;

        let result = response["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| {
                GraphError::InternalError(
                    "Invalid response from Neo4j for execute_query".to_string(),
                )
            })?;

        if let Some(errors) = result["errors"].as_array() {
            if !errors.is_empty() {
                return Err(GraphError::InvalidQuery(errors[0].to_string()));
            }
        }

        let columns: Vec<String> = result["columns"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_str().unwrap_or_default().to_string())
                    .collect()
            })
            .unwrap_or_default();

        let empty_vec = vec![];
        let data = result["data"].as_array().unwrap_or(&empty_vec);
        let mut rows = Vec::new();
        for item in data {
            if let Some(row_data) = item["row"].as_array() {
                rows.push(row_data.clone());
            }
        }

        let query_result_value = if columns.len() == 1 {
            let mut values = Vec::new();
            for row in rows {
                if let Some(val) = row.first() {
                    values.push(conversions::from_json_value(val.clone())?);
                }
            }
            golem_graph::golem::graph::query::QueryResult::Values(values)
        } else {
            let mut maps = Vec::new();
            for row in rows {
                let mut map_row = Vec::new();
                for (i, col_name) in columns.iter().enumerate() {
                    if let Some(val) = row.get(i) {
                        map_row
                            .push((col_name.clone(), conversions::from_json_value(val.clone())?));
                    }
                }
                maps.push(map_row);
            }
            golem_graph::golem::graph::query::QueryResult::Maps(maps)
        };

        Ok(QueryExecutionResult {
            query_result_value,
            execution_time_ms: None,
            rows_affected: None,
            explanation: None,
            profile_data: None,
        })
    }

    pub(crate) fn execute_schema_query_and_extract_string_list(
        &self,
        query: &str,
    ) -> Result<Vec<String>, GraphError> {
        let statement = json!({ "statement": query, "parameters": {} });
        let statements = json!({ "statements": [statement] });
        let response = self
            .api
            .execute_in_transaction(&self.transaction_url, statements)?;

        let result = response["results"]
            .as_array()
            .and_then(|r| r.first())
            .ok_or_else(|| {
                GraphError::InternalError("Invalid response for schema query".to_string())
            })?;

        if let Some(errors) = result["errors"].as_array() {
            if !errors.is_empty() {
                return Err(GraphError::InvalidQuery(errors[0].to_string()));
            }
        }

        let empty_vec = vec![];
        let data = result["data"].as_array().unwrap_or(&empty_vec);
        let mut items = Vec::new();

        for item in data {
            if let Some(row) = item["row"].as_array() {
                if let Some(value) = row.first().and_then(|v| v.as_str()) {
                    items.push(value.to_string());
                }
            }
        }
        Ok(items)
    }
}

impl QueryGuest for GraphNeo4jComponent {
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
mod tests {
    use crate::client::Neo4jApi;
    use crate::Transaction;
    use golem_graph::golem::graph::{
        query::{QueryParameters, QueryResult},
        types::PropertyValue,
    };
    use std::{env, sync::Arc};

    fn create_test_transaction() -> Transaction {
        let host = env::var("NEO4J_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port = env::var("NEO4J_PORT")
            .unwrap_or_else(|_| "7474".to_string())
            .parse()
            .unwrap();
        let user = env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string());
        let password = env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".to_string());

        let api = Neo4jApi::new(&host, port, &user, &password);
        let transaction_url = api.begin_transaction().unwrap();
        Transaction {
            api: Arc::new(api),
            transaction_url,
        }
    }

    fn setup_test_data(tx: &Transaction) {
        tx.execute_query(
            "CREATE (p:Player {name: 'Alice', score: 100})".to_string(),
            None,
            None,
        )
        .unwrap();
        tx.execute_query(
            "CREATE (p:Player {name: 'Bob', score: 200})".to_string(),
            None,
            None,
        )
        .unwrap();
    }

    fn cleanup_test_data(tx: &Transaction) {
        tx.execute_query("MATCH (p:Player) DETACH DELETE p".to_string(), None, None)
            .unwrap();
    }

    #[test]
    fn test_simple_query() {
        if env::var("NEO4J_HOST").is_err() {
            println!("Skipping test_simple_query: NEO4J_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        setup_test_data(&tx);

        let result = tx
            .execute_query(
                "MATCH (p:Player) WHERE p.name = 'Alice' RETURN p.score".to_string(),
                None,
                None,
            )
            .unwrap();
        match result.query_result_value {
            QueryResult::Values(values) => {
                assert_eq!(values.len(), 1);
                assert_eq!(values[0], PropertyValue::Int64(100));
            }
            _ => panic!(
                "Expected Values result, got {:?}",
                result.query_result_value
            ),
        }

        cleanup_test_data(&tx);
        tx.commit().unwrap();
    }

    #[test]
    fn test_map_query_with_params() {
        if env::var("NEO4J_HOST").is_err() {
            println!("Skipping test_map_query_with_params: NEO4J_HOST not set");
            return;
        }
        let tx = create_test_transaction();
        setup_test_data(&tx);

        let params: QueryParameters = vec![(
            "player_name".to_string(),
            PropertyValue::StringValue("Alice".to_string()),
        )];
        let result = tx
            .execute_query(
                "MATCH (p:Player {name: $player_name}) RETURN p.name AS name, p.score AS score"
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
                let score = row.iter().find(|(k, _)| k == "score").unwrap();
                assert_eq!(name.1, PropertyValue::StringValue("Alice".to_string()));
                assert_eq!(score.1, PropertyValue::Int64(100));
            }
            _ => panic!("Expected Maps result, got {:?}", result.query_result_value),
        }

        cleanup_test_data(&tx);
        tx.commit().unwrap();
    }

    #[test]
    fn test_complex_query_and_cleanup() {
        if env::var("NEO4J_HOST").is_err() {
            println!("Skipping test_complex_query_and_cleanup: NEO4J_HOST not set");
            return;
        }

        let tx = create_test_transaction();

        // Create nodes and relationships
        tx.execute_query(
            "CREATE (:User {id: 1})-[:FRIENDS_WITH]->(:User {id: 2})".to_string(),
            None,
            None,
        )
        .unwrap();
        tx.execute_query(
            "CREATE (:User {id: 2})-[:FRIENDS_WITH]->(:User {id: 3})".to_string(),
            None,
            None,
        )
        .unwrap();

        // Find paths
        let result = tx
            .execute_query(
                "MATCH path = (:User)-[:FRIENDS_WITH*]->(:User) RETURN length(path) AS len"
                    .to_string(),
                None,
                None,
            )
            .unwrap();

        match result.query_result_value {
            QueryResult::Values(values) => {
                assert_eq!(values.len(), 4); // 2 of length 1, 2 of length 2
            }
            _ => panic!(
                "Expected Values result, got {:?}",
                result.query_result_value
            ),
        }

        // Cleanup
        tx.execute_query("MATCH (n:User) DETACH DELETE n".to_string(), None, None)
            .unwrap();
        tx.commit().unwrap();
    }
}
