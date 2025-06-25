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
            "resultDataContents": ["row","graph"]
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
