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
