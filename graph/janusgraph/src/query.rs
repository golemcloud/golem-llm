use crate::conversions;
use crate::{GraphJanusGraphComponent, Transaction};
use golem_graph::golem::graph::types::PropertyValue;
use golem_graph::golem::graph::{
    errors::GraphError,
    query::{Guest as QueryGuest, QueryExecutionResult, QueryParameters, QueryResult},
};
use serde_json::{json, Map, Value};

fn to_bindings(parameters: QueryParameters) -> Result<Map<String, Value>, GraphError> {
    let mut bindings = Map::new();
    for (key, value) in parameters {
        bindings.insert(key, conversions::to_json_value(value)?);
    }
    Ok(bindings)
}

fn parse_gremlin_response(response: Value) -> Result<QueryResult, GraphError> {
    let result_data = response
        .get("result")
        .and_then(|r| r.get("data"))
        .ok_or_else(|| {
            GraphError::InternalError("Invalid response structure from Gremlin".to_string())
        })?;

    // Handling GraphSON format: {"@type": "g:List", "@value": [...]}
    let arr = if let Some(graphson_obj) = result_data.as_object() {
        if let Some(value_array) = graphson_obj.get("@value").and_then(|v| v.as_array()) {
            value_array
        } else {
            return Ok(QueryResult::Values(vec![]));
        }
    } else if let Some(direct_array) = result_data.as_array() {
        direct_array
    } else {
        return Ok(QueryResult::Values(vec![]));
    };

    if arr.is_empty() {
        return Ok(QueryResult::Values(vec![]));
    }

    if let Some(first_item) = arr.first() {
        if first_item.is_object() {
            if let Some(obj) = first_item.as_object() {
                if obj.get("@type") == Some(&Value::String("g:Map".to_string())) {
                    let mut maps = Vec::new();
                    for item in arr {
                        if let Some(obj) = item.as_object() {
                            if let Some(map_array) = obj.get("@value").and_then(|v| v.as_array()) {
                                let mut row: Vec<(String, PropertyValue)> = Vec::new();
                                // Processing GraphSON Map: array contains alternating keys and values
                                let mut i = 0;
                                while i + 1 < map_array.len() {
                                    if let (Some(key_val), Some(value_val)) =
                                        (map_array.get(i), map_array.get(i + 1))
                                    {
                                        if let Some(key_str) = key_val.as_str() {
                                            // Handling GraphSON List format for valueMap results
                                            if let Some(graphson_obj) = value_val.as_object() {
                                                if graphson_obj.get("@type")
                                                    == Some(&Value::String("g:List".to_string()))
                                                {
                                                    if let Some(list_values) = graphson_obj
                                                        .get("@value")
                                                        .and_then(|v| v.as_array())
                                                    {
                                                        if let Some(first_value) =
                                                            list_values.first()
                                                        {
                                                            row.push((
                                                                key_str.to_string(),
                                                                conversions::from_gremlin_value(
                                                                    first_value,
                                                                )?,
                                                            ));
                                                        }
                                                    }
                                                } else {
                                                    row.push((
                                                        key_str.to_string(),
                                                        conversions::from_gremlin_value(value_val)?,
                                                    ));
                                                }
                                            } else {
                                                row.push((
                                                    key_str.to_string(),
                                                    conversions::from_gremlin_value(value_val)?,
                                                ));
                                            }
                                        }
                                    }
                                    i += 2;
                                }
                                maps.push(row);
                            }
                        }
                    }
                    return Ok(QueryResult::Maps(maps));
                } else if obj.contains_key("@type") && obj.contains_key("@value") {
                    let values = arr
                        .iter()
                        .map(conversions::from_gremlin_value)
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok(QueryResult::Values(values));
                } else {
                    let mut maps = Vec::new();
                    for item in arr {
                        if let Some(gremlin_map) = item.as_object() {
                            let mut row: Vec<(String, PropertyValue)> = Vec::new();
                            for (key, gremlin_value) in gremlin_map {
                                if let Some(graphson_obj) = gremlin_value.as_object() {
                                    if graphson_obj.get("@type")
                                        == Some(&Value::String("g:List".to_string()))
                                    {
                                        if let Some(list_values) =
                                            graphson_obj.get("@value").and_then(|v| v.as_array())
                                        {
                                            if let Some(first_value) = list_values.first() {
                                                row.push((
                                                    key.clone(),
                                                    conversions::from_gremlin_value(first_value)?,
                                                ));
                                            }
                                        }
                                    } else {
                                        row.push((
                                            key.clone(),
                                            conversions::from_gremlin_value(gremlin_value)?,
                                        ));
                                    }
                                } else if let Some(inner_array) = gremlin_value.as_array() {
                                    if let Some(actual_value) = inner_array.first() {
                                        row.push((
                                            key.clone(),
                                            conversions::from_gremlin_value(actual_value)?,
                                        ));
                                    }
                                } else {
                                    row.push((
                                        key.clone(),
                                        conversions::from_gremlin_value(gremlin_value)?,
                                    ));
                                }
                            }
                            maps.push(row);
                        }
                    }
                    return Ok(QueryResult::Maps(maps));
                }
            }
        }
    }

    let values = arr
        .iter()
        .map(conversions::from_gremlin_value)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(QueryResult::Values(values))
}

impl Transaction {
    pub fn execute_query(
        &self,
        query: String,
        parameters: Option<QueryParameters>,
        _options: Option<golem_graph::golem::graph::query::QueryOptions>,
    ) -> Result<QueryExecutionResult, GraphError> {
        let params = parameters.unwrap_or_default();
        let bindings_map = to_bindings(params)?;

        let response = self.api.execute(&query, Some(json!(bindings_map)))?;
        let query_result_value = parse_gremlin_response(response)?;

        Ok(QueryExecutionResult {
            query_result_value,
            execution_time_ms: None,
            rows_affected: None,
            explanation: None,
            profile_data: None,
        })
    }
}

impl QueryGuest for GraphJanusGraphComponent {
    fn execute_query(
        transaction: golem_graph::golem::graph::transactions::TransactionBorrow<'_>,
        query: String,
        parameters: Option<QueryParameters>,
        options: Option<golem_graph::golem::graph::query::QueryOptions>,
    ) -> Result<QueryExecutionResult, GraphError> {
        let tx: &Transaction = transaction.get();
        tx.execute_query(query, parameters, options)
    }
}
