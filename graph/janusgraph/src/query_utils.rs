use golem_graph::golem::graph::{
    errors::GraphError,
    types::{ComparisonOperator, FilterCondition, SortSpec},
};
use serde_json::{Map, Value};

/// Builds a Gremlin `has()` step chain from a WIT FilterCondition.
/// Returns the query segment and the bound values.
pub(crate) fn build_gremlin_filter_step(
    condition: &FilterCondition,
    binding_map: &mut Map<String, Value>,
) -> Result<String, GraphError> {
    let key_binding = format!("fk_{}", binding_map.len());
    binding_map.insert(
        key_binding.clone(),
        Value::String(condition.property.clone()),
    );

    let predicate = match condition.operator {
        ComparisonOperator::Equal => "eq".to_string(),
        ComparisonOperator::NotEqual => "neq".to_string(),
        ComparisonOperator::GreaterThan => "gt".to_string(),
        ComparisonOperator::GreaterThanOrEqual => "gte".to_string(),
        ComparisonOperator::LessThan => "lt".to_string(),
        ComparisonOperator::LessThanOrEqual => "lte".to_string(),
        ComparisonOperator::Contains => "textContains".to_string(),
        ComparisonOperator::StartsWith => "textStartsWith".to_string(),
        ComparisonOperator::EndsWith => "textEndsWith".to_string(),
        ComparisonOperator::RegexMatch => "textRegex".to_string(),
        _ => {
            return Err(GraphError::UnsupportedOperation(
                "This filter predicate is not yet supported.".to_string(),
            ))
        }
    };

    let value_binding = format!("fv_{}", binding_map.len());
    let json_value = crate::conversions::to_json_value(condition.value.clone())?;
    binding_map.insert(value_binding.clone(), json_value);

    Ok(format!(
        ".has({}, {}({}))",
        key_binding, predicate, value_binding
    ))
}

pub(crate) fn build_gremlin_sort_clause(sort_specs: &[SortSpec]) -> String {
    if sort_specs.is_empty() {
        return String::new();
    }

    let mut sort_clause = ".order()".to_string();

    for spec in sort_specs {
        let order = if spec.ascending { "incr" } else { "decr" };
        sort_clause.push_str(&format!(".by('{}', {})", spec.property, order));
    }

    sort_clause
}
