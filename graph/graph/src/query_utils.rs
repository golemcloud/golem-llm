use crate::golem::graph::{
    errors::GraphError,
    types::{ComparisonOperator, FilterCondition, PropertyValue, SortSpec},
};
use serde_json::{Map, Value};

/// A struct to hold the syntax for a specific query language (e.g., Cypher, AQL).
pub struct QuerySyntax {
    pub equal: &'static str,
    pub not_equal: &'static str,
    pub less_than: &'static str,
    pub less_than_or_equal: &'static str,
    pub greater_than: &'static str,
    pub greater_than_or_equal: &'static str,
    pub contains: &'static str,
    pub starts_with: &'static str,
    pub ends_with: &'static str,
    pub regex_match: &'static str,
    pub param_prefix: &'static str,
}

impl QuerySyntax {
    fn map_operator(&self, op: ComparisonOperator) -> Result<&'static str, GraphError> {
        Ok(match op {
            ComparisonOperator::Equal => self.equal,
            ComparisonOperator::NotEqual => self.not_equal,
            ComparisonOperator::LessThan => self.less_than,
            ComparisonOperator::LessThanOrEqual => self.less_than_or_equal,
            ComparisonOperator::GreaterThan => self.greater_than,
            ComparisonOperator::GreaterThanOrEqual => self.greater_than_or_equal,
            ComparisonOperator::Contains => self.contains,
            ComparisonOperator::StartsWith => self.starts_with,
            ComparisonOperator::EndsWith => self.ends_with,
            ComparisonOperator::RegexMatch => self.regex_match,
            ComparisonOperator::InList | ComparisonOperator::NotInList => {
                return Err(GraphError::UnsupportedOperation(
                    "IN and NOT IN operators are not yet supported by the query builder."
                        .to_string(),
                ))
            }
        })
    }
}

pub fn build_where_clause<F>(
    filters: &Option<Vec<FilterCondition>>,
    variable: &str,
    params: &mut Map<String, Value>,
    syntax: &QuerySyntax,
    value_converter: F,
) -> Result<String, GraphError>
where
    F: Fn(PropertyValue) -> Result<Value, GraphError>,
{
    let mut where_clauses = Vec::new();
    if let Some(filters) = filters {
        for filter in filters.iter() {
            let op_str = syntax.map_operator(filter.operator)?;
            let param_name = format!("p{}", params.len());
            let clause = format!(
                "{}.{} {} {}{}",
                variable, filter.property, op_str, syntax.param_prefix, param_name
            );
            where_clauses.push(clause);
            params.insert(param_name, value_converter(filter.value.clone())?);
        }
    }

    if where_clauses.is_empty() {
        Ok("".to_string())
    } else {
        Ok(format!("WHERE {}", where_clauses.join(" AND ")))
    }
}

pub fn build_sort_clause(sort: &Option<Vec<SortSpec>>, variable: &str) -> String {
    if let Some(sort_specs) = sort {
        if !sort_specs.is_empty() {
            let order_items: Vec<String> = sort_specs
                .iter()
                .map(|s| {
                    format!(
                        "{}.{} {}",
                        variable,
                        s.property,
                        if s.ascending { "ASC" } else { "DESC" }
                    )
                })
                .collect();
            return format!("ORDER BY {}", order_items.join(", "));
        }
    }
    "".to_string()
}
