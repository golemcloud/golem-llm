use crate::client::{MeilisearchDoc, MeilisearchHit, MeilisearchSearchQuery, MeilisearchSearchResponse, MeilisearchIndexSettings};
use golem_search::golem::search::types::{
    Doc, FieldType, Schema, SchemaField, SearchError, SearchHit, SearchQuery, SearchResults,
};
use log::trace;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub fn query_to_meilisearch_query(query: &SearchQuery) -> Result<MeilisearchSearchQuery, SearchError> {
    let mut meilisearch_query = MeilisearchSearchQuery {
        q: query.q.clone(),
        offset: None,
        limit: None,
        filter: None,
        sort: None,
        facets: None,
        attributes_to_highlight: None,
        highlight_pre_tag: None,
        highlight_post_tag: None,
    };

    // Handle pagination
    if let Some(page) = query.page {
        let per_page = query.per_page.unwrap_or(20);
        meilisearch_query.offset = Some((page.saturating_sub(1)) * per_page);
        meilisearch_query.limit = Some(per_page);
    } else if let Some(offset) = query.offset {
        meilisearch_query.offset = Some(offset);
        if let Some(per_page) = query.per_page {
            meilisearch_query.limit = Some(per_page);
        }
    } else if let Some(per_page) = query.per_page {
        meilisearch_query.limit = Some(per_page);
    }

    // Handle filters
    if !query.filters.is_empty() {
        let filter_expressions: Vec<String> = query
            .filters
            .iter()
            .map(|filter| convert_filter(filter))
            .collect::<Result<Vec<_>, _>>()?;
        
        if !filter_expressions.is_empty() {
            meilisearch_query.filter = Some(filter_expressions.join(" AND "));
        }
    }

    // Handle sorting
    if !query.sort.is_empty() {
        let sort_clauses: Vec<String> = query
            .sort
            .iter()
            .map(|sort| convert_sort(sort))
            .collect::<Result<Vec<_>, _>>()?;
        
        if !sort_clauses.is_empty() {
            meilisearch_query.sort = Some(sort_clauses);
        }
    }

    // Handle facets
    if !query.facets.is_empty() {
        meilisearch_query.facets = Some(query.facets.clone());
    }

    // Handle highlighting
    if let Some(highlight) = &query.highlight {
        if let Ok(highlight_config) = serde_json::from_str::<Value>(highlight) {
            if let Some(fields) = highlight_config.get("fields").and_then(|v| v.as_array()) {
                let field_names: Vec<String> = fields
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                
                if !field_names.is_empty() {
                    meilisearch_query.attributes_to_highlight = Some(field_names);
                }
            }
            
            if let Some(pre_tag) = highlight_config.get("pre_tag").and_then(|v| v.as_str()) {
                meilisearch_query.highlight_pre_tag = Some(pre_tag.to_string());
            }
            
            if let Some(post_tag) = highlight_config.get("post_tag").and_then(|v| v.as_str()) {
                meilisearch_query.highlight_post_tag = Some(post_tag.to_string());
            }
        } else {
            // Default highlighting for all fields
            meilisearch_query.attributes_to_highlight = Some(vec!["*".to_string()]);
        }
    }

    Ok(meilisearch_query)
}

fn convert_filter(filter: &str) -> Result<String, SearchError> {
    // Try to parse as JSON first
    if let Ok(filter_value) = serde_json::from_str::<Value>(filter) {
        return convert_json_filter(&filter_value);
    }

    // Parse simple filter format: field:op:value
    let parts: Vec<&str> = filter.splitn(3, ':').collect();
    if parts.len() != 3 {
        return Err(SearchError::InvalidQuery {
            message: format!("Invalid filter format: {}", filter),
        });
    }

    let field = parts[0];
    let op = parts[1];
    let value = parts[2];

    match op {
        "eq" => Ok(format!("{} = \"{}\"", field, escape_filter_value(value))),
        "ne" => Ok(format!("{} != \"{}\"", field, escape_filter_value(value))),
        "gt" => Ok(format!("{} > {}", field, value)),
        "gte" => Ok(format!("{} >= {}", field, value)),
        "lt" => Ok(format!("{} < {}", field, value)),
        "lte" => Ok(format!("{} <= {}", field, value)),
        "in" => {
            let values: Vec<&str> = value.split(',').collect();
            let quoted_values: Vec<String> = values
                .iter()
                .map(|v| format!("\"{}\"", escape_filter_value(v)))
                .collect();
            Ok(format!("{} IN [{}]", field, quoted_values.join(", ")))
        }
        "exists" => Ok(format!("{} EXISTS", field)),
        "prefix" => Ok(format!("{} = \"{}*\"", field, escape_filter_value(value))),
        _ => Err(SearchError::InvalidQuery {
            message: format!("Unsupported filter operator: {}", op),
        }),
    }
}

fn convert_json_filter(filter: &Value) -> Result<String, SearchError> {
    // Handle complex nested filters
    if let Some(obj) = filter.as_object() {
        if let Some(bool_filter) = obj.get("bool") {
            return convert_bool_filter(bool_filter);
        }
        
        if let Some(term_filter) = obj.get("term") {
            return convert_term_filter(term_filter);
        }
        
        if let Some(range_filter) = obj.get("range") {
            return convert_range_filter(range_filter);
        }
        
        if let Some(terms_filter) = obj.get("terms") {
            return convert_terms_filter(terms_filter);
        }
        
        if let Some(exists_filter) = obj.get("exists") {
            return convert_exists_filter(exists_filter);
        }
    }

    Err(SearchError::InvalidQuery {
        message: "Unsupported filter format".to_string(),
    })
}

fn convert_bool_filter(bool_filter: &Value) -> Result<String, SearchError> {
    let mut clauses = Vec::new();
    
    if let Some(must) = bool_filter.get("must").and_then(|v| v.as_array()) {
        for clause in must {
            clauses.push(convert_json_filter(clause)?);
        }
    }
    
    if let Some(filter) = bool_filter.get("filter").and_then(|v| v.as_array()) {
        for clause in filter {
            clauses.push(convert_json_filter(clause)?);
        }
    }
    
    if let Some(must_not) = bool_filter.get("must_not").and_then(|v| v.as_array()) {
        for clause in must_not {
            clauses.push(format!("NOT ({})", convert_json_filter(clause)?));
        }
    }
    
    if clauses.is_empty() {
        return Err(SearchError::InvalidQuery {
            message: "Empty bool filter".to_string(),
        });
    }
    
    Ok(format!("({})", clauses.join(" AND ")))
}

fn convert_term_filter(term_filter: &Value) -> Result<String, SearchError> {
    if let Some(obj) = term_filter.as_object() {
        for (field, value) in obj {
            if let Some(val_str) = value.as_str() {
                return Ok(format!("{} = \"{}\"", field, escape_filter_value(val_str)));
            } else if let Some(val_num) = value.as_number() {
                return Ok(format!("{} = {}", field, val_num));
            }
        }
    }
    
    Err(SearchError::InvalidQuery {
        message: "Invalid term filter".to_string(),
    })
}

fn convert_range_filter(range_filter: &Value) -> Result<String, SearchError> {
    if let Some(obj) = range_filter.as_object() {
        for (field, range) in obj {
            if let Some(range_obj) = range.as_object() {
                let mut conditions = Vec::new();
                
                if let Some(gte) = range_obj.get("gte") {
                    conditions.push(format!("{} >= {}", field, gte));
                }
                if let Some(gt) = range_obj.get("gt") {
                    conditions.push(format!("{} > {}", field, gt));
                }
                if let Some(lte) = range_obj.get("lte") {
                    conditions.push(format!("{} <= {}", field, lte));
                }
                if let Some(lt) = range_obj.get("lt") {
                    conditions.push(format!("{} < {}", field, lt));
                }
                
                if !conditions.is_empty() {
                    return Ok(conditions.join(" AND "));
                }
            }
        }
    }
    
    Err(SearchError::InvalidQuery {
        message: "Invalid range filter".to_string(),
    })
}

fn convert_terms_filter(terms_filter: &Value) -> Result<String, SearchError> {
    if let Some(obj) = terms_filter.as_object() {
        for (field, values) in obj {
            if let Some(values_array) = values.as_array() {
                let quoted_values: Vec<String> = values_array
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(|v| format!("\"{}\"", escape_filter_value(v)))
                    .collect();
                
                if !quoted_values.is_empty() {
                    return Ok(format!("{} IN [{}]", field, quoted_values.join(", ")));
                }
            }
        }
    }
    
    Err(SearchError::InvalidQuery {
        message: "Invalid terms filter".to_string(),
    })
}

fn convert_exists_filter(exists_filter: &Value) -> Result<String, SearchError> {
    if let Some(field) = exists_filter.get("field").and_then(|v| v.as_str()) {
        Ok(format!("{} EXISTS", field))
    } else {
        Err(SearchError::InvalidQuery {
            message: "Invalid exists filter".to_string(),
        })
    }
}

fn convert_sort(sort: &str) -> Result<String, SearchError> {
    // Handle JSON sort format
    if let Ok(sort_value) = serde_json::from_str::<Value>(sort) {
        if let Some(obj) = sort_value.as_object() {
            for (field, order) in obj {
                if let Some(order_obj) = order.as_object() {
                    if let Some(order_str) = order_obj.get("order").and_then(|v| v.as_str()) {
                        return Ok(match order_str {
                            "desc" => format!("{}:desc", field),
                            _ => format!("{}:asc", field),
                        });
                    }
                } else if let Some(order_str) = order.as_str() {
                    return Ok(match order_str {
                        "desc" => format!("{}:desc", field),
                        _ => format!("{}:asc", field),
                    });
                }
            }
        }
    }

    // Handle simple format: field:direction
    let parts: Vec<&str> = sort.split(':').collect();
    match parts.len() {
        1 => Ok(format!("{}:asc", parts[0])),
        2 => {
            let field = parts[0];
            let direction = match parts[1] {
                "desc" | "DESC" => "desc",
                _ => "asc",
            };
            Ok(format!("{}:{}", field, direction))
        }
        _ => Err(SearchError::InvalidQuery {
            message: format!("Invalid sort format: {}", sort),
        }),
    }
}

fn escape_filter_value(value: &str) -> String {
    value.replace('"', "\\\"")
}

pub fn doc_to_meilisearch_doc(doc: Doc) -> Result<MeilisearchDoc, SearchError> {
    Ok(MeilisearchDoc {
        id: doc.id,
        content: doc.content,
    })
}

pub fn meilisearch_hit_to_search_hit(hit: MeilisearchHit) -> SearchHit {
    let id = hit.source
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let score = 1.0; // Meilisearch doesn't provide relevance scores in the same way

    SearchHit {
        id,
        source: hit.source,
        score: Some(score),
        highlight: hit.formatted,
    }
}

pub fn meilisearch_response_to_results(
    response: MeilisearchSearchResponse,
    query: &SearchQuery,
) -> SearchResults {
    let hits: Vec<SearchHit> = response
        .hits
        .into_iter()
        .map(meilisearch_hit_to_search_hit)
        .collect();

    let total = response.estimated_total_hits.unwrap_or(hits.len() as u64);
    let page = query.page.unwrap_or(1);
    let per_page = query.per_page.unwrap_or(20);

    // Convert facet distribution to the expected format
    let mut facets = HashMap::new();
    if let Some(facet_distribution) = response.facet_distribution {
        for (field, distribution) in facet_distribution {
            let facet_values: Vec<Value> = distribution
                .into_iter()
                .map(|(value, count)| json!({
                    "value": value,
                    "count": count
                }))
                .collect();
            facets.insert(field, facet_values);
        }
    }

    SearchResults {
        hits,
        total,
        page,
        per_page,
        total_pages: ((total as f64) / (per_page as f64)).ceil() as u32,
        processing_time_ms: Some(response.processing_time_ms),
        facets: if facets.is_empty() { None } else { Some(facets) },
    }
}

pub fn schema_to_meilisearch_settings(schema: &Schema) -> MeilisearchIndexSettings {
    let mut searchable_attributes = Vec::new();
    let mut filterable_attributes = Vec::new();
    let mut sortable_attributes = Vec::new();

    for field in &schema.fields {
        match field.field_type {
            FieldType::Text => {
                searchable_attributes.push(field.name.clone());
                if field.searchable.unwrap_or(true) {
                    // Already added to searchable
                }
                if field.filterable.unwrap_or(false) {
                    filterable_attributes.push(field.name.clone());
                }
                if field.sortable.unwrap_or(false) {
                    sortable_attributes.push(field.name.clone());
                }
            }
            FieldType::Number => {
                if field.searchable.unwrap_or(false) {
                    searchable_attributes.push(field.name.clone());
                }
                filterable_attributes.push(field.name.clone());
                sortable_attributes.push(field.name.clone());
            }
            FieldType::Boolean => {
                filterable_attributes.push(field.name.clone());
            }
            FieldType::Date => {
                filterable_attributes.push(field.name.clone());
                sortable_attributes.push(field.name.clone());
            }
            FieldType::Keyword => {
                filterable_attributes.push(field.name.clone());
                sortable_attributes.push(field.name.clone());
                if field.searchable.unwrap_or(false) {
                    searchable_attributes.push(field.name.clone());
                }
            }
            FieldType::Object => {
                // For objects, we'll add them as filterable by default
                filterable_attributes.push(field.name.clone());
            }
        }
    }

    MeilisearchIndexSettings {
        searchable_attributes: if searchable_attributes.is_empty() {
            None
        } else {
            Some(searchable_attributes)
        },
        filterable_attributes: if filterable_attributes.is_empty() {
            None
        } else {
            Some(filterable_attributes)
        },
        sortable_attributes: if sortable_attributes.is_empty() {
            None
        } else {
            Some(sortable_attributes)
        },
        ranking_rules: Some(vec![
            "words".to_string(),
            "typo".to_string(),
            "proximity".to_string(),
            "attribute".to_string(),
            "sort".to_string(),
            "exactness".to_string(),
        ]),
    }
}

pub fn meilisearch_settings_to_schema(settings: &MeilisearchIndexSettings) -> Schema {
    let mut fields = Vec::new();

    // Add searchable fields as text fields
    if let Some(searchable) = &settings.searchable_attributes {
        for field_name in searchable {
            if field_name != "*" {
                fields.push(SchemaField {
                    name: field_name.clone(),
                    field_type: FieldType::Text,
                    searchable: Some(true),
                    filterable: settings
                        .filterable_attributes
                        .as_ref()
                        .map(|f| f.contains(field_name))
                        .unwrap_or(false)
                        .into(),
                    sortable: settings
                        .sortable_attributes
                        .as_ref()
                        .map(|f| f.contains(field_name))
                        .unwrap_or(false)
                        .into(),
                    facetable: None,
                    config: None,
                });
            }
        }
    }

    // Add filterable-only fields
    if let Some(filterable) = &settings.filterable_attributes {
        for field_name in filterable {
            if !fields.iter().any(|f| f.name == *field_name) {
                fields.push(SchemaField {
                    name: field_name.clone(),
                    field_type: FieldType::Keyword, // Default assumption
                    searchable: Some(false),
                    filterable: Some(true),
                    sortable: settings
                        .sortable_attributes
                        .as_ref()
                        .map(|f| f.contains(field_name))
                        .unwrap_or(false)
                        .into(),
                    facetable: None,
                    config: None,
                });
            }
        }
    }

    Schema { fields }
}
