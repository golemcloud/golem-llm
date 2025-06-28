use crate::client::{TypesenseDoc, TypesenseField, TypesenseHit, TypesenseSearchResponse};
use golem_search::golem::search::types::{
    Doc, FieldType, Schema, SchemaField, SearchError, SearchHit, SearchQuery, SearchResults,
};
use log::trace;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub fn query_to_typesense_params(query: &SearchQuery) -> Result<HashMap<String, String>, SearchError> {
    let mut params = HashMap::new();
    
    // Query string
    if let Some(q) = &query.q {
        if !q.trim().is_empty() {
            params.insert("q".to_string(), q.clone());
            params.insert("query_by".to_string(), "*".to_string());
        }
    } else {
        params.insert("q".to_string(), "*".to_string());
    }
    
    // Filters
    if !query.filters.is_empty() {
        let mut filter_parts = Vec::new();
        
        for filter in &query.filters {
            if let Ok(filter_value) = serde_json::from_str::<Value>(filter) {
                // Handle JSON filters - convert to Typesense filter syntax
                if let Some(obj) = filter_value.as_object() {
                    for (field, conditions) in obj {
                        if let Some(cond_obj) = conditions.as_object() {
                            for (op, value) in cond_obj {
                                let filter_str = match op.as_str() {
                                    "eq" => format!("{}:={}", field, value_to_string(value)),
                                    "ne" => format!("{}:!={}", field, value_to_string(value)),
                                    "gt" => format!("{}:>{}", field, value_to_string(value)),
                                    "gte" => format!("{}:>={}", field, value_to_string(value)),
                                    "lt" => format!("{}:<{}", field, value_to_string(value)),
                                    "lte" => format!("{}:<={}", field, value_to_string(value)),
                                    "in" => {
                                        if let Some(arr) = value.as_array() {
                                            let values: Vec<String> = arr.iter().map(value_to_string).collect();
                                            format!("{}:[{}]", field, values.join(","))
                                        } else {
                                            continue;
                                        }
                                    }
                                    _ => continue,
                                };
                                filter_parts.push(filter_str);
                            }
                        }
                    }
                }
            } else {
                // Handle string filters in field:op:value format
                let parts: Vec<&str> = filter.splitn(3, ':').collect();
                if parts.len() == 3 {
                    let field = parts[0];
                    let op = parts[1];
                    let value = parts[2];

                    let filter_str = match op {
                        "eq" => format!("{}:={}", field, value),
                        "ne" => format!("{}:!={}", field, value),
                        "gt" => format!("{}:>{}", field, value),
                        "gte" => format!("{}:>={}", field, value),
                        "lt" => format!("{}:<{}", field, value),
                        "lte" => format!("{}:<={}", field, value),
                        "in" => {
                            let values: Vec<&str> = value.split(',').collect();
                            format!("{}:[{}]", field, values.join(","))
                        }
                        "exists" => format!("{}:!=''", field),
                        "prefix" => format!("{}:{}*", field, value),
                        "wildcard" => format!("{}:{}", field, value), // Typesense supports wildcards natively
                        _ => return Err(SearchError::InvalidQuery(format!("Unknown filter operator: {}", op)))
                    };
                    filter_parts.push(filter_str);
                } else {
                    return Err(SearchError::InvalidQuery(format!("Invalid filter format: {}", filter)));
                }
            }
        }
        
        if !filter_parts.is_empty() {
            params.insert("filter_by".to_string(), filter_parts.join(" && "));
        }
    }

    // Sorting
    if !query.sort.is_empty() {
        let mut sort_clauses = Vec::new();
        for sort_field in &query.sort {
            if sort_field.starts_with('-') {
                let field = &sort_field[1..];
                sort_clauses.push(format!("{}:desc", field));
            } else {
                sort_clauses.push(format!("{}:asc", sort_field));
            }
        }
        params.insert("sort_by".to_string(), sort_clauses.join(","));
    }

    // Pagination
    let per_page = query.per_page.unwrap_or(10);
    params.insert("per_page".to_string(), per_page.to_string());
    
    if let Some(page) = query.page {
        params.insert("page".to_string(), page.to_string());
    } else if let Some(offset) = query.offset {
        let page = (offset / per_page) + 1;
        params.insert("page".to_string(), page.to_string());
    }

    // Faceting
    if !query.facets.is_empty() {
        params.insert("facet_by".to_string(), query.facets.join(","));
    }

    // Highlighting
    if let Some(highlight) = &query.highlight {
        if !highlight.fields.is_empty() {
            params.insert("highlight_fields".to_string(), highlight.fields.join(","));
            
            if let Some(pre_tag) = &highlight.pre_tag {
                params.insert("highlight_start_tag".to_string(), pre_tag.clone());
            }
            
            if let Some(post_tag) = &highlight.post_tag {
                params.insert("highlight_end_tag".to_string(), post_tag.clone());
            }
            
            if let Some(max_length) = highlight.max_length {
                params.insert("snippet_threshold".to_string(), max_length.to_string());
            }
        }
    }

    // Configuration
    if let Some(config) = &query.config {
        if let Some(timeout_ms) = config.timeout_ms {
            params.insert("search_cutoff_ms".to_string(), timeout_ms.to_string());
        }
        
        if !config.attributes_to_retrieve.is_empty() {
            params.insert("include_fields".to_string(), config.attributes_to_retrieve.join(","));
        }
    }

    trace!("Generated Typesense parameters: {:?}", params);
    Ok(params)
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => value.to_string(),
    }
}

pub fn typesense_hit_to_search_hit(hit: TypesenseHit) -> SearchHit {
    let content = Some(serde_json::to_string(&hit.document).unwrap_or_default());
    
    let highlights = if !hit.highlights.is_empty() {
        let mut highlight_map = Map::new();
        for highlight in hit.highlights {
            highlight_map.insert(highlight.field, json!([highlight.snippet]));
        }
        Some(serde_json::to_string(&highlight_map).unwrap_or_default())
    } else {
        None
    };

    // Typesense doesn't return a traditional score, but we can use text_match as a proxy
    let score = hit.text_match.map(|tm| tm as f64 / 100.0);

    SearchHit {
        id: hit.document.get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        score,
        content,
        highlights,
    }
}

pub fn typesense_response_to_results(response: TypesenseSearchResponse, query: &SearchQuery) -> SearchResults {
    let hits = response.hits.into_iter().map(typesense_hit_to_search_hit).collect();
    let total = Some(response.found as u32);
    
    let page = query.page.or(response.page);
    let per_page = query.per_page;
    let took_ms = Some(response.search_time_ms as u32);

    SearchResults {
        total,
        page,
        per_page,
        hits,
        facets: None, // TODO: Add facets support
        took_ms,
    }
}

pub fn schema_to_typesense_fields(schema: &Schema) -> Vec<TypesenseField> {
    let mut fields = Vec::new();
    
    for field in &schema.fields {
        let field_type = match field.type_ {
            FieldType::Text => "string",
            FieldType::Keyword => "string",
            FieldType::Integer => "int64",
            FieldType::Float => "float",
            FieldType::Boolean => "bool",
            FieldType::Date => "string", // Typesense handles dates as strings with auto parsing
            FieldType::GeoPoint => "geopoint",
        };

        fields.push(TypesenseField {
            name: field.name.clone(),
            field_type: field_type.to_string(),
            facet: field.facet,
            index: field.index,
            sort: field.sort,
            optional: !field.required,
        });
    }
    
    fields
}

pub fn typesense_fields_to_schema(fields: &[TypesenseField]) -> Schema {
    let mut schema_fields = Vec::new();
    let mut primary_key = None;

    for field in fields {
        if field.name == ".*" {
            continue; // Skip auto fields
        }

        let field_type = match field.field_type.as_str() {
            "string" => FieldType::Text,
            "int32" | "int64" => FieldType::Integer,
            "float" => FieldType::Float,
            "bool" => FieldType::Boolean,
            "geopoint" => FieldType::GeoPoint,
            _ => FieldType::Text,
        };

        schema_fields.push(SchemaField {
            name: field.name.clone(),
            type_: field_type,
            required: !field.optional,
            facet: field.facet,
            sort: field.sort,
            index: field.index,
        });

        if field.name == "id" {
            primary_key = Some(field.name.clone());
        }
    }

    Schema {
        fields: schema_fields,
        primary_key,
    }
}

pub fn doc_to_typesense_doc(doc: Doc) -> Result<TypesenseDoc, SearchError> {
    let mut fields: Value = serde_json::from_str(&doc.content)
        .map_err(|e| SearchError::InvalidQuery(format!("Invalid JSON in document: {}", e)))?;
    
    // Ensure the id field is set
    if let Some(obj) = fields.as_object_mut() {
        obj.insert("id".to_string(), json!(doc.id.clone()));
    }
    
    Ok(TypesenseDoc {
        id: doc.id,
        fields,
    })
}

pub fn export_params_from_query(query: &SearchQuery) -> HashMap<String, String> {
    let mut params = HashMap::new();
    
    // Apply filters if any
    if !query.filters.is_empty() {
        if let Ok(ts_params) = query_to_typesense_params(query) {
            if let Some(filter_by) = ts_params.get("filter_by") {
                params.insert("filter_by".to_string(), filter_by.clone());
            }
        }
    }
    
    // For export, we want all fields unless specified otherwise
    if let Some(config) = &query.config {
        if !config.attributes_to_retrieve.is_empty() {
            params.insert("include_fields".to_string(), config.attributes_to_retrieve.join(","));
        }
    }
    
    params
}
