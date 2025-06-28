use crate::client::{AlgoliaDoc, AlgoliaHit, AlgoliaSearchResponse};
use golem_search::golem::search::types::{
    Doc, FieldType, Schema, SchemaField, SearchError, SearchHit, SearchQuery, SearchResults,
};
use log::trace;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub fn query_to_algolia_query(query: &SearchQuery) -> Result<Value, SearchError> {
    let mut algolia_query = json!({});

    // Main query text
    if let Some(q) = &query.q {
        if !q.trim().is_empty() {
            algolia_query["query"] = json!(q);
        }
    }

    // Filters - Algolia uses a different syntax
    if !query.filters.is_empty() {
        let mut filter_parts = Vec::new();
        
        for filter in &query.filters {
            if let Ok(filter_value) = serde_json::from_str::<Value>(filter) {
                // Direct JSON filter
                if let Some(filter_str) = filter_value.as_str() {
                    filter_parts.push(filter_str.to_string());
                }
            } else {
                let parts: Vec<&str> = filter.splitn(3, ':').collect();
                if parts.len() == 3 {
                    let field = parts[0];
                    let op = parts[1];
                    let value = parts[2];

                    let filter_clause = match op {
                        "eq" => format!("{}:{}", field, value),
                        "ne" => format!("NOT {}:{}", field, value),
                        "gt" => format!("{} > {}", field, value),
                        "gte" => format!("{} >= {}", field, value),
                        "lt" => format!("{} < {}", field, value),
                        "lte" => format!("{} <= {}", field, value),
                        "in" => {
                            let values: Vec<&str> = value.split(',').collect();
                            format!("({})", values.iter()
                                .map(|v| format!("{}:{}", field, v))
                                .collect::<Vec<_>>()
                                .join(" OR "))
                        }
                        "exists" => format!("{} > 0", field), // Algolia doesn't have direct exists
                        "prefix" => format!("{}:{}", field, value), // Algolia handles prefix naturally
                        _ => return Err(SearchError::InvalidQuery(format!("Unknown filter operator: {}", op)))
                    };
                    filter_parts.push(filter_clause);
                } else {
                    return Err(SearchError::InvalidQuery(format!("Invalid filter format: {}", filter)));
                }
            }
        }
        
        if !filter_parts.is_empty() {
            algolia_query["filters"] = json!(filter_parts.join(" AND "));
        }
    }

    // Facets
    if !query.facets.is_empty() {
        algolia_query["facets"] = json!(query.facets);
    }

    // Pagination
    let hits_per_page = query.per_page.unwrap_or(10);
    algolia_query["hitsPerPage"] = json!(hits_per_page);

    if let Some(page) = query.page {
        algolia_query["page"] = json!(page - 1); // Algolia is 0-indexed
    } else if let Some(offset) = query.offset {
        let page = offset / hits_per_page;
        algolia_query["page"] = json!(page);
    }

    // Highlighting
    if let Some(highlight) = &query.highlight {
        let mut highlight_config = json!({});
        
        if !highlight.fields.is_empty() {
            highlight_config["attributesToHighlight"] = json!(highlight.fields);
        }
        
        if let Some(pre_tag) = &highlight.pre_tag {
            highlight_config["highlightPreTag"] = json!(pre_tag);
        }
        
        if let Some(post_tag) = &highlight.post_tag {
            highlight_config["highlightPostTag"] = json!(post_tag);
        }
        
        // Merge highlight config into main query
        if let Some(obj) = highlight_config.as_object() {
            for (key, value) in obj {
                algolia_query[key] = value.clone();
            }
        }
    }

    // Attributes to retrieve
    if let Some(config) = &query.config {
        if !config.attributes_to_retrieve.is_empty() {
            algolia_query["attributesToRetrieve"] = json!(config.attributes_to_retrieve);
        }
    }

    // Sorting - Algolia uses ranking/custom ranking
    if !query.sort.is_empty() {
        // Algolia requires pre-configured ranking attributes
        // For dynamic sorting, would need replica indexes
        algolia_query["customRanking"] = json!(query.sort);
    }

    trace!("Generated Algolia query: {}", serde_json::to_string_pretty(&algolia_query).unwrap_or_default());
    Ok(algolia_query)
}

pub fn algolia_hit_to_search_hit(hit: AlgoliaHit) -> SearchHit {
    // Extract content excluding objectID
    let mut content_value = hit.attributes.clone();
    if let Some(obj) = content_value.as_object_mut() {
        obj.remove("objectID");
    }
    let content = Some(serde_json::to_string(&content_value).unwrap_or_default());
    
    // Extract highlights
    let highlights = if let Some(highlight_map) = hit.highlight_result {
        let mut highlights = Map::new();
        for (field, highlight) in highlight_map {
            highlights.insert(field, json!(highlight.value));
        }
        Some(serde_json::to_string(&highlights).unwrap_or_default())
    } else {
        None
    };

    SearchHit {
        id: hit.object_id,
        score: None, // Algolia doesn't expose raw scores
        content,
        highlights,
    }
}

pub fn algolia_response_to_results(response: AlgoliaSearchResponse, query: &SearchQuery) -> SearchResults {
    let hits = response.hits.into_iter().map(algolia_hit_to_search_hit).collect();
    let total = Some(response.nb_hits as u32);
    
    let page = Some(response.page + 1); // Convert back to 1-indexed
    let per_page = Some(response.hits_per_page);
    let took_ms = Some(response.processing_time_ms);

    // Convert facets
    let facets = if let Some(algolia_facets) = response.facets {
        let mut facet_map = Map::new();
        for (facet_name, facet_values) in algolia_facets {
            let mut values = Map::new();
            for (value, count) in facet_values {
                values.insert(value, json!(count));
            }
            facet_map.insert(facet_name, json!(values));
        }
        Some(serde_json::to_string(&facet_map).unwrap_or_default())
    } else {
        None
    };

    SearchResults {
        total,
        page,
        per_page,
        hits,
        facets,
        took_ms,
    }
}

pub fn schema_to_algolia_settings(schema: &Schema) -> Value {
    let mut settings = json!({});
    let mut attributes_for_faceting = Vec::new();
    let mut searchable_attributes = Vec::new();
    let mut ranking = Vec::new();

    for field in &schema.fields {
        if field.index {
            searchable_attributes.push(&field.name);
        }
        
        if field.facet {
            match field.type_ {
                FieldType::Text | FieldType::Keyword => {
                    attributes_for_faceting.push(format!("filterOnly({})", field.name));
                }
                _ => {
                    attributes_for_faceting.push(field.name.clone());
                }
            }
        }
        
        if field.sort {
            ranking.push(format!("desc({})", field.name));
        }
    }

    if !searchable_attributes.is_empty() {
        settings["searchableAttributes"] = json!(searchable_attributes);
    }
    
    if !attributes_for_faceting.is_empty() {
        settings["attributesForFaceting"] = json!(attributes_for_faceting);
    }
    
    if !ranking.is_empty() {
        settings["customRanking"] = json!(ranking);
    }

    // Algolia default settings
    settings["attributesToHighlight"] = json!(["*"]);
    settings["attributesToSnippet"] = json!(["*:20"]);
    settings["hitsPerPage"] = json!(20);
    
    settings
}

pub fn algolia_settings_to_schema(settings: &Value) -> Result<Schema, SearchError> {
    let mut fields = Vec::new();
    let mut primary_key = Some("objectID".to_string());

    // Extract searchable attributes
    if let Some(searchable) = settings.get("searchableAttributes")
        .and_then(|v| v.as_array()) {
        for attr in searchable {
            if let Some(attr_name) = attr.as_str() {
                fields.push(SchemaField {
                    name: attr_name.to_string(),
                    type_: FieldType::Text, // Default to text for searchable
                    required: false,
                    facet: false,
                    sort: false,
                    index: true,
                });
            }
        }
    }

    // Extract facetable attributes
    if let Some(facets) = settings.get("attributesForFaceting")
        .and_then(|v| v.as_array()) {
        for facet in facets {
            if let Some(facet_str) = facet.as_str() {
                let field_name = if facet_str.starts_with("filterOnly(") {
                    facet_str.trim_start_matches("filterOnly(")
                        .trim_end_matches(")")
                        .to_string()
                } else {
                    facet_str.to_string()
                };

                // Check if field already exists
                if let Some(existing_field) = fields.iter_mut().find(|f| f.name == field_name) {
                    existing_field.facet = true;
                } else {
                    fields.push(SchemaField {
                        name: field_name,
                        type_: FieldType::Keyword, // Default to keyword for facets
                        required: false,
                        facet: true,
                        sort: false,
                        index: true,
                    });
                }
            }
        }
    }

    // Extract custom ranking (sort fields)
    if let Some(ranking) = settings.get("customRanking")
        .and_then(|v| v.as_array()) {
        for rank in ranking {
            if let Some(rank_str) = rank.as_str() {
                let field_name = if rank_str.starts_with("desc(") {
                    rank_str.trim_start_matches("desc(")
                        .trim_end_matches(")")
                        .to_string()
                } else if rank_str.starts_with("asc(") {
                    rank_str.trim_start_matches("asc(")
                        .trim_end_matches(")")
                        .to_string()
                } else {
                    rank_str.to_string()
                };

                // Check if field already exists
                if let Some(existing_field) = fields.iter_mut().find(|f| f.name == field_name) {
                    existing_field.sort = true;
                } else {
                    fields.push(SchemaField {
                        name: field_name,
                        type_: FieldType::Integer, // Assume numeric for ranking
                        required: false,
                        facet: false,
                        sort: true,
                        index: true,
                    });
                }
            }
        }
    }

    // Add objectID field if not present
    if !fields.iter().any(|f| f.name == "objectID") {
        fields.insert(0, SchemaField {
            name: "objectID".to_string(),
            type_: FieldType::Keyword,
            required: true,
            facet: false,
            sort: false,
            index: false,
        });
    }

    Ok(Schema {
        fields,
        primary_key,
    })
}

pub fn doc_to_algolia_doc(doc: Doc) -> Result<AlgoliaDoc, SearchError> {
    let mut content: Value = serde_json::from_str(&doc.content)
        .map_err(|e| SearchError::InvalidQuery(format!("Invalid JSON in document: {}", e)))?;
    
    // Ensure objectID is set
    if let Some(obj) = content.as_object_mut() {
        obj.insert("objectID".to_string(), json!(doc.id));
    }
    
    Ok(AlgoliaDoc {
        object_id: doc.id,
        content,
    })
}
