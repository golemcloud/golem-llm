use crate::client::{EsDoc, EsHit, EsSearchResponse, EsTotal};
use golem_search::golem::search::types::{
    Doc, FieldType, Schema, SchemaField, SearchError, SearchHit, SearchQuery, SearchResults,
};
use log::trace;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub fn query_to_es_query(query: &SearchQuery) -> Result<Value, SearchError> {
    let mut es_query = json!({});
    let mut bool_query = json!({});
    let mut must_clauses = Vec::new();
    let mut filter_clauses = Vec::new();

    if let Some(q) = &query.q {
        if !q.trim().is_empty() {
            must_clauses.push(json!({
                "multi_match": {
                    "query": q,
                    "type": "best_fields",
                    "fields": ["*"],
                    "fuzziness": "AUTO"
                }
            }));
        }
    }

    for filter in &query.filters {
        if let Ok(filter_value) = serde_json::from_str::<Value>(filter) {
            filter_clauses.push(filter_value);
        } else {
            let parts: Vec<&str> = filter.splitn(3, ':').collect();
            if parts.len() == 3 {
                let field = parts[0];
                let op = parts[1];
                let value = parts[2];

                let filter_clause = match op {
                    "eq" => json!({"term": {field: value}}),
                    "ne" => json!({"bool": {"must_not": {"term": {field: value}}}}),
                    "gt" => json!({"range": {field: {"gt": value}}}),
                    "gte" => json!({"range": {field: {"gte": value}}}),
                    "lt" => json!({"range": {field: {"lt": value}}}),
                    "lte" => json!({"range": {field: {"lte": value}}}),
                    "in" => {
                        let values: Vec<&str> = value.split(',').collect();
                        json!({"terms": {field: values}})
                    }
                    "exists" => json!({"exists": {"field": field}}),
                    "prefix" => json!({"prefix": {field: value}}),
                    "wildcard" => json!({"wildcard": {field: value}}),
                    _ => return Err(SearchError::InvalidQuery(format!("Unknown filter operator: {}", op)))
                };
                filter_clauses.push(filter_clause);
            } else {
                return Err(SearchError::InvalidQuery(format!("Invalid filter format: {}", filter)));
            }
        }
    }

    if !must_clauses.is_empty() || !filter_clauses.is_empty() {
        if !must_clauses.is_empty() {
            bool_query["must"] = json!(must_clauses);
        }
        if !filter_clauses.is_empty() {
            bool_query["filter"] = json!(filter_clauses);
        }
        es_query["query"] = json!({"bool": bool_query});
    } else {
        es_query["query"] = json!({"match_all": {}});
    }

    if !query.sort.is_empty() {
        let mut sort_clauses = Vec::new();
        for sort_field in &query.sort {
            if sort_field.starts_with('-') {
                let field = &sort_field[1..];
                sort_clauses.push(json!({field: {"order": "desc"}}));
            } else {
                sort_clauses.push(json!({sort_field: {"order": "asc"}}));
            }
        }
        es_query["sort"] = json!(sort_clauses);
    }

    if let Some(highlight) = &query.highlight {
        let mut highlight_config = json!({
            "fields": {}
        });
        
        for field in &highlight.fields {
            highlight_config["fields"][field] = json!({});
        }
        
        if let Some(pre_tag) = &highlight.pre_tag {
            highlight_config["pre_tags"] = json!([pre_tag]);
        }
        
        if let Some(post_tag) = &highlight.post_tag {
            highlight_config["post_tags"] = json!([post_tag]);
        }
        
        if let Some(max_length) = highlight.max_length {
            highlight_config["fragment_size"] = json!(max_length);
        }
        
        es_query["highlight"] = highlight_config;
    }

    let size = query.per_page.unwrap_or(10) as usize;
    es_query["size"] = json!(size);

    if let Some(page) = query.page {
        let from = ((page - 1) * query.per_page.unwrap_or(10)) as usize;
        es_query["from"] = json!(from);
    } else if let Some(offset) = query.offset {
        es_query["from"] = json!(offset);
    }

    if !query.facets.is_empty() {
        let mut aggs = json!({});
        for facet in &query.facets {
            aggs[facet] = json!({
                "terms": {
                    "field": facet
                }
            });
        }
        es_query["aggs"] = aggs;
    }

    if let Some(config) = &query.config {
        if let Some(timeout_ms) = config.timeout_ms {
            es_query["timeout"] = json!(format!("{}ms", timeout_ms));
        }
        
        if !config.attributes_to_retrieve.is_empty() {
            es_query["_source"] = json!(config.attributes_to_retrieve);
        }
    }

    trace!("Generated ES query: {}", serde_json::to_string_pretty(&es_query).unwrap_or_default());
    Ok(es_query)
}

pub fn es_hit_to_search_hit(hit: EsHit) -> SearchHit {
    let content = hit.source.map(|s| serde_json::to_string(&s).unwrap_or_default());
    
    let highlights = if let Some(highlight_map) = hit.highlight {
        let mut highlights = Map::new();
        for (field, fragments) in highlight_map {
            highlights.insert(field, json!(fragments));
        }
        Some(serde_json::to_string(&highlights).unwrap_or_default())
    } else {
        None
    };

    SearchHit {
        id: hit.id,
        score: hit.score,
        content,
        highlights,
    }
}

pub fn es_response_to_results(response: EsSearchResponse, query: &SearchQuery) -> SearchResults {
    let hits = response.hits.hits.into_iter().map(es_hit_to_search_hit).collect();
    let total = Some(response.hits.total.value() as u32);
    
    let page = query.page;
    let per_page = query.per_page;
    let took_ms = response.took.map(|t| t as u32);

    SearchResults {
        total,
        page,
        per_page,
        hits,
        facets: None,
        took_ms,
    }
}

pub fn schema_to_mapping(schema: &Schema) -> Value {
    let mut properties = json!({});
    
    for field in &schema.fields {
        let mut field_mapping = json!({});
        
        match field.type_ {
            FieldType::Text => {
                field_mapping["type"] = json!("text");
                if field.facet {
                    field_mapping["fields"] = json!({
                        "keyword": {
                            "type": "keyword"
                        }
                    });
                }
            }
            FieldType::Keyword => {
                field_mapping["type"] = json!("keyword");
            }
            FieldType::Integer => {
                field_mapping["type"] = json!("long");
            }
            FieldType::Float => {
                field_mapping["type"] = json!("double");
            }
            FieldType::Boolean => {
                field_mapping["type"] = json!("boolean");
            }
            FieldType::Date => {
                field_mapping["type"] = json!("date");
            }
            FieldType::GeoPoint => {
                field_mapping["type"] = json!("geo_point");
            }
        }
        
        if !field.index {
            field_mapping["index"] = json!(false);
        }
        
        properties[&field.name] = field_mapping;
    }
    
    json!({
        "properties": properties
    })
}

pub fn mapping_to_schema(mapping: &Value) -> Result<Schema, SearchError> {
    let mut fields = Vec::new();
    let mut primary_key = None;

    if let Some(index_mappings) = mapping.as_object() {
        for (index_name, index_mapping) in index_mappings {
            if let Some(mappings) = index_mapping.get("mappings") {
                if let Some(properties) = mappings.get("properties") {
                    if let Some(props) = properties.as_object() {
                        for (field_name, field_def) in props {
                            if let Some(field_type_str) = field_def.get("type").and_then(|v| v.as_str()) {
                                let field_type = match field_type_str {
                                    "text" => FieldType::Text,
                                    "keyword" => FieldType::Keyword,
                                    "long" | "integer" | "short" | "byte" => FieldType::Integer,
                                    "double" | "float" => FieldType::Float,
                                    "boolean" => FieldType::Boolean,
                                    "date" => FieldType::Date,
                                    "geo_point" => FieldType::GeoPoint,
                                    _ => FieldType::Text,
                                };

                                let index = field_def.get("index")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(true);

                                let facet = field_def.get("fields")
                                    .and_then(|v| v.get("keyword"))
                                    .is_some() || field_type == FieldType::Keyword;

                                fields.push(SchemaField {
                                    name: field_name.clone(),
                                    type_: field_type,
                                    required: false,
                                    facet,
                                    sort: true,
                                    index,
                                });

                                if field_name == "_id" || field_name == "id" {
                                    primary_key = Some(field_name.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(Schema {
        fields,
        primary_key,
    })
}

pub fn doc_to_es_doc(doc: Doc) -> Result<EsDoc, SearchError> {
    let content: Value = serde_json::from_str(&doc.content)
        .map_err(|e| SearchError::InvalidQuery(format!("Invalid JSON in document: {}", e)))?;
    
    Ok(EsDoc {
        id: doc.id,
        content,
    })
}
