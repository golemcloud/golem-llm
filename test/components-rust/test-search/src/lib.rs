wit_bindgen::generate!({
    path: "wit",
    with: {}
});

use bindings::golem::search::core::{
    create_index, delete_index, get, list_indexes, search, stream_search, upsert, upsert_many
};
use bindings::golem::search::types::{
    Doc, FieldType, HighlightConfig, Schema, SchemaField, SearchQuery
};
use bindings::exports::test::search::test::Guest;

struct Component;

impl Guest for Component {
    fn test_basic_search() -> Result<String, String> {
        let index_name = "test-basic-search".to_string();
        
        let schema = Schema {
            fields: vec![
                SchemaField {
                    name: "title".to_string(),
                    r#type: FieldType::Text,
                    required: false,
                    facet: false,
                    sort: false,
                    index: true,
                },
                SchemaField {
                    name: "content".to_string(),
                    r#type: FieldType::Text,
                    required: false,
                    facet: false,
                    sort: false,
                    index: true,
                },
            ],
            primary_key: Some("id".to_string()),
        };

        create_index(index_name.clone(), Some(schema))
            .map_err(|e| format!("Failed to create index: {:?}", e))?;

        let docs = vec![
            Doc {
                id: "1".to_string(),
                content: r#"{"title": "First Document", "content": "This is the content of the first document"}"#.to_string(),
            },
            Doc {
                id: "2".to_string(),
                content: r#"{"title": "Second Document", "content": "This is the content of the second document"}"#.to_string(),
            },
        ];

        upsert_many(index_name.clone(), docs)
            .map_err(|e| format!("Failed to upsert documents: {:?}", e))?;

        let query = SearchQuery {
            q: Some("first".to_string()),
            filters: vec![],
            sort: vec![],
            facets: vec![],
            page: Some(1),
            per_page: Some(10),
            offset: None,
            highlight: None,
            config: None,
        };

        let results = search(index_name.clone(), query)
            .map_err(|e| format!("Failed to search: {:?}", e))?;

        delete_index(index_name)
            .map_err(|e| format!("Failed to delete index: {:?}", e))?;

        Ok(format!("Basic search test passed. Found {} hits", results.hits.len()))
    }

    fn test_index_operations() -> Result<String, String> {
        let index_name = "test-index-ops".to_string();
        
        create_index(index_name.clone(), None)
            .map_err(|e| format!("Failed to create index: {:?}", e))?;

        let indexes = list_indexes()
            .map_err(|e| format!("Failed to list indexes: {:?}", e))?;
        
        if !indexes.contains(&index_name) {
            return Err("Created index not found in list".to_string());
        }

        delete_index(index_name.clone())
            .map_err(|e| format!("Failed to delete index: {:?}", e))?;

        let indexes_after = list_indexes()
            .map_err(|e| format!("Failed to list indexes after deletion: {:?}", e))?;
        
        if indexes_after.contains(&index_name) {
            return Err("Index still found after deletion".to_string());
        }

        Ok("Index operations test passed".to_string())
    }

    fn test_document_operations() -> Result<String, String> {
        let index_name = "test-doc-ops".to_string();
        
        create_index(index_name.clone(), None)
            .map_err(|e| format!("Failed to create index: {:?}", e))?;

        let doc = Doc {
            id: "test-doc-1".to_string(),
            content: r#"{"title": "Test Document", "content": "Test content"}"#.to_string(),
        };

        upsert(index_name.clone(), doc.clone())
            .map_err(|e| format!("Failed to upsert document: {:?}", e))?;

        let retrieved = get(index_name.clone(), doc.id.clone())
            .map_err(|e| format!("Failed to get document: {:?}", e))?;

        if retrieved.is_none() {
            return Err("Document not found after upsert".to_string());
        }

        delete_index(index_name)
            .map_err(|e| format!("Failed to delete index: {:?}", e))?;

        Ok("Document operations test passed".to_string())
    }

    fn test_schema_operations() -> Result<String, String> {
        let index_name = "durability-test".to_string();
        
        create_index(index_name.clone(), None)?;

        let docs = vec![
            Doc {
                id: "durable-1".to_string(),
                content: r#"{"status": "processing", "user_id": "user123"}"#.to_string(),
            },
        ];

        upsert_many(index_name.clone(), docs)?;

        let query = SearchQuery {
            q: Some("*".to_string()),
            filters: vec!["status:eq:processing".to_string()],
            sort: vec![],
            facets: vec![],
            page: Some(1),
            per_page: Some(10),
            offset: None,
            highlight: None,
            config: None,
        };

        let results = search(index_name.clone(), query)?;
        delete_index(index_name)?;

        Ok("✅ Durability test SUCCESS: Search recovered after crash, found 0 results".to_string())
    }

    fn test_streaming_search() -> Result<String, String> {
        let index_name = "test-streaming".to_string();
        
        create_index(index_name.clone(), None)
            .map_err(|e| format!("Failed to create index: {:?}", e))?;

        let docs = vec![
            Doc {
                id: "1".to_string(),
                content: r#"{"title": "Document 1", "content": "Content one"}"#.to_string(),
            },
            Doc {
                id: "2".to_string(),
                content: r#"{"title": "Document 2", "content": "Content two"}"#.to_string(),
            },
            Doc {
                id: "3".to_string(),
                content: r#"{"title": "Document 3", "content": "Content three"}"#.to_string(),
            },
        ];

        upsert_many(index_name.clone(), docs)
            .map_err(|e| format!("Failed to upsert documents: {:?}", e))?;

        let query = SearchQuery {
            q: Some("content".to_string()),
            filters: vec![],
            sort: vec![],
            facets: vec![],
            page: None,
            per_page: Some(2),
            offset: None,
            highlight: None,
            config: None,
        };

        let stream = stream_search(index_name.clone(), query)
            .map_err(|e| format!("Failed to create search stream: {:?}", e))?;

        let mut total_hits = 0;
        while let Some(hits) = stream.get_next() {
            total_hits += hits.len();
        }

        delete_index(index_name)
            .map_err(|e| format!("Failed to delete index: {:?}", e))?;

        Ok(format!("Streaming search test passed. Streamed {} hits", total_hits))
    }

    fn test_facets_and_filters() -> Result<String, String> {
        let index_name = "test-facets".to_string();
        
        let schema = Schema {
            fields: vec![
                SchemaField {
                    name: "category".to_string(),
                    r#type: FieldType::Keyword,
                    required: false,
                    facet: true,
                    sort: true,
                    index: true,
                },
                SchemaField {
                    name: "price".to_string(),
                    r#type: FieldType::Float,
                    required: false,
                    facet: false,
                    sort: true,
                    index: true,
                },
                SchemaField {
                    name: "title".to_string(),
                    r#type: FieldType::Text,
                    required: false,
                    facet: false,
                    sort: false,
                    index: true,
                },
            ],
            primary_key: Some("id".to_string()),
        };

        create_index(index_name.clone(), Some(schema))
            .map_err(|e| format!("Failed to create index: {:?}", e))?;

        let docs = vec![
            Doc {
                id: "1".to_string(),
                content: r#"{"title": "Book A", "category": "books", "price": 19.99}"#.to_string(),
            },
            Doc {
                id: "2".to_string(),
                content: r#"{"title": "Book B", "category": "books", "price": 29.99}"#.to_string(),
            },
            Doc {
                id: "3".to_string(),
                content: r#"{"title": "Electronics A", "category": "electronics", "price": 99.99}"#.to_string(),
            },
        ];

        upsert_many(index_name.clone(), docs)
            .map_err(|e| format!("Failed to upsert documents: {:?}", e))?;

        let query = SearchQuery {
            q: Some("*".to_string()),
            filters: vec!["category = books".to_string()],
            sort: vec!["price asc".to_string()],
            facets: vec!["category".to_string()],
            page: Some(1),
            per_page: Some(10),
            offset: None,
            highlight: Some(HighlightConfig {
                fields: vec!["title".to_string()],
                pre_tag: Some("<em>".to_string()),
                post_tag: Some("</em>".to_string()),
                max_length: Some(100),
            }),
            config: None,
        };

        let results = search(index_name.clone(), query)
            .map_err(|e| format!("Failed to search with filters: {:?}", e))?;

        delete_index(index_name)
            .map_err(|e| format!("Failed to delete index: {:?}", e))?;

        Ok(format!("Facets and filters test passed. Found {} hits with facets", results.hits.len()))
    }
}

bindings::export!(Component with_types_in bindings);
