//! Qdrant vector database component placeholder.
//! This does not implement actual functionality yet.

use golem_vector::golem::vector::connection::{self, Guest as ConnGuest};
use golem_vector::golem::vector::types::VectorError;

struct QdrantComponent;

impl ConnGuest for QdrantComponent {
    fn connect(
        _endpoint: String,
        _credentials: Option<connection::Credentials>,
        _timeout_ms: Option<u32>,
        _options: Option<golem_vector::golem::vector::types::Metadata>,
    ) -> Result<(), VectorError> {
        Err(VectorError::UnsupportedFeature("not implemented".to_string()))
    }

    fn disconnect() -> Result<(), VectorError> {
        Ok(())
    }

    fn get_connection_status(
    ) -> Result<connection::ConnectionStatus, VectorError> {
        Ok(connection::ConnectionStatus {
            connected: false,
            provider: Some("qdrant".to_string()),
            endpoint: None,
            last_activity: None,
            connection_id: None,
        })
    }

    fn test_connection(
        _endpoint: String,
        _credentials: Option<connection::Credentials>,
        _timeout_ms: Option<u32>,
        _options: Option<golem_vector::golem::vector::types::Metadata>,
    ) -> Result<bool, VectorError> {
        Ok(false)
    }
}

wit_bindgen::export!(QdrantComponent);
