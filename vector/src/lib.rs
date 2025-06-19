//! Placeholder implementation of the `golem:vector` interfaces.
//! This crate only defines the bindings without providing a working
//! vector database implementation.

use golem_rust::export;

wit_bindgen::generate!("../wit/vector.wit");

struct VectorComponent;

impl golem_vector::connection::Guest for VectorComponent {
    fn connect(
        _endpoint: String,
        _credentials: Option<golem_vector::connection::Credentials>,
        _timeout_ms: Option<u32>,
        _options: Option<golem_vector::types::Metadata>,
    ) -> Result<(), golem_vector::types::VectorError> {
        Err(golem_vector::types::VectorError::UnsupportedFeature(
            "not implemented".to_string(),
        ))
    }

    fn disconnect() -> Result<(), golem_vector::types::VectorError> {
        Ok(())
    }

    fn get_connection_status(
    ) -> Result<golem_vector::connection::ConnectionStatus, golem_vector::types::VectorError> {
        Ok(golem_vector::connection::ConnectionStatus {
            connected: false,
            provider: None,
            endpoint: None,
            last_activity: None,
            connection_id: None,
        })
    }

    fn test_connection(
        _endpoint: String,
        _credentials: Option<golem_vector::connection::Credentials>,
        _timeout_ms: Option<u32>,
        _options: Option<golem_vector::types::Metadata>,
    ) -> Result<bool, golem_vector::types::VectorError> {
        Ok(false)
    }
}

export!(VectorComponent);
