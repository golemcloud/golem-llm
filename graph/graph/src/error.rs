use crate::golem::graph::errors::GraphError;

pub fn unsupported_operation<T>(message: &str) -> Result<T, GraphError> {
    Err(GraphError::UnsupportedOperation(message.to_string()))
}

pub fn internal_error<T>(message: &str) -> Result<T, GraphError> {
    Err(GraphError::InternalError(message.to_string()))
}

impl<'a> From<&'a GraphError> for GraphError {
    fn from(e: &'a GraphError) -> GraphError {
        e.clone()
    }
}
