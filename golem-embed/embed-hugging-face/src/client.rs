use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub input: Vec<String>,
    pub normalize: Option<bool>,
    /// The name of the prompt that should be used by for encoding.
    /// If not set, no prompt will be applied. Must be a key in the
    /// sentence-transformers configuration prompts dictionary.
    /// For example if prompt_name is “query” and the prompts is {“query”: “query: ”, …},
    /// then the sentence “What is the capital of France?” will be encoded as
    /// “query: What is the capital of France?” because the prompt text will
    /// be prepended before any text to encode.
    pub prompt_name: Option<String>,
    pub truncate: Option<bool>,
    pub truncate_direction: Option<TruncateDirection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TruncateDirection {
    #[serde(rename = "left")]
    Left,
    #[serde(rename = "right")]
    Right,
}
