use golem_rust::durability::{DurabilityDirective, ProceduralDurability};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod types {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum TaskType {
        RetrievalQuery,
        RetrievalDocument,
        SemanticSimilarity,
        Classification,
        Clustering,
        QuestionAnswering,
        FactVerification,
        CodeRetrieval,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum OutputFormat {
        FloatArray,
        Binary,
        Base64,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum OutputDtype {
        Float32,
        Int8,
        Uint8,
        Binary,
        Ubinary,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum ErrorCode {
        InvalidRequest,
        ModelNotFound,
        Unsupported,
        ProviderError,
        RateLimitExceeded,
        InternalError,
        Unknown,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ImageUrl {
        pub url: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum ContentPart {
        Text(String),
        Image(ImageUrl),
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KeyValue {
        pub key: String,
        pub value: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Config {
        pub model: Option<String>,
        pub task_type: Option<TaskType>,
        pub dimensions: Option<u32>,
        pub truncation: Option<bool>,
        pub output_format: Option<OutputFormat>,
        pub output_dtype: Option<OutputDtype>,
        pub user: Option<String>,
        pub provider_options: Vec<KeyValue>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Usage {
        pub input_tokens: Option<u32>,
        pub total_tokens: Option<u32>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Embedding {
        pub index: u32,
        pub vector: Vec<f32>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingResponse {
        pub embeddings: Vec<Embedding>,
        pub usage: Option<Usage>,
        pub model: String,
        pub provider_metadata_json: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RerankResult {
        pub index: u32,
        pub relevance_score: f32,
        pub document: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RerankResponse {
        pub results: Vec<RerankResult>,
        pub usage: Option<Usage>,
        pub model: String,
        pub provider_metadata_json: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Error {
        pub code: ErrorCode,
        pub message: String,
        pub provider_error_json: Option<String>,
    }
}

#[derive(Error, Debug)]
pub enum EmbedError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("unsupported feature: {0}")]
    Unsupported(String),

    #[error("provider error: {0}")]
    ProviderError(String),

    #[error("rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    #[error("internal error: {0}")]
    InternalError(String),

    #[error("unknown error: {0}")]
    Unknown(String),

    #[error("reqwest error: {0}")]
    ReqwestError(#[from] reqwest::Error),

    #[error("serde json error: {0}")]
    SerdeJsonError(#[from] serde_json::Error),
}

impl EmbedError {
    pub fn to_wit_error(&self) -> types::Error {
        let (code, message) = match self {
            EmbedError::InvalidRequest(msg) => (types::ErrorCode::InvalidRequest, msg.clone()),
            EmbedError::ModelNotFound(msg) => (types::ErrorCode::ModelNotFound, msg.clone()),
            EmbedError::Unsupported(msg) => (types::ErrorCode::Unsupported, msg.clone()),
            EmbedError::ProviderError(msg) => (types::ErrorCode::ProviderError, msg.clone()),
            EmbedError::RateLimitExceeded(msg) => (types::ErrorCode::RateLimitExceeded, msg.clone()),
            EmbedError::InternalError(msg) => (types::ErrorCode::InternalError, msg.clone()),
            EmbedError::Unknown(msg) => (types::ErrorCode::Unknown, msg.clone()),
            EmbedError::ReqwestError(e) => (types::ErrorCode::ProviderError, e.to_string()),
            EmbedError::SerdeJsonError(e) => (types::ErrorCode::InternalError, e.to_string()),
        };

        types::Error {
            code,
            message,
            provider_error_json: None,
        }
    }
}

// 自定义耐久性处理
pub trait DurableEmbed {
    fn generate_durable(
        &self,
        inputs: Vec<types::ContentPart>,
        config: types::Config,
    ) -> Result<types::EmbeddingResponse, EmbedError>;

    fn rerank_durable(
        &self,
        query: String,
        documents: Vec<String>,
        config: types::Config,
    ) -> Result<types::RerankResponse, EmbedError>;
}

#[derive(Serialize, Deserialize)]
pub struct GenerateRequest {
    inputs: Vec<types::ContentPart>,
    config: types::Config,
}

#[derive(Serialize, Deserialize)]
pub struct RerankRequest {
    query: String,
    documents: Vec<String>,
    config: types::Config,
}

pub struct DurableEmbedWrapper<T: DurableEmbed> {
    pub embed_provider: T,
}

impl<T: DurableEmbed> DurableEmbedWrapper<T> {
    pub fn new(embed_provider: T) -> Self {
        Self { embed_provider }
    }

    pub fn generate(
        &self,
        inputs: Vec<types::ContentPart>,
        config: types::Config,
    ) -> Result<types::EmbeddingResponse, types::Error> {
        let request = GenerateRequest { 
            inputs: inputs.clone(),
            config: config.clone(),
        };

        let result = golem_rust::durability::with_durable_storage(
            DurabilityDirective::Mandatory,
            "generate_embedding",
            &request,
            || self.embed_provider.generate_durable(inputs, config),
        );

        match result {
            Ok(response) => Ok(response),
            Err(err) => Err(err.to_wit_error()),
        }
    }

    pub fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
        config: types::Config,
    ) -> Result<types::RerankResponse, types::Error> {
        let request = RerankRequest {
            query: query.clone(),
            documents: documents.clone(),
            config: config.clone(),
        };

        let result = golem_rust::durability::with_durable_storage(
            DurabilityDirective::Mandatory,
            "rerank",
            &request,
            || self.embed_provider.rerank_durable(query, documents, config),
        );

        match result {
            Ok(response) => Ok(response),
            Err(err) => Err(err.to_wit_error()),
        }
    }
}

// 辅助函数，用于从环境变量中获取API密钥
pub fn get_env_var(name: &str) -> Result<String, EmbedError> {
    std::env::var(name).map_err(|_| {
        EmbedError::InvalidRequest(format!("env var {} not set", name))
    })
} 