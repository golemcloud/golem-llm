wit_bindgen::generate!({world: "embed-library", path: "../../wit"});

use exports::golem::embed::embed::*;
use golem_embed::DurableEmbedWrapper;
use golem_embed_openai::OpenAIEmbedding;

struct Component;

impl Guest for Component {
    fn generate(inputs: Vec<ContentPart>, config: Config) -> Result<EmbeddingResponse, Error> {
        // 从环境变量获取API密钥
        let api_key = match std::env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                return Err(Error {
                    code: ErrorCode::InvalidRequest,
                    message: "环境变量OPENAI_API_KEY未设置".to_string(),
                    provider_error_json: None,
                });
            }
        };

        // 可选: 获取组织ID
        let organization_id = std::env::var("OPENAI_ORGANIZATION_ID").ok();
        
        // 初始化OpenAI嵌入模型
        let openai = match OpenAIEmbedding::new(api_key, organization_id) {
            Ok(client) => client,
            Err(err) => {
                return Err(err.to_wit_error());
            }
        };
        
        // 包装为支持持久性的实现
        let durable_openai = DurableEmbedWrapper::new(openai);
        
        // 调用通用接口方法
        durable_openai.generate(inputs, config)
    }

    fn rerank(query: String, documents: Vec<String>, config: Config) -> Result<RerankResponse, Error> {
        // 从环境变量获取API密钥
        let api_key = match std::env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                return Err(Error {
                    code: ErrorCode::InvalidRequest,
                    message: "环境变量OPENAI_API_KEY未设置".to_string(),
                    provider_error_json: None,
                });
            }
        };

        // 可选: 获取组织ID
        let organization_id = std::env::var("OPENAI_ORGANIZATION_ID").ok();
        
        // 初始化OpenAI嵌入模型
        let openai = match OpenAIEmbedding::new(api_key, organization_id) {
            Ok(client) => client,
            Err(err) => {
                return Err(err.to_wit_error());
            }
        };
        
        // 包装为支持持久性的实现
        let durable_openai = DurableEmbedWrapper::new(openai);
        
        // 调用通用接口方法
        durable_openai.rerank(query, documents, config)
    }
}

fn main() {
    // 初始化运行时
    golem_rust::init_logger();
    exports::golem::embed::embed::run_reactor();
} 