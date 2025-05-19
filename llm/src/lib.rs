pub mod chat_stream;
pub mod config;
pub mod durability;
pub mod error;

#[allow(dead_code)]
pub mod event_source;

pub mod image;
use image::{ImageError, ImageProcessor};
use std::sync::OnceLock;

wit_bindgen::generate!({
    path: "../wit",
    world: "llm-library",
    generate_all,
    generate_unused_types: true,
    additional_derives: [PartialEq],
    pub_export_macro: true,
});

pub use crate::exports::golem;
use crate::golem::llm::llm::ImageDetail;
pub use __export_llm_library_impl as export_llm;
use std::cell::RefCell;
use std::str::FromStr;

pub struct LoggingState {
    logging_initialized: bool,
}

impl LoggingState {
    /// Initializes WASI logging based on the `GOLEM_LLM_LOG` environment variable.
    pub fn init(&mut self) {
        if !self.logging_initialized {
            let _ = wasi_logger::Logger::install();
            let max_level: log::LevelFilter =
                log::LevelFilter::from_str(&std::env::var("GOLEM_LLM_LOG").unwrap_or_default())
                    .unwrap_or(log::LevelFilter::Info);
            log::set_max_level(max_level);
            self.logging_initialized = true;
        }
    }
}

thread_local! {
    /// This holds the state of our application.
    pub static LOGGING_STATE: RefCell<LoggingState> = const { RefCell::new(LoggingState {
        logging_initialized: false,
    }) };
}

static IMAGE_PROCESSOR: OnceLock<ImageProcessor> = OnceLock::new();

fn get_image_processor() -> &'static ImageProcessor {
    IMAGE_PROCESSOR.get_or_init(ImageProcessor::default)
}

pub fn process_image_data(data: &[u8], mime_type: &str) -> Result<Vec<u8>, ImageError> {
    get_image_processor().validate_and_process(data, mime_type)
}

// Update the ImageData struct to use our processor
pub struct ImageData {
    pub data: Vec<u8>,
    pub mime_type: String,
    pub detail: Option<ImageDetail>,
}

impl ImageData {
    pub fn new(
        data: Vec<u8>,
        mime_type: String,
        detail: Option<ImageDetail>,
    ) -> Result<Self, ImageError> {
        // Process the image data
        let processed_data = process_image_data(&data, &mime_type)?;

        Ok(Self {
            data: processed_data,
            mime_type,
            detail,
        })
    }
}
