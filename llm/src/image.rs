//! Image processing and validation module for golem-llm
//!
//! This module provides functionality for processing and validating images before they are sent to LLM providers.
//! It includes features such as:
//! - Image size validation
//! - MIME type validation
//! - Automatic image resizing for large images
//! - Image format conversion
//! - Compression and optimization
//!
//! # Examples
//!
//! ```rust
//! use golem_llm::image::{ImageProcessor, ImageError};
//!
//! let processor = ImageProcessor::default();
//!
//! // Create some sample image data
//! let image_data = vec![0u8; 100]; // Sample data
//!
//! // Process an image
//! match processor.validate_and_process(&image_data, "image/jpeg") {
//!     Ok(processed_data) => {
//!         // Use the processed image data
//!     },
//!     Err(ImageError::TooLarge(size, max)) => {
//!         println!("Image too large: {} bytes (max: {} bytes)", size, max);
//!     },
//!     Err(ImageError::InvalidMimeType(mime)) => {
//!         println!("Unsupported MIME type: {}", mime);
//!     },
//!     Err(e) => {
//!         println!("Error processing image: {}", e);
//!     }
//! }
//! ```

use image::GenericImageView;
use image::{DynamicImage, ImageFormat};
use std::io::Cursor;
use thiserror::Error;

/// Maximum allowed image size (10MB)
pub const MAX_IMAGE_SIZE: usize = 10 * 1024 * 1024;

/// Supported MIME types for images
pub const SUPPORTED_MIME_TYPES: &[&str] = &[
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/svg+xml",
];

/// Errors that can occur during image processing
#[derive(Error, Debug)]
pub enum ImageError {
    /// Invalid image format
    #[error("Invalid image format: {0}")]
    InvalidFormat(String),

    /// Image exceeds maximum allowed size
    #[error("Image too large: {0} bytes (max: {1} bytes)")]
    TooLarge(usize, usize),

    /// Unsupported MIME type
    #[error("Invalid MIME type: {0}")]
    InvalidMimeType(String),

    /// Error during image processing
    #[error("Image processing error: {0}")]
    ProcessingError(String),
}

/// Image processor for validating and processing images
///
/// This struct provides methods for validating and processing images before they are sent to LLM providers.
/// It handles tasks such as:
/// - Validating image size and format
/// - Resizing large images
/// - Converting between formats
/// - Optimizing image quality
pub struct ImageProcessor {
    /// Maximum allowed image size in bytes
    max_size: usize,
    /// Compression quality (0-100)
    #[allow(dead_code)]
    compression_quality: u8,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self {
            max_size: MAX_IMAGE_SIZE,
            compression_quality: 85,
        }
    }
}

impl ImageProcessor {
    /// Creates a new image processor with custom settings
    ///
    /// # Arguments
    ///
    /// * `max_size` - Maximum allowed image size in bytes
    /// * `compression_quality` - Compression quality (0-100)
    pub fn new(max_size: usize, compression_quality: u8) -> Self {
        Self {
            max_size,
            compression_quality,
        }
    }

    /// Validates and processes an image
    ///
    /// This method performs the following operations:
    /// 1. Validates the image size
    /// 2. Validates the MIME type
    /// 3. Loads and validates the image format
    /// 4. Processes the image (resize if needed)
    /// 5. Encodes the image in the appropriate format
    ///
    /// # Arguments
    ///
    /// * `data` - Raw image data
    /// * `mime_type` - MIME type of the image
    ///
    /// # Returns
    ///
    /// Returns the processed image data if successful, or an error if validation or processing fails
    pub fn validate_and_process(
        &self,
        data: &[u8],
        mime_type: &str,
    ) -> Result<Vec<u8>, ImageError> {
        // Validate size
        if data.len() > self.max_size {
            return Err(ImageError::TooLarge(data.len(), self.max_size));
        }

        // Validate MIME type
        if !SUPPORTED_MIME_TYPES.contains(&mime_type) {
            return Err(ImageError::InvalidMimeType(mime_type.to_string()));
        }

        // Load and validate image
        let img = image::load_from_memory(data)
            .map_err(|e| ImageError::ProcessingError(e.to_string()))?;

        // Process image (resize if too large, optimize)
        let processed = self.process_image(img)?;

        // Encode back to bytes
        let mut bytes = Vec::new();
        processed
            .write_to(
                &mut Cursor::new(&mut bytes),
                self.get_output_format(mime_type),
            )
            .map_err(|e| ImageError::ProcessingError(e.to_string()))?;

        Ok(bytes)
    }

    /// Processes an image, performing necessary transformations
    ///
    /// Currently handles:
    /// - Resizing large images to a maximum dimension of 2048 pixels
    /// - Maintaining aspect ratio during resizing
    /// - Using high-quality Lanczos3 filter for resizing
    fn process_image(&self, img: DynamicImage) -> Result<DynamicImage, ImageError> {
        // Get dimensions
        let (width, height) = img.dimensions();

        // Resize if too large (e.g., if either dimension > 2048)
        let processed = if width > 2048 || height > 2048 {
            let scale = 2048.0 / width.max(height) as f32;
            let new_width = (width as f32 * scale) as u32;
            let new_height = (height as f32 * scale) as u32;
            img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
        } else {
            img
        };

        Ok(processed)
    }

    /// Gets the appropriate output format for a given MIME type
    fn get_output_format(&self, mime_type: &str) -> ImageFormat {
        match mime_type {
            "image/jpeg" => ImageFormat::Jpeg,
            "image/png" => ImageFormat::Png,
            "image/gif" => ImageFormat::Gif,
            "image/webp" => ImageFormat::WebP,
            _ => ImageFormat::Png, // Default to PNG for other formats
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_validation() {
        let processor = ImageProcessor::default();

        // Test with valid image
        let valid_image = include_bytes!("../test_data/valid_image.jpg");
        let result = processor.validate_and_process(valid_image, "image/jpeg");
        assert!(result.is_ok());

        // Test with invalid MIME type
        let result = processor.validate_and_process(valid_image, "image/invalid");
        assert!(matches!(result, Err(ImageError::InvalidMimeType(_))));

        // Test with too large image
        let large_image = vec![0u8; MAX_IMAGE_SIZE + 1];
        let result = processor.validate_and_process(&large_image, "image/jpeg");
        assert!(matches!(result, Err(ImageError::TooLarge(_, _))));
    }
}
