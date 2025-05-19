use golem_llm::image::{ImageError, ImageProcessor};

#[test]
fn test_invalid_mime_type() {
    let processor = ImageProcessor::default();
    let image_data = vec![0u8; 100];

    let result = processor.validate_and_process(&image_data, "image/invalid");
    assert!(matches!(result, Err(ImageError::InvalidMimeType(_))));
}

#[test]
fn test_too_large_image() {
    let processor = ImageProcessor::default();
    let image_data = vec![0u8; golem_llm::image::MAX_IMAGE_SIZE + 1];

    let result = processor.validate_and_process(&image_data, "image/jpeg");
    assert!(matches!(result, Err(ImageError::TooLarge(_, _))));
}

#[test]
fn test_valid_image_mime_types() {
    let processor = ImageProcessor::default();

    // A small valid byte array
    let small_image_data = vec![0u8; 10];

    // Try each of the supported MIME types with validation only (no actual processing)
    for mime_type in ["image/jpeg", "image/png", "image/gif", "image/webp"] {
        // We expect this to fail with a ProcessingError because our data isn't a valid image
        // but it should pass the MIME type validation
        let result = processor.validate_and_process(&small_image_data, mime_type);
        assert!(matches!(result, Err(ImageError::ProcessingError(_))));

        // It should not fail with InvalidMimeType
        assert!(!matches!(result, Err(ImageError::InvalidMimeType(_))));

        // It should not fail with TooLarge
        assert!(!matches!(result, Err(ImageError::TooLarge(_, _))));
    }
}

#[test]
fn test_custom_processor_settings() {
    // Create processor with custom settings
    let processor = ImageProcessor::new(1024 * 1024, 90); // 1MB max size, 90% quality

    // Test with image just under the custom size limit
    let image_data_under_limit = vec![0u8; 1024 * 1024 - 1];
    let result_under = processor.validate_and_process(&image_data_under_limit, "image/jpeg");

    // This should fail but with a ProcessingError, not a TooLarge error
    assert!(matches!(result_under, Err(ImageError::ProcessingError(_))));

    // Test with image just over the custom size limit
    let image_data_over_limit = vec![0u8; 1024 * 1024 + 1];
    let result_over = processor.validate_and_process(&image_data_over_limit, "image/jpeg");

    // This should fail with a TooLarge error
    assert!(matches!(result_over, Err(ImageError::TooLarge(_, _))));
}
