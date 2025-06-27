use crate::error::internal_error;
use crate::exports::golem::video::video::VideoError;

/// Downloads an image from a URL and returns the bytes
pub fn download_image_from_url(url: &str) -> Result<Vec<u8>, VideoError> {
    use reqwest::Client;

    let client = Client::builder()
        .build()
        .map_err(|err| internal_error(format!("Failed to create HTTP client: {err}")))?;

    let response = client
        .get(url)
        .send()
        .map_err(|err| internal_error(format!("Failed to download image from {url}: {err}")))?;

    if !response.status().is_success() {
        return Err(internal_error(format!(
            "Failed to download image from {}: HTTP {}",
            url,
            response.status()
        )));
    }

    let bytes = response
        .bytes()
        .map_err(|err| internal_error(format!("Failed to read image data from {url}: {err}")))?;

    Ok(bytes.to_vec())
}
