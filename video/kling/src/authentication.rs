use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hmac_sha256::HMAC;
use log::{debug, trace};
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

/// Generate JWT token for Kling API authentication
pub fn generate_jwt_token(access_key: &str, secret_key: &str) -> Result<String, String> {
    // Get current time in seconds since Unix epoch
    let now = get_current_time_seconds()?;

    trace!("Generating JWT token with timestamp: {}", now);

    // Create JWT header (HS256 algorithm)
    let header = json!({
        "alg": "HS256",
        "typ": "JWT"
    });

    // Create JWT payload/claims matching Python implementation exactly
    let payload = json!({
        "iss": access_key,
        "exp": now + 1800, // Valid for 30 minutes (like Python)
        "nbf": now.saturating_sub(5) // Effective 5 seconds ago
    });

    debug!(
        "JWT claims: iss={}, exp={}, nbf={}",
        access_key,
        now + 1800,
        now.saturating_sub(5)
    );

    // Encode header and payload to base64url
    let header_b64 = base64url_encode(
        &serde_json::to_vec(&header).map_err(|e| format!("Failed to serialize header: {}", e))?,
    );
    let payload_b64 = base64url_encode(
        &serde_json::to_vec(&payload).map_err(|e| format!("Failed to serialize payload: {}", e))?,
    );

    // Create the message to sign
    let message = format!("{}.{}", header_b64, payload_b64);

    // Create HMAC-SHA256 signature
    let signature = HMAC::mac(message.as_bytes(), secret_key.as_bytes());
    let signature_b64 = base64url_encode(&signature);

    // Combine all parts to create the final JWT
    let token = format!("{}.{}", message, signature_b64);

    // Print the generated token for debugging
    debug!("Generated JWT token: {}", token);
    println!("DEBUG: Generated JWT token: {}", token);

    Ok(token)
}

/// Get current time in seconds since Unix epoch
fn get_current_time_seconds() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|e| format!("Failed to get current time: {}", e))
}

/// Encode bytes to base64url (no padding)
fn base64url_encode(data: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(data)
}
