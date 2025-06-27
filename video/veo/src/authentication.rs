use data_encoding::BASE64URL_NOPAD;
use golem_video::error::internal_error;
use golem_video::exports::golem::video::video::VideoError;
use log::{debug, trace};
use rsa::pkcs8::DecodePrivateKey;
use rsa::RsaPrivateKey;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

/// SHA-256 DigestInfo prefix for PKCS#1 v1.5 signatures (RFC 8017)
const SHA256_PREFIX: &[u8] = &[
    0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05,
    0x00, 0x04, 0x20,
];

/// Generate GCP access token using service account credentials
pub fn generate_access_token(
    client_email: &str,
    private_key_pem: &str,
    scope: &str,
) -> Result<String, VideoError> {
    trace!("Generating GCP access token for client: {client_email}");

    // Step 1: Generate JWT
    let jwt = generate_jwt(client_email, private_key_pem, scope)?;

    // Step 2: Exchange JWT for access token
    exchange_jwt_for_token(&jwt)
}

/// Generate a signed JWT for GCP authentication
fn generate_jwt(
    client_email: &str,
    private_key_pem: &str,
    scope: &str,
) -> Result<String, VideoError> {
    // Convert literal \n characters to actual newlines (like echo -e in bash)
    let processed_key = private_key_pem.replace("\\n", "\n");

    // Parse the private key
    let private_key = RsaPrivateKey::from_pkcs8_pem(&processed_key)
        .map_err(|e| internal_error(format!("Failed to parse private key: {e}")))?;

    // Get current time
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| internal_error(format!("Failed to get current time: {e}")))?
        .as_secs();

    // Create JWT header
    let header = json!({
        "alg": "RS256",
        "typ": "JWT"
    });

    // Create JWT payload
    let payload = json!({
        "iss": client_email,
        "scope": scope,
        "aud": "https://oauth2.googleapis.com/token",
        "iat": now,
        "exp": now + 3600  // Valid for 1 hour
    });

    // Encode header and payload
    let encoded_header = BASE64URL_NOPAD.encode(
        &serde_json::to_vec(&header)
            .map_err(|e| internal_error(format!("Failed to serialize header: {e}")))?,
    );

    let encoded_payload = BASE64URL_NOPAD.encode(
        &serde_json::to_vec(&payload)
            .map_err(|e| internal_error(format!("Failed to serialize payload: {e}")))?,
    );

    // Create signing input
    let signing_input = format!("{encoded_header}.{encoded_payload}");

    // Hash the signing input
    let mut hasher = Sha256::new();
    hasher.update(signing_input.as_bytes());
    let hash = hasher.finalize();

    // Create DigestInfo structure for PKCS#1 v1.5 (ASN.1 DER encoded)
    let mut digest_info = Vec::new();
    digest_info.extend_from_slice(SHA256_PREFIX);
    digest_info.extend_from_slice(&hash);

    // Sign using PKCS#1 v1.5 padding with manual DigestInfo
    use rsa::pkcs1v15::Pkcs1v15Sign;
    let padding = Pkcs1v15Sign::new_unprefixed();
    let signature = private_key
        .sign(padding, &digest_info)
        .map_err(|e| internal_error(format!("Failed to sign JWT: {e}")))?;

    let encoded_signature = BASE64URL_NOPAD.encode(&signature);

    // Assemble final JWT
    let jwt = format!("{signing_input}.{encoded_signature}");

    debug!("Generated JWT token for GCP authentication");
    Ok(jwt)
}

/// Exchange JWT for GCP access token
fn exchange_jwt_for_token(jwt: &str) -> Result<String, VideoError> {
    use reqwest::Client;

    let client = Client::builder()
        .build()
        .map_err(|e| internal_error(format!("Failed to create HTTP client: {e}")))?;

    // Prepare request body
    let body = format!("grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion={jwt}");

    // Make request to Google's token endpoint
    let response = client
        .post("https://oauth2.googleapis.com/token")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .map_err(|e| internal_error(format!("Failed to request access token: {e}")))?;

    let status = response.status();

    if !status.is_success() {
        let error_body = response
            .text()
            .map_err(|e| internal_error(format!("Failed to read error response: {e}")))?;
        return Err(internal_error(format!(
            "Token exchange failed with status {status}: {error_body}"
        )));
    }

    // Parse response
    let response_body: serde_json::Value = response
        .json()
        .map_err(|e| internal_error(format!("Failed to parse token response: {e}")))?;

    // Extract access token
    let access_token = response_body
        .get("access_token")
        .and_then(|v| v.as_str())
        .ok_or_else(|| internal_error("No access_token in response"))?;

    debug!("Successfully obtained GCP access token");
    Ok(access_token.to_string())
}
