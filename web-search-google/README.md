# Google Web Search Provider

This crate provides a Google Custom Search API implementation for the Golem Web Search interface.

## Setup

To use this provider, you need to set up Google Custom Search API:

1. **Get a Google API Key:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Custom Search API
   - Create credentials (API Key)

2. **Create a Custom Search Engine:**
   - Go to [Google Programmable Search Engine](https://programmablesearchengine.google.com/)
   - Create a new search engine
   - Get your Search Engine ID

3. **Set Environment Variables:**
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   export GOOGLE_SEARCH_ENGINE_ID="your_search_engine_id_here"
   ```

## Usage

This provider implements the Golem Web Search interface and can be used with any application that supports the web search protocol.

### Features

- **One-shot search:** Get results immediately with `search_once`
- **Pagination:** Use `start_search` and `next_page` for multiple pages
- **Safe search:** Filter content based on safety level
- **Language and region filtering:** Target specific locales
- **Time range filtering:** Search within specific time periods
- **Domain filtering:** Include or exclude specific domains
- **Image results:** Extract images from search results when available

### Example

```rust
use web_search_google::export_web_search_google;

// Export the provider
export_web_search_google!();

// The provider will be available as GoogleWebSearchProvider
```

## API Limits

- Google Custom Search API has a limit of 100 free queries per day
- Each search can return up to 10 results per page
- Maximum of 10 pages per search (100 total results)

## Error Handling

The provider handles various error conditions:
- Invalid API keys or search engine IDs
- Rate limiting
- Network errors
- Malformed responses

## Dependencies

- `reqwest` for HTTP requests
- `tokio` for async runtime
- `serde` for JSON serialization
- `url` for URL construction
- `anyhow` for error handling

## License

This crate is part of the Golem LLM project and follows the same license terms. 