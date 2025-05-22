import os
import requests
import json

# Load API key from .env file
with open('.env', 'r') as f:
    for line in f:
        if line.startswith('OPENAI_API_KEY='):
            os.environ['OPENAI_API_KEY'] = line.strip().split('=', 1)[1]

api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

print(f"Using API key: {api_key[:5]}...{api_key[-4:]}")

# Set up the API request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "text-embedding-3-small",
    "input": ["This is a test sentence for OpenAI embeddings."]
}

# Make the API request
response = requests.post(
    "https://api.openai.com/v1/embeddings",
    headers=headers,
    json=payload
)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    print("\n✅ API request successful!")
    print(f"Model: {data['model']}")
    print(f"Number of embeddings: {len(data['data'])}")
    print(f"Embedding dimensions: {len(data['data'][0]['embedding'])}")
    print(f"Usage - Prompt tokens: {data['usage']['prompt_tokens']}")
    print(f"Usage - Total tokens: {data['usage']['total_tokens']}")
    print("\nThe OpenAI API key is working correctly with the embeddings service.")
    print("This confirms that the bounty issue can be solved with this API key.")
else:
    print(f"\n❌ API request failed with status code: {response.status_code}")
    print(f"Error message: {response.text}")