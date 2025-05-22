#!/bin/bash

echo "🧪 Running OpenAI Embedding Tests for Bounty Verification 🧪"
echo "=================================================="

# Check if .env file exists, if not, create it from .env.example
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found, creating from .env.example"
    cp .env.example .env
    echo "⚠️ Please edit .env file to add your OpenAI API key before running tests"
    echo "⚠️ Replace 'your-openai-api-key-here' with your actual API key"
    exit 1
fi

# Run Python test script
echo "\n📝 Running Python API test script..."
python3 test_embed_openai_bounty.py

# Run Rust tests
echo "\n📝 Running Rust implementation tests..."
cd embed-openai
echo "Running API direct test..."
cargo test --test api_test -- --nocapture
echo "\nRunning OpenAI WIT interface test..."
cargo test --test openai_test -- --nocapture
echo "\nRunning bounty requirements test..."
cargo test --test bounty_test -- --nocapture

echo "\n✅ All tests completed!"
echo "If all tests passed, the OpenAI embedding implementation meets the bounty requirements."