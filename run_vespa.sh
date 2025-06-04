#!/bin/bash

# Exit on any error
set -e

echo "🚀 Mini Search Engine Quick Start"
echo "================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

if ! command_exists docker; then
    echo "❌ Docker is required but not installed."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is required but not installed."
    exit 1
fi

echo "✅ All prerequisites met!"
echo ""

# Check if Poetry is installed
if ! command_exists poetry; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies with Poetry..."
if ! poetry install; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi
echo ""

# Start Vespa
echo "🐳 Starting Vespa with Docker Compose..."
if ! docker-compose up -d; then
    echo "❌ Failed to start Vespa containers"
    exit 1
fi
echo ""

# Wait for Vespa to be ready
echo "⏳ Waiting for Vespa to be ready..."
sleep 30

# Deploy Vespa application
echo "📤 Deploying Vespa application..."
if ! bash deploy_vespa.sh; then
    echo "❌ Failed to deploy Vespa application"
    exit 1
fi
echo ""

# Instructions
echo "✨ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Crawl documents:    cd crawler && poetry run python doc_scraper.py"
echo "2. Index documents:    poetry run python indexer/indexer.py"
echo "3. Start API server:   poetry run python api/main.py"
echo "4. Open browser:       http://localhost:8000"
echo ""
echo "For more details, see README.md"