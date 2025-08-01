#!/bin/bash

# Een Consciousness API Startup Script

set -e

echo "🧠 Starting Een Consciousness API..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from example..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "✅ Created .env file from example. Please update the values."
    else
        echo "❌ env.example not found. Please create a .env file manually."
        exit 1
    fi
fi

# Load environment variables
source .env

# Check if required environment variables are set
if [ -z "$EEN_SECRET_KEY" ] || [ "$EEN_SECRET_KEY" = "your-super-secret-key-here-change-this-in-production" ]; then
    echo "❌ Please set EEN_SECRET_KEY in your .env file"
    exit 1
fi

# Create SSL directory if it doesn't exist
mkdir -p ssl

# Generate self-signed SSL certificate if it doesn't exist
if [ ! -f ssl/cert.pem ] || [ ! -f ssl/key.pem ]; then
    echo "🔐 Generating self-signed SSL certificate..."
    openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Een/CN=localhost"
    echo "✅ SSL certificate generated"
fi

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "🐳 Running in Docker container..."
    exec uvicorn api.main:app --host 0.0.0.0 --port 8000
else
    echo "🖥️  Running in development mode..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    
    # Run the API
    echo "🚀 Starting API server..."
    exec uvicorn api.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --reload
fi 