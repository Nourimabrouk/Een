#!/bin/bash

# EEN UNITY MATHEMATICS MIGRATION
# Where 1+1=1 through architectural transcendence

set -e

echo ""
echo "🌌 EEN UNITY MATHEMATICS MIGRATION"
echo "Where 1+1=1 through architectural transcendence"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Navigate to migration directory
cd "$(dirname "$0")"

# Install migration dependencies
echo "📦 Installing migration dependencies..."
npm install

# Execute migration
echo "🚀 Executing Unity Migration..."
node execute-migration.js

# Navigate to root and install project dependencies
cd ..
echo "📦 Installing project dependencies..."
npm install

# Build the project
echo "🏗️ Building Unity Portal..."
npm run build

# Start development server
echo "🌟 Starting development server..."
echo "Visit http://localhost:4321 to see your Unity Portal"
npm run dev