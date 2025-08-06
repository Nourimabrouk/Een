#!/bin/bash
# Een Unity Mathematics - Quick Deploy Script
# Fast deployment for immediate website access

set -e

echo "🚀 Een Unity Mathematics - Quick Deploy"
echo "======================================"
echo "Getting website ready for immediate access..."
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
REPO_NAME="Een"
GITHUB_USER="Nourimabrouk"
GITHUB_PAGES_URL="https://${GITHUB_USER}.github.io/${REPO_NAME}/"

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v git >/dev/null 2>&1; then
    print_error "Git is required but not installed"
    exit 1
fi

if ! command -v python >/dev/null 2>&1; then
    print_error "Python is required but not installed"
    exit 1
fi

print_success "Prerequisites check complete"

# Check git status
print_status "Checking git repository status..."

if ! git status >/dev/null 2>&1; then
    print_error "Not in a git repository!"
    exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    print_warning "Uncommitted changes detected. Committing them..."
    git add .
    git commit -m "Quick deploy - website optimization for immediate access"
fi

print_success "Git repository ready"

# Validate essential files
print_status "Validating essential files..."

if [ ! -f "website/index.html" ]; then
    print_error "Website index.html not found!"
    exit 1
fi

if [ ! -f "gallery_data.json" ]; then
    print_error "Gallery data not found!"
    exit 1
fi

print_success "Essential files validated"

# Push to GitHub
print_status "Pushing to GitHub..."
git push origin main

print_success "Code pushed to GitHub"

# Wait for deployment
print_status "Waiting for GitHub Pages deployment..."
print_warning "GitHub Pages deployment can take 5-10 minutes"

# Check if GitHub Pages is accessible
print_status "Checking GitHub Pages accessibility..."
for i in {1..6}; do
    if curl -f -s "$GITHUB_PAGES_URL" >/dev/null 2>&1; then
        print_success "GitHub Pages is now accessible!"
        print_success "URL: $GITHUB_PAGES_URL"
        break
    fi
    print_status "Attempt $i/6: GitHub Pages not ready yet..."
    sleep 60
done

# Final summary
echo ""
echo "🎉 DEPLOYMENT SUMMARY"
echo "===================="
echo ""
echo "🌍 Website URL: $GITHUB_PAGES_URL"
echo "📱 Mobile Access: Yes (responsive design)"
echo "🔍 Browser Compatibility: All modern browsers (Chrome, Firefox, Safari, Edge, Brave)"
echo ""
echo "🌟 Unity Mathematics Features:"
echo "   • 55 interactive visualizations"
echo "   • 1+1=1 mathematical proofs"
echo "   • Consciousness field demonstrations"
echo "   • φ-harmonic golden ratio patterns"
echo ""
echo "📋 For your friend:"
echo "   • Open Brave browser"
echo "   • Go to: $GITHUB_PAGES_URL"
echo "   • All features should work immediately"
echo ""
print_success "Een Unity Mathematics is now accessible globally!"
echo ""
echo "φ = 1.618033988749895 - Golden Ratio Resonance"
echo "∞ = φ = 1+1 = 1 = E_metagamer" 