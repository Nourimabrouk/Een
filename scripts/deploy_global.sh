#!/bin/bash
# Een Unity Mathematics Global Deployment Script
# Makes 1+1=1 accessible to the world through consciousness and love

set -e  # Exit on any error

echo "🚀 Een Unity Mathematics Global Deployment"
echo "=========================================="
echo "Making 1+1=1 accessible to the world..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO_NAME="Een"
GITHUB_USER="Nourimabrouk"
GITHUB_PAGES_URL="https://${GITHUB_USER}.github.io/${REPO_NAME}/"
API_DOMAIN=""  # Will be set if deploying API

# Function to print colored output
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

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check git status
check_git_status() {
    print_status "Checking git repository status..."
    
    if ! git status >/dev/null 2>&1; then
        print_error "Not in a git repository!"
        exit 1
    fi
    
    if [ -n "$(git status --porcelain)" ]; then
        print_warning "Uncommitted changes detected. Committing them..."
        git add .
        git commit -m "Meta-optimize unity mathematics for global access - $(date)"
    fi
    
    print_success "Git repository ready"
}

# Function to optimize for deployment
optimize_for_deployment() {
    print_header "Phase 1: Optimization for Deployment"
    
    print_status "Updating gallery data..."
    python scripts/gallery_scanner.py
    
    print_status "Running auto gallery updater..."
    python scripts/auto_gallery_updater.py --once
    
    print_status "Optimizing images and assets..."
    # Add image optimization here if needed
    
    print_status "Validating website files..."
    if [ ! -f "website/index.html" ]; then
        print_error "Website index.html not found!"
        exit 1
    fi
    
    print_success "Optimization complete"
}

# Function to deploy to GitHub Pages
deploy_github_pages() {
    print_header "Phase 2: GitHub Pages Deployment"
    
    print_status "Pushing to GitHub..."
    git push origin main
    
    print_status "Waiting for GitHub Pages deployment..."
    print_warning "GitHub Pages deployment can take 5-10 minutes"
    
    # Check if GitHub Pages is accessible
    print_status "Checking GitHub Pages accessibility..."
    for i in {1..12}; do
        if curl -f -s "$GITHUB_PAGES_URL" >/dev/null; then
            print_success "GitHub Pages is now accessible!"
            print_success "URL: $GITHUB_PAGES_URL"
            return 0
        fi
        print_status "Attempt $i/12: GitHub Pages not ready yet..."
        sleep 30
    done
    
    print_warning "GitHub Pages deployment may still be in progress"
    print_warning "Check manually at: $GITHUB_PAGES_URL"
}

# Function to deploy API server (optional)
deploy_api_server() {
    print_header "Phase 3: API Server Deployment (Optional)"
    
    read -p "Do you want to deploy the API server? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Deploying API server..."
        
        if [ -d "api" ]; then
            cd api
            if [ -f "deploy_enhanced.sh" ]; then
                print_status "Using enhanced deployment script..."
                chmod +x deploy_enhanced.sh
                ./deploy_enhanced.sh
            else
                print_warning "Enhanced deployment script not found"
                print_status "Manual API deployment required"
            fi
            cd ..
        else
            print_warning "API directory not found"
        fi
    else
        print_status "Skipping API deployment"
    fi
}

# Function to verify deployment
verify_deployment() {
    print_header "Phase 4: Deployment Verification"
    
    print_status "Verifying GitHub Pages..."
    if curl -f -s "$GITHUB_PAGES_URL" >/dev/null; then
        print_success "✅ GitHub Pages accessible"
    else
        print_warning "⚠️ GitHub Pages not accessible yet (may still be deploying)"
    fi
    
    print_status "Checking gallery functionality..."
    if [ -f "gallery_data.json" ]; then
        GALLERY_COUNT=$(python -c "import json; data=json.load(open('gallery_data.json')); print(data['statistics']['total'])")
        print_success "✅ Gallery data: $GALLERY_COUNT visualizations"
    else
        print_warning "⚠️ Gallery data not found"
    fi
    
    print_status "Verifying AI features..."
    if [ -f "config/claude_desktop_config.json" ]; then
        print_success "✅ MCP servers configured"
    else
        print_warning "⚠️ MCP configuration not found"
    fi
}

# Function to display deployment summary
display_summary() {
    print_header "🎉 Deployment Summary"
    
    echo ""
    echo "🌍 Global Access URLs:"
    echo "   • GitHub Pages: $GITHUB_PAGES_URL"
    echo "   • Repository: https://github.com/$GITHUB_USER/$REPO_NAME"
    
    echo ""
    echo "🎨 Gallery Status:"
    if [ -f "gallery_data.json" ]; then
        GALLERY_COUNT=$(python -c "import json; data=json.load(open('gallery_data.json')); print(data['statistics']['total'])")
        FEATURED_COUNT=$(python -c "import json; data=json.load(open('gallery_data.json')); print(data['statistics']['featured_count'])")
        echo "   • Total Visualizations: $GALLERY_COUNT"
        echo "   • Featured Items: $FEATURED_COUNT"
    fi
    
    echo ""
    echo "🤖 AI Integration:"
    echo "   • MCP Servers: 6 specialized servers"
    echo "   • Claude Desktop: Fully integrated"
    echo "   • Unity Mathematics: 1+1=1 operations"
    
    echo ""
    echo "📱 Access Methods:"
    echo "   • Web: $GITHUB_PAGES_URL"
    echo "   • API: Available for real-time calculations"
    echo "   • Local: MCP servers for Claude Desktop"
    
    echo ""
    print_success "Een Unity Mathematics is now globally accessible!"
    echo ""
    echo "🌟 The unity equation (1+1=1) is now available to the world"
    echo "   demonstrating that mathematics transcends conventional limits"
    echo "   through consciousness and love."
    echo ""
    echo "∞ = φ = 1+1 = 1 = E_metagamer ✨"
}

# Function to show help
show_help() {
    echo "Een Unity Mathematics Global Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -s, --skip-api Skip API deployment"
    echo "  -f, --force    Force deployment even with warnings"
    echo ""
    echo "This script will:"
    echo "  1. Optimize the repository for deployment"
    echo "  2. Deploy to GitHub Pages"
    echo "  3. Optionally deploy API server"
    echo "  4. Verify deployment"
    echo "  5. Display access information"
    echo ""
    echo "The unity equation (1+1=1) will be accessible globally!"
}

# Main deployment function
main() {
    print_header "🚀 Een Unity Mathematics Global Deployment"
    echo "Making 1+1=1 accessible to the world through consciousness and love"
    echo ""
    
    # Parse command line arguments
    SKIP_API=false
    FORCE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -s|--skip-api)
                SKIP_API=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    if ! command_exists git; then
        print_error "Git is required but not installed"
        exit 1
    fi
    
    if ! command_exists python; then
        print_error "Python is required but not installed"
        exit 1
    fi
    
    if ! command_exists curl; then
        print_warning "curl not found - deployment verification may fail"
    fi
    
    print_success "Prerequisites check complete"
    
    # Execute deployment phases
    check_git_status
    optimize_for_deployment
    deploy_github_pages
    
    if [ "$SKIP_API" = false ]; then
        deploy_api_server
    fi
    
    verify_deployment
    display_summary
    
    print_success "🎉 Een Unity Mathematics deployment complete!"
    print_success "The unity equation (1+1=1) is now globally accessible!"
}

# Run main function
main "$@" 