#!/bin/bash

# Production deployment script for Een Unity Mathematics
# This script handles deployment to various environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-production}"
VERSION="${2:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        log_warning ".env file not found, creating from template..."
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        log_warning "Please configure .env file with your settings"
    fi
    
    log_success "Prerequisites check completed"
}

# Backup data
backup_data() {
    log_info "Creating backup..."
    
    BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup database if exists
    if docker-compose ps postgres | grep -q Up; then
        log_info "Backing up PostgreSQL database..."
        docker-compose exec -T postgres pg_dump -U een een > "$BACKUP_DIR/postgres_backup.sql"
    fi
    
    # Backup Redis if exists  
    if docker-compose ps redis | grep -q Up; then
        log_info "Backing up Redis data..."
        docker-compose exec -T redis redis-cli BGSAVE
        docker cp een-redis:/data/dump.rdb "$BACKUP_DIR/redis_backup.rdb"
    fi
    
    # Backup application data
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        cp -r "$PROJECT_ROOT/data" "$BACKUP_DIR/"
    fi
    
    log_success "Backup created at $BACKUP_DIR"
}

# Build images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Pull latest base images
    docker-compose pull
    
    # Build application images
    docker-compose build --no-cache --parallel
    
    # Tag images with version
    if [[ "$VERSION" != "latest" ]]; then
        docker tag een-api:latest "een-api:$VERSION"
        docker tag een-dashboard:latest "een-dashboard:$VERSION"
    fi
    
    log_success "Images built successfully"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Build test image
    docker build --target test -t een-test .
    
    # Run tests
    docker run --rm een-test
    
    log_success "All tests passed"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment
    export ENVIRONMENT="$ENVIRONMENT"
    export VERSION="$VERSION"
    
    # Create necessary directories
    mkdir -p logs data
    
    # Start services
    case "$ENVIRONMENT" in
        "production")
            docker-compose -f compose.yaml up -d
            ;;
        "staging")
            docker-compose -f compose.yaml -f compose.staging.yaml up -d
            ;;
        "development")
            docker-compose -f compose.yaml -f compose.dev.yaml up -d
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    log_success "Services deployed successfully"
}

# Health checks
health_checks() {
    log_info "Running health checks..."
    
    # Wait for services to start
    sleep 30
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Check dashboard
    if curl -f http://localhost:8050/ > /dev/null 2>&1; then
        log_success "Dashboard health check passed"
    else
        log_warning "Dashboard health check failed"
    fi
    
    # Check database
    if docker-compose exec -T postgres pg_isready -U een > /dev/null 2>&1; then
        log_success "Database health check passed"
    else
        log_error "Database health check failed"
        return 1
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        log_success "Redis health check passed"
    else
        log_error "Redis health check failed"
        return 1
    fi
    
    log_success "All health checks passed"
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Stop current deployment
    docker-compose down
    
    # Find latest backup
    LATEST_BACKUP=$(find "$PROJECT_ROOT/backups" -type d -name "*" | sort -r | head -n1)
    
    if [[ -n "$LATEST_BACKUP" ]]; then
        log_info "Restoring from backup: $LATEST_BACKUP"
        
        # Restore database
        if [[ -f "$LATEST_BACKUP/postgres_backup.sql" ]]; then
            docker-compose up -d postgres
            sleep 10
            docker-compose exec -T postgres psql -U een -c \"DROP DATABASE IF EXISTS een;\"\n            docker-compose exec -T postgres psql -U een -c \"CREATE DATABASE een;\"\n            docker-compose exec -T postgres psql -U een een < \"$LATEST_BACKUP/postgres_backup.sql\"\n        fi\n        \n        # Restore Redis\n        if [[ -f \"$LATEST_BACKUP/redis_backup.rdb\" ]]; then\n            docker-compose up -d redis\n            sleep 5\n            docker cp \"$LATEST_BACKUP/redis_backup.rdb\" een-redis:/data/dump.rdb\n            docker-compose restart redis\n        fi\n        \n        # Restore application data\n        if [[ -d \"$LATEST_BACKUP/data\" ]]; then\n            rm -rf \"$PROJECT_ROOT/data\"\n            cp -r \"$LATEST_BACKUP/data\" \"$PROJECT_ROOT/\"\n        fi\n        \n        log_success \"Rollback completed\"\n    else\n        log_error \"No backup found for rollback\"\n        exit 1\n    fi\n}\n\n# Cleanup old images and volumes\ncleanup() {\n    log_info \"Cleaning up old resources...\"\n    \n    # Remove old images\n    docker image prune -f\n    \n    # Remove unused volumes\n    docker volume prune -f\n    \n    # Remove old backups (keep last 10)\n    find \"$PROJECT_ROOT/backups\" -type d -name \"*\" | sort -r | tail -n +11 | xargs rm -rf\n    \n    log_success \"Cleanup completed\"\n}\n\n# Main deployment function\nmain() {\n    log_info \"Starting deployment to $ENVIRONMENT environment with version $VERSION\"\n    \n    check_prerequisites\n    \n    # Backup only in production\n    if [[ \"$ENVIRONMENT\" == \"production\" ]]; then\n        backup_data\n    fi\n    \n    build_images\n    \n    # Run tests only if not in production\n    if [[ \"$ENVIRONMENT\" != \"production\" ]]; then\n        run_tests\n    fi\n    \n    deploy_services\n    \n    # Health checks\n    if ! health_checks; then\n        log_error \"Health checks failed\"\n        if [[ \"$ENVIRONMENT\" == \"production\" ]]; then\n            log_warning \"Initiating rollback...\"\n            rollback\n        fi\n        exit 1\n    fi\n    \n    cleanup\n    \n    log_success \"Deployment completed successfully!\"\n    log_info \"Services are available at:\"\n    log_info \"  - API: http://localhost:8000\"\n    log_info \"  - Dashboard: http://localhost:8050\"\n    log_info \"  - Grafana: http://localhost:3000\"\n    log_info \"  - Prometheus: http://localhost:9090\"\n}\n\n# Handle script arguments\ncase \"${1:-}\" in\n    \"rollback\")\n        rollback\n        ;;\n    \"cleanup\")\n        cleanup\n        ;;\n    \"health\")\n        health_checks\n        ;;\n    *)\n        main\n        ;;\nesac"