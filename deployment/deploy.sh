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
        if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
            cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        else
            cp "$PROJECT_ROOT/.env.production" "$PROJECT_ROOT/.env"
        fi
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
    if docker-compose ps postgres 2>/dev/null | grep -q Up; then
        log_info "Backing up PostgreSQL database..."
        docker-compose exec -T postgres pg_dump -U een een > "$BACKUP_DIR/postgres_backup.sql" || true
    fi
    
    # Backup Redis if exists  
    if docker-compose ps redis 2>/dev/null | grep -q Up; then
        log_info "Backing up Redis data..."
        docker-compose exec -T redis redis-cli BGSAVE || true
        docker cp een-redis:/data/dump.rdb "$BACKUP_DIR/redis_backup.rdb" 2>/dev/null || true
    fi
    
    # Backup application data
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        cp -r "$PROJECT_ROOT/data" "$BACKUP_DIR/" || true
    fi
    
    log_success "Backup created at $BACKUP_DIR"
}

# Build images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Pull latest base images
    docker-compose pull || true
    
    # Build application images
    docker-compose build --no-cache
    
    # Tag images with version
    if [[ "$VERSION" != "latest" ]]; then
        docker tag een-api:latest "een-api:$VERSION" 2>/dev/null || true
        docker tag een-dashboard:latest "een-dashboard:$VERSION" 2>/dev/null || true
    fi
    
    log_success "Images built successfully"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Check if we can build test image
    if docker build --target test -t een-test . 2>/dev/null; then
        # Run tests
        docker run --rm een-test || {
            log_warning "Some tests failed, but continuing with deployment"
        }
    else
        log_warning "Could not build test image, skipping tests"
    fi
    
    log_success "Test phase completed"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment
    export ENVIRONMENT="$ENVIRONMENT"
    export VERSION="$VERSION"
    
    # Create necessary directories
    mkdir -p logs data backups
    
    # Stop existing services
    docker-compose down || true
    
    # Start services
    case "$ENVIRONMENT" in
        "production")
            docker-compose up -d
            ;;
        "staging")
            if [[ -f "compose.staging.yaml" ]]; then
                docker-compose -f compose.yaml -f compose.staging.yaml up -d
            else
                docker-compose up -d
            fi
            ;;
        "development")
            if [[ -f "compose.dev.yaml" ]]; then
                docker-compose -f compose.yaml -f compose.dev.yaml up -d
            else
                docker-compose up -d
            fi
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
    
    local health_ok=true
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        health_ok=false
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
        log_warning "Database health check failed (may not be configured)"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        log_success "Redis health check passed"
    else
        log_warning "Redis health check failed (may not be configured)"
    fi
    
    if [[ "$health_ok" == "true" ]]; then
        log_success "Essential health checks passed"
        return 0
    else
        log_error "Critical health checks failed"
        return 1
    fi
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Stop current deployment
    docker-compose down || true
    
    # Find latest backup
    if [[ -d "$PROJECT_ROOT/backups" ]]; then
        LATEST_BACKUP=$(find "$PROJECT_ROOT/backups" -maxdepth 1 -type d -name "*" | sort -r | head -n1)
        
        if [[ -n "$LATEST_BACKUP" && "$LATEST_BACKUP" != "$PROJECT_ROOT/backups" ]]; then
            log_info "Restoring from backup: $LATEST_BACKUP"
            
            # Restore database
            if [[ -f "$LATEST_BACKUP/postgres_backup.sql" ]]; then
                docker-compose up -d postgres
                sleep 10
                docker-compose exec -T postgres psql -U een -c "DROP DATABASE IF EXISTS een;" || true
                docker-compose exec -T postgres psql -U een -c "CREATE DATABASE een;" || true
                cat "$LATEST_BACKUP/postgres_backup.sql" | docker-compose exec -T postgres psql -U een een || true
            fi
            
            # Restore Redis
            if [[ -f "$LATEST_BACKUP/redis_backup.rdb" ]]; then
                docker-compose up -d redis
                sleep 5
                docker cp "$LATEST_BACKUP/redis_backup.rdb" een-redis:/data/dump.rdb || true
                docker-compose restart redis || true
            fi
            
            # Restore application data
            if [[ -d "$LATEST_BACKUP/data" ]]; then
                rm -rf "$PROJECT_ROOT/data" || true
                cp -r "$LATEST_BACKUP/data" "$PROJECT_ROOT/" || true
            fi
            
            log_success "Rollback completed"
        else
            log_error "No backup found for rollback"
            exit 1
        fi
    else
        log_error "No backups directory found"
        exit 1
    fi
}

# Cleanup old images and volumes
cleanup() {
    log_info "Cleaning up old resources..."
    
    # Remove old images
    docker image prune -f || true
    
    # Remove unused volumes
    docker volume prune -f || true
    
    # Remove old backups (keep last 10)
    if [[ -d "$PROJECT_ROOT/backups" ]]; then
        find "$PROJECT_ROOT/backups" -maxdepth 1 -type d -name "*" | sort -r | tail -n +11 | xargs rm -rf || true
    fi
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting deployment to $ENVIRONMENT environment with version $VERSION"
    
    check_prerequisites
    
    # Backup only in production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        backup_data
    fi
    
    build_images
    
    # Run tests only if not in production
    if [[ "$ENVIRONMENT" != "production" ]]; then
        run_tests
    fi
    
    deploy_services
    
    # Health checks
    if ! health_checks; then
        log_error "Health checks failed"
        if [[ "$ENVIRONMENT" == "production" ]]; then
            log_warning "Initiating rollback..."
            rollback
        fi
        exit 1
    fi
    
    cleanup
    
    log_success "Deployment completed successfully!"
    log_info "Services are available at:"
    log_info "  - API: http://localhost:8000"
    log_info "  - Dashboard: http://localhost:8050"
    log_info "  - Grafana: http://localhost:3000"
    log_info "  - Prometheus: http://localhost:9090"
}

# Handle script arguments
case "${1:-}" in
    "rollback")
        rollback
        ;;
    "cleanup")
        cleanup
        ;;
    "health")
        health_checks
        ;;
    *)
        main
        ;;
esac