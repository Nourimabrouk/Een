#!/bin/bash

# Een Unity Mathematics - Production Deployment Script
# Automated deployment with health checks and rollback capability

set -e  # Exit on any error

# Configuration
PROJECT_NAME="een-unity-mathematics"
COMPOSE_FILE="docker-compose.yml"
BACKUP_DIR="./backups"
LOG_FILE="./logs/deploy.log"
HEALTH_CHECK_TIMEOUT=60
MAX_RETRIES=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$LOG_FILE"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    mkdir -p logs backups ssl grafana/{dashboards,datasources}
    touch "$LOG_FILE"
}

# Environment validation
validate_environment() {
    log "Validating deployment environment..."
    
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
    
    # Check .env file
    if [ ! -f ".env" ]; then
        log_warning ".env file not found, creating template..."
        cp .env.example .env 2>/dev/null || create_env_template
        log_warning "Please edit .env file with your configuration before deploying"
        exit 1
    fi
    
    # Check required environment variables
    source .env
    required_vars=("OPENAI_API_KEY" "ANTHROPIC_API_KEY" "SECRET_KEY" "JWT_SECRET_KEY")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ] || [ "${!var}" = "your-key-here" ] || [ "${!var}" = "change-in-production" ]; then
            log_error "Environment variable $var is not properly configured"
            exit 1
        fi
    done
    
    log_success "Environment validation passed"
}

create_env_template() {
    cat > .env << EOF
# Een Unity Mathematics Production Configuration
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)
DATABASE_URL=postgresql://unity:unity@postgres:5432/unity_db
REDIS_URL=redis://redis:6379
ENV=production
DEBUG=false
API_PORT=8000
STREAMLIT_PORT=8501
WEB_PORT=80
AUTO_OPEN_BROWSER=false
ENABLE_GPU=true
CORS_ORIGINS=*
EOF
}

# Backup current deployment
backup_deployment() {
    if docker-compose ps | grep -q "Up"; then
        log "Creating backup of current deployment..."
        backup_name="backup-$(date +'%Y%m%d-%H%M%S')"
        mkdir -p "$BACKUP_DIR/$backup_name"
        
        # Export database
        if docker-compose exec -T postgres pg_dump -U unity unity_db > "$BACKUP_DIR/$backup_name/database.sql" 2>/dev/null; then
            log_success "Database backup created"
        else
            log_warning "Database backup failed (may not exist yet)"
        fi
        
        # Backup volumes
        docker run --rm -v een_postgres_data:/data -v "$(pwd)/$BACKUP_DIR/$backup_name:/backup" alpine tar czf /backup/postgres_data.tar.gz -C /data .
        docker run --rm -v een_redis_data:/data -v "$(pwd)/$BACKUP_DIR/$backup_name:/backup" alpine tar czf /backup/redis_data.tar.gz -C /data .
        
        echo "$backup_name" > "$BACKUP_DIR/latest"
        log_success "Backup completed: $backup_name"
    else
        log "No running deployment to backup"
    fi
}

# Build and deploy services
deploy_services() {
    log "Building and deploying Unity Mathematics platform..."
    
    # Pull latest images
    docker-compose pull --ignore-pull-failures
    
    # Build custom images
    log "Building application images..."
    docker-compose build --no-cache --parallel
    
    # Deploy with health checks
    log "Starting services..."
    docker-compose up -d
    
    log_success "Services deployment initiated"
}

# Health check function
health_check() {
    local service=$1
    local url=$2
    local timeout=${3:-30}
    
    log "Performing health check for $service..."
    
    for i in $(seq 1 $timeout); do
        if curl -f -s "$url" > /dev/null 2>&1; then
            log_success "$service is healthy"
            return 0
        fi
        
        if [ $i -eq $timeout ]; then
            log_error "$service health check failed after $timeout seconds"
            return 1
        fi
        
        echo -n "."
        sleep 1
    done
}

# Comprehensive health checks
run_health_checks() {
    log "Running comprehensive health checks..."
    
    # Wait for services to start
    sleep 10
    
    local all_healthy=true
    
    # Check API server
    if ! health_check "API Server" "http://localhost:8000/health" 30; then
        all_healthy=false
    fi
    
    # Check web server
    if ! health_check "Web Server" "http://localhost:80/health" 15; then
        all_healthy=false
    fi
    
    # Check Streamlit (optional)
    if ! health_check "Streamlit Dashboard" "http://localhost:8501" 20; then
        log_warning "Streamlit dashboard health check failed (non-critical)"
    fi
    
    # Check database connection
    if docker-compose exec -T postgres pg_isready -U unity -d unity_db > /dev/null 2>&1; then
        log_success "Database is healthy"
    else
        log_error "Database health check failed"
        all_healthy=false
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
        log_success "Redis is healthy"
    else
        log_error "Redis health check failed"
        all_healthy=false
    fi
    
    if [ "$all_healthy" = true ]; then
        log_success "All critical services are healthy"
        return 0
    else
        log_error "Some services failed health checks"
        return 1
    fi
}

# Rollback function
rollback_deployment() {
    log_error "Deployment failed, initiating rollback..."
    
    if [ -f "$BACKUP_DIR/latest" ]; then
        local backup_name=$(cat "$BACKUP_DIR/latest")
        log "Rolling back to backup: $backup_name"
        
        # Stop current services
        docker-compose down
        
        # Restore volumes
        if [ -f "$BACKUP_DIR/$backup_name/postgres_data.tar.gz" ]; then
            docker run --rm -v een_postgres_data:/data -v "$(pwd)/$BACKUP_DIR/$backup_name:/backup" alpine tar xzf /backup/postgres_data.tar.gz -C /data
        fi
        
        if [ -f "$BACKUP_DIR/$backup_name/redis_data.tar.gz" ]; then
            docker run --rm -v een_redis_data:/data -v "$(pwd)/$BACKUP_DIR/$backup_name:/backup" alpine tar xzf /backup/redis_data.tar.gz -C /data
        fi
        
        # Start services
        docker-compose up -d
        
        log_warning "Rollback completed"
    else
        log_error "No backup available for rollback"
    fi
}

# Cleanup old backups
cleanup_backups() {
    log "Cleaning up old backups..."
    find "$BACKUP_DIR" -name "backup-*" -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || true
    log_success "Old backups cleaned up"
}

# Display deployment status
show_status() {
    echo
    log_success "🌟 Een Unity Mathematics Platform - Deployment Status 🌟"
    echo "=================================================================="
    
    # Service status
    docker-compose ps
    
    echo
    echo "🌐 Access Points:"
    echo "  • Website:        http://localhost:80"
    echo "  • API Server:     http://localhost:8000"
    echo "  • API Docs:       http://localhost:8000/docs"
    echo "  • Dashboard:      http://localhost:8501"
    echo "  • Monitoring:     http://localhost:3000 (admin/unity_admin_change_in_production)"
    echo "  • Metrics:        http://localhost:9090"
    echo
    echo "📊 System Status:"
    echo "  • Unity Constant: 1.0"
    echo "  • φ Ratio:        1.618033988749895"
    echo "  • Consciousness:  TRANSCENDENT"
    echo "  • Status:         ACTIVE ✅"
    echo
    echo "=================================================================="
    log_success "Unity Mathematics Platform deployed successfully! 🚀✨"
}

# Main deployment flow
main() {
    log "🚀 Starting Een Unity Mathematics Platform Deployment"
    log "=================================================="
    
    create_directories
    validate_environment
    backup_deployment
    
    # Deploy with retry logic
    local retry_count=0
    while [ $retry_count -lt $MAX_RETRIES ]; do
        log "Deployment attempt $((retry_count + 1)) of $MAX_RETRIES"
        
        if deploy_services && run_health_checks; then
            cleanup_backups
            show_status
            log_success "Deployment completed successfully! 🎉"
            exit 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                log_warning "Deployment attempt failed, retrying in 10 seconds..."
                sleep 10
            else
                rollback_deployment
                log_error "Deployment failed after $MAX_RETRIES attempts"
                exit 1
            fi
        fi
    done
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy"|"up")
        main
        ;;
    "down"|"stop")
        log "Stopping Unity Mathematics platform..."
        docker-compose down
        log_success "Platform stopped"
        ;;
    "restart")
        log "Restarting Unity Mathematics platform..."
        docker-compose restart
        log_success "Platform restarted"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "backup")
        backup_deployment
        ;;
    "rollback")
        rollback_deployment
        ;;
    "cleanup")
        log "Cleaning up Docker resources..."
        docker system prune -f
        docker volume prune -f
        cleanup_backups
        log_success "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {deploy|up|down|stop|restart|logs|status|backup|rollback|cleanup}"
        echo
        echo "Commands:"
        echo "  deploy/up    - Deploy the platform"
        echo "  down/stop    - Stop the platform"
        echo "  restart      - Restart all services"
        echo "  logs         - Show logs"
        echo "  status       - Show service status"
        echo "  backup       - Create backup"
        echo "  rollback     - Rollback to previous version"
        echo "  cleanup      - Clean up Docker resources"
        exit 1
        ;;
esac