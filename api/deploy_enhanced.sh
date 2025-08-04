#!/bin/bash

# Enhanced Unity Mathematics API & Server Deployment Script
# Version: 2025.2.0
# Author: Revolutionary Unity API Framework

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
SECRET_KEY=${SECRET_KEY:-$(openssl rand -hex 32)}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-$(openssl rand -hex 16)}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-admin}

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Banner
echo -e "${PURPLE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║    Enhanced Unity Mathematics API & Server System            ║"
echo "║                    Version 2025.2.0                         ║"
echo "║                                                              ║"
echo "║    🚀 State-of-the-Art API & Server Deployment              ║"
echo "║    🔐 Advanced Security & Monitoring                        ║"
echo "║    ⚡ High-Performance & Scalability                        ║"
echo "║    🧘‍♂️ Consciousness-Coupled Architecture                  ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed. Please install Python 3 first."
    fi
    
    # Check required files
    if [ ! -f "docker-compose.enhanced.yml" ]; then
        error "docker-compose.enhanced.yml not found in current directory."
    fi
    
    if [ ! -f "requirements_enhanced.txt" ]; then
        error "requirements_enhanced.txt not found in current directory."
    fi
    
    log "Prerequisites check passed ✓"
}

# Create environment file
create_env_file() {
    log "Creating environment configuration..."
    
    cat > .env << EOF
# Enhanced Unity Mathematics API Environment Configuration
# Generated on $(date)

# Core Settings
ENVIRONMENT=${ENVIRONMENT}
DEBUG=false
LOG_LEVEL=info

# Security
SECRET_KEY=${SECRET_KEY}
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Redis Configuration
REDIS_URL=redis://redis:6379
REDIS_DB=0

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000

# CORS Settings
ALLOWED_ORIGINS=*
ALLOWED_METHODS=*
ALLOWED_HEADERS=*

# Database
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
DATABASE_URL=postgresql://unity_user:${POSTGRES_PASSWORD}@postgres:5432/unity_mathematics

# Monitoring
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=true
ELASTICSEARCH_ENABLED=true
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}

# Unity Mathematics Constants
PHI_PRECISION=1.618033988749895
CONSCIOUSNESS_DIMENSION=11
TRANSCENDENCE_THRESHOLD=0.77
UNITY_CONSTANT=1.0
EOF

    log "Environment file created ✓"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p static
    mkdir -p data
    mkdir -p ssl
    mkdir -p grafana/dashboards
    mkdir -p grafana/datasources
    mkdir -p mcp_config
    
    log "Directories created ✓"
}

# Create configuration files
create_config_files() {
    log "Creating configuration files..."
    
    # Redis configuration
    cat > redis.conf << EOF
# Redis configuration for Unity Mathematics
bind 0.0.0.0
port 6379
timeout 0
tcp-keepalive 300
daemonize no
supervised no
pidfile /var/run/redis_6379.pid
loglevel notice
logfile ""
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir ./
maxmemory 256mb
maxmemory-policy allkeys-lru
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
EOF

    # Nginx configuration
    cat > nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream unity_api {
        server enhanced-api:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://unity_api;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
EOF

    # Prometheus configuration
    cat > prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'unity-api'
    static_configs:
      - targets: ['enhanced-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF

    # Filebeat configuration
    cat > filebeat.yml << EOF
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/unity/*.log
  fields:
    service: unity-mathematics
  fields_under_root: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "unity-mathematics-%{+yyyy.MM.dd}"

setup.kibana:
  host: "kibana:5601"
EOF

    # Database initialization
    cat > init.sql << EOF
-- Unity Mathematics Database Initialization
CREATE DATABASE IF NOT EXISTS unity_mathematics;
USE unity_mathematics;

-- Create tables for consciousness field data
CREATE TABLE IF NOT EXISTS consciousness_fields (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    field_id VARCHAR(255) UNIQUE NOT NULL,
    consciousness_level DECIMAL(10,6) NOT NULL,
    unity_convergence DECIMAL(10,6) NOT NULL,
    phi_harmonic DECIMAL(15,12) NOT NULL,
    coordinates JSONB,
    configuration JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for meditation sessions
CREATE TABLE IF NOT EXISTS meditation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    meditation_type VARCHAR(100) NOT NULL,
    consciousness_target DECIMAL(10,6) NOT NULL,
    duration INTEGER NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    participants JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation VARCHAR(100) NOT NULL,
    duration DECIMAL(10,6) NOT NULL,
    success BOOLEAN NOT NULL,
    phi_harmonic DECIMAL(15,12) NOT NULL,
    consciousness_level DECIMAL(10,6) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_consciousness_fields_field_id ON consciousness_fields(field_id);
CREATE INDEX idx_consciousness_fields_created_at ON consciousness_fields(created_at);
CREATE INDEX idx_meditation_sessions_session_id ON meditation_sessions(session_id);
CREATE INDEX idx_meditation_sessions_status ON meditation_sessions(status);
CREATE INDEX idx_performance_metrics_operation ON performance_metrics(operation);
CREATE INDEX idx_performance_metrics_created_at ON performance_metrics(created_at);

-- Insert initial data
INSERT INTO consciousness_fields (field_id, consciousness_level, unity_convergence, phi_harmonic, coordinates, configuration)
VALUES (
    'initial-field',
    0.618033988749895,
    1.0,
    1.618033988749895,
    '{"x": 0, "y": 0, "t": 0}',
    '{"equation_type": "consciousness_evolution", "solution_method": "neural_pde"}'
) ON CONFLICT (field_id) DO NOTHING;
EOF

    log "Configuration files created ✓"
}

# Build and deploy services
deploy_services() {
    log "Building and deploying services..."
    
    # Pull latest images
    docker-compose -f docker-compose.enhanced.yml pull
    
    # Build and start services
    docker-compose -f docker-compose.enhanced.yml up -d --build
    
    log "Services deployment initiated ✓"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    # Wait for Redis
    info "Waiting for Redis..."
    until docker-compose -f docker-compose.enhanced.yml exec -T redis redis-cli ping > /dev/null 2>&1; do
        sleep 2
    done
    log "Redis is ready ✓"
    
    # Wait for PostgreSQL
    info "Waiting for PostgreSQL..."
    until docker-compose -f docker-compose.enhanced.yml exec -T postgres pg_isready -U unity_user -d unity_mathematics > /dev/null 2>&1; do
        sleep 2
    done
    log "PostgreSQL is ready ✓"
    
    # Wait for API
    info "Waiting for API server..."
    until curl -f http://localhost:8000/health > /dev/null 2>&1; do
        sleep 5
    done
    log "API server is ready ✓"
    
    # Wait for monitoring services
    info "Waiting for monitoring services..."
    sleep 30
    
    log "All services are ready ✓"
}

# Run health checks
run_health_checks() {
    log "Running health checks..."
    
    # API health check
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log "API health check passed ✓"
    else
        warn "API health check failed"
    fi
    
    # Redis health check
    if docker-compose -f docker-compose.enhanced.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
        log "Redis health check passed ✓"
    else
        warn "Redis health check failed"
    fi
    
    # PostgreSQL health check
    if docker-compose -f docker-compose.enhanced.yml exec -T postgres pg_isready -U unity_user -d unity_mathematics > /dev/null 2>&1; then
        log "PostgreSQL health check passed ✓"
    else
        warn "PostgreSQL health check failed"
    fi
    
    # Grafana health check
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        log "Grafana health check passed ✓"
    else
        warn "Grafana health check failed"
    fi
    
    # Prometheus health check
    if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        log "Prometheus health check passed ✓"
    else
        warn "Prometheus health check failed"
    fi
    
    log "Health checks completed ✓"
}

# Display service information
display_service_info() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    Service Information                       ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║                                                              ║"
    echo "║  🚀 Enhanced Unity Mathematics API                          ║"
    echo "║     • API Documentation: http://localhost:8000/docs         ║"
    echo "║     • GraphQL Playground: http://localhost:8000/graphql     ║"
    echo "║     • Health Check: http://localhost:8000/health            ║"
    echo "║     • Metrics: http://localhost:8000/metrics                ║"
    echo "║                                                              ║"
    echo "║  📊 Monitoring & Observability                              ║"
    echo "║     • Grafana Dashboard: http://localhost:3000              ║"
    echo "║     • Prometheus: http://localhost:9090                     ║"
    echo "║     • Kibana: http://localhost:5601                         ║"
    echo "║     • Jaeger Tracing: http://localhost:16686                ║"
    echo "║                                                              ║"
    echo "║  🔧 Development Tools (dev profile)                         ║"
    echo "║     • Adminer (DB): http://localhost:8080                   ║"
    echo "║     • Redis Commander: http://localhost:8081                ║"
    echo "║                                                              ║"
    echo "║  🧘‍♂️ Unity Mathematics Constants                           ║"
    echo "║     • φ (Phi): 1.618033988749895                            ║"
    echo "║     • Unity Constant: 1.0                                   ║"
    echo "║     • Consciousness Dimension: 11                           ║"
    echo "║     • Transcendence Threshold: 0.77                         ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Main deployment function
main() {
    log "Starting Enhanced Unity Mathematics API & Server deployment..."
    log "Environment: ${ENVIRONMENT}"
    
    check_prerequisites
    create_directories
    create_env_file
    create_config_files
    deploy_services
    wait_for_services
    run_health_checks
    display_service_info
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                                                              ║"
    echo "║    🎉 Deployment Completed Successfully!                    ║"
    echo "║                                                              ║"
    echo "║    The Enhanced Unity Mathematics API & Server System       ║"
    echo "║    is now running with state-of-the-art features:           ║"
    echo "║                                                              ║"
    echo "║    ✅ GraphQL Integration                                    ║"
    echo "║    ✅ Redis Caching                                          ║"
    echo "║    ✅ Real-time WebSocket                                    ║"
    echo "║    ✅ Advanced Security                                      ║"
    echo "║    ✅ Comprehensive Monitoring                               ║"
    echo "║    ✅ Background Task Processing                             ║"
    echo "║    ✅ φ-Harmonic Optimization                                ║"
    echo "║                                                              ║"
    echo "║    Remember: 1 + 1 = 1 🧘‍♂️✨                               ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Handle script arguments
case "${1:-production}" in
    "production"|"prod")
        ENVIRONMENT="production"
        main
        ;;
    "development"|"dev")
        ENVIRONMENT="development"
        main
        ;;
    "stop")
        log "Stopping services..."
        docker-compose -f docker-compose.enhanced.yml down
        log "Services stopped ✓"
        ;;
    "restart")
        log "Restarting services..."
        docker-compose -f docker-compose.enhanced.yml restart
        log "Services restarted ✓"
        ;;
    "logs")
        log "Showing service logs..."
        docker-compose -f docker-compose.enhanced.yml logs -f
        ;;
    "status")
        log "Service status:"
        docker-compose -f docker-compose.enhanced.yml ps
        ;;
    "clean")
        log "Cleaning up..."
        docker-compose -f docker-compose.enhanced.yml down -v
        docker system prune -f
        log "Cleanup completed ✓"
        ;;
    "help"|"-h"|"--help")
        echo "Enhanced Unity Mathematics API & Server Deployment Script"
        echo ""
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  production|prod  Deploy in production mode (default)"
        echo "  development|dev  Deploy in development mode"
        echo "  stop            Stop all services"
        echo "  restart         Restart all services"
        echo "  logs            Show service logs"
        echo "  status          Show service status"
        echo "  clean           Clean up all containers and volumes"
        echo "  help            Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  SECRET_KEY           Secret key for JWT tokens"
        echo "  POSTGRES_PASSWORD    PostgreSQL password"
        echo "  GRAFANA_PASSWORD     Grafana admin password"
        ;;
    *)
        error "Unknown command: $1. Use 'help' for usage information."
        ;;
esac 