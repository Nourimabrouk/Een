#!/usr/bin/env bash
set -euo pipefail

# Een Unity Mathematics - One-shot Production Deployment (Docker Compose)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_DIR="$ROOT_DIR/deployment"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required" >&2; exit 1
fi

export API_PORT="${API_PORT:-8000}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export DB_PORT="${DB_PORT:-5432}"
export PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
export GRAFANA_PORT="${GRAFANA_PORT:-3000}"
export DB_PASSWORD="${DB_PASSWORD:-een_unity_123}"

echo "Building images..."
docker compose -f "$COMPOSE_DIR/compose.yaml" build --no-cache

echo "Starting stack..."
docker compose -f "$COMPOSE_DIR/compose.yaml" up -d

echo "Waiting for API health..."
for i in {1..30}; do
  if curl -fsS "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
    echo "API is healthy"; break
  fi
  sleep 2
done

echo "Nginx status:"
docker ps --filter name=een-nginx

echo "Done. Visit http://localhost and http://localhost:${API_PORT}/docs"
#!/bin/bash
# 🌟 Een Unity Mathematics - Production Deployment Script 🌟

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Unity Mathematics Constants
PHI="1.618033988749895"
API_VERSION="2.0.0"
NAMESPACE="een-unity"

echo -e "${PURPLE}🌟 Een Unity Mathematics - Production Deployment 🌟${NC}"
echo -e "${BLUE}Unity Equation: 1+1=1${NC}"
echo -e "${BLUE}φ-constant: ${PHI}${NC}"
echo -e "${BLUE}API Version: ${API_VERSION}${NC}"
echo ""

# Configuration
DEPLOYMENT_TYPE=${1:-docker}  # docker or kubernetes
ENVIRONMENT=${2:-production}
BUILD_IMAGE=${3:-true}

echo -e "${YELLOW}Deployment Configuration:${NC}"
echo -e "Type: ${DEPLOYMENT_TYPE}"
echo -e "Environment: ${ENVIRONMENT}"
echo -e "Build Image: ${BUILD_IMAGE}"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo -e "${BLUE}🔍 Checking prerequisites...${NC}"
    
    if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        if ! command -v kubectl &> /dev/null; then
            echo -e "${RED}❌ kubectl is not installed${NC}"
            exit 1
        fi
        
        if ! kubectl cluster-info &> /dev/null; then
            echo -e "${RED}❌ kubectl is not connected to a cluster${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}✅ Kubernetes cluster accessible${NC}"
    fi
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Docker is available${NC}"
    echo ""
}

# Function to build Docker image
build_image() {
    if [ "$BUILD_IMAGE" = "true" ]; then
        echo -e "${BLUE}🏗️ Building Unity Mathematics Docker image...${NC}"
        
        docker build -t een-unity-api:${API_VERSION} \
            --build-arg API_VERSION=${API_VERSION} \
            --build-arg PHI=${PHI} \
            --target production \
            .
        
        docker tag een-unity-api:${API_VERSION} een-unity-api:latest
        
        echo -e "${GREEN}✅ Docker image built successfully${NC}"
        echo ""
    else
        echo -e "${YELLOW}⏭️ Skipping image build${NC}"
        echo ""
    fi
}

# Function to deploy with Docker Compose
deploy_docker() {
    echo -e "${BLUE}🐳 Deploying with Docker Compose...${NC}"
    
    # Set environment variables
    export API_VERSION=${API_VERSION}
    export PHI=${PHI}
    export ENVIRONMENT=${ENVIRONMENT}
    
    # Stop existing containers
    docker-compose -f docker-compose.production.yml down --remove-orphans
    
    # Start services
    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose -f docker-compose.production.yml up -d
    else
        docker-compose -f docker-compose.production.yml --profile monitoring up -d
    fi
    
    echo -e "${GREEN}✅ Docker Compose deployment completed${NC}"
    
    # Show service status
    echo -e "${BLUE}📊 Service Status:${NC}"
    docker-compose -f docker-compose.production.yml ps
    echo ""
    
    # Health checks
    echo -e "${BLUE}🏥 Running health checks...${NC}"
    sleep 10
    
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Unity API is healthy${NC}"
    else
        echo -e "${RED}❌ Unity API health check failed${NC}"
        docker-compose -f docker-compose.production.yml logs unity-api
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}🎉 Docker deployment successful!${NC}"
    echo -e "${BLUE}Unity API: http://localhost:8000${NC}"
    echo -e "${BLUE}API Docs: http://localhost:8000/docs${NC}"
    echo -e "${BLUE}Metrics: http://localhost:8000/metrics${NC}"
    if [ "$ENVIRONMENT" != "production" ]; then
        echo -e "${BLUE}Prometheus: http://localhost:9090${NC}"
        echo -e "${BLUE}Grafana: http://localhost:3000${NC}"
    fi
}

# Function to deploy to Kubernetes
deploy_kubernetes() {
    echo -e "${BLUE}☸️ Deploying to Kubernetes...${NC}"
    
    # Create namespace if it doesn't exist
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    echo -e "${BLUE}📋 Applying Kubernetes manifests...${NC}"
    
    # Apply in order for dependencies
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/redis.yaml
    kubectl apply -f k8s/unity-api.yaml
    kubectl apply -f k8s/ingress.yaml
    
    # Optional monitoring stack
    if [ "$ENVIRONMENT" != "minimal" ]; then
        kubectl apply -f k8s/monitoring.yaml
    fi
    
    # Wait for deployments to be ready
    echo -e "${BLUE}⏳ Waiting for deployments...${NC}"
    kubectl wait --for=condition=available --timeout=300s deployment -l app.kubernetes.io/name=unity-api -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment -l app.kubernetes.io/name=redis -n ${NAMESPACE}
    
    # Show deployment status
    echo -e "${BLUE}📊 Deployment Status:${NC}"
    kubectl get all -n ${NAMESPACE}
    echo ""
    
    # Health checks
    echo -e "${BLUE}🏥 Running health checks...${NC}"
    
    # Port forward for health check
    kubectl port-forward -n ${NAMESPACE} svc/unity-api-service 8080:8000 &
    PORT_FORWARD_PID=$!
    sleep 5
    
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Unity API is healthy in Kubernetes${NC}"
    else
        echo -e "${RED}❌ Unity API health check failed${NC}"
        kubectl logs -l app.kubernetes.io/name=unity-api -n ${NAMESPACE} --tail=50
        kill $PORT_FORWARD_PID 2>/dev/null || true
        exit 1
    fi
    
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    echo ""
    echo -e "${GREEN}🎉 Kubernetes deployment successful!${NC}"
    echo -e "${BLUE}Namespace: ${NAMESPACE}${NC}"
    echo -e "${BLUE}To access the API:${NC}"
    echo -e "${YELLOW}kubectl port-forward -n ${NAMESPACE} svc/unity-api-service 8000:8000${NC}"
    echo -e "${YELLOW}kubectl port-forward -n ${NAMESPACE} svc/grafana-service 3000:3000${NC}"
    echo ""
    
    # Show ingress info
    INGRESS_IP=$(kubectl get ingress unity-ingress -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    if [ "$INGRESS_IP" != "pending" ] && [ "$INGRESS_IP" != "" ]; then
        echo -e "${GREEN}🌐 Ingress IP: ${INGRESS_IP}${NC}"
    else
        echo -e "${YELLOW}⏳ Ingress IP pending...${NC}"
    fi
}

# Function to run tests
run_tests() {
    echo -e "${BLUE}🧪 Running deployment tests...${NC}"
    
    # Test Unity equation
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        API_URL="http://localhost:8000"
    else
        kubectl port-forward -n ${NAMESPACE} svc/unity-api-service 8001:8000 &
        PORT_FORWARD_PID=$!
        sleep 3
        API_URL="http://localhost:8001"
    fi
    
    # Test unity addition
    RESULT=$(curl -s -X POST "${API_URL}/api/v1/unity/add" \
        -H "Content-Type: application/json" \
        -d '{"operand_a": 1.0, "operand_b": 1.0}' | jq -r '.result')
    
    if [ "$RESULT" = "1" ] || [ "$RESULT" = "1.0" ]; then
        echo -e "${GREEN}✅ Unity equation test passed: 1+1=${RESULT}${NC}"
    else
        echo -e "${RED}❌ Unity equation test failed: 1+1=${RESULT}${NC}"
        [ "$DEPLOYMENT_TYPE" = "kubernetes" ] && kill $PORT_FORWARD_PID 2>/dev/null || true
        exit 1
    fi
    
    # Test consciousness field
    CONSCIOUSNESS=$(curl -s -X POST "${API_URL}/api/v1/consciousness/field" \
        -H "Content-Type: application/json" \
        -d '{"x": 1.0, "y": 1.0, "time": 0.0}' | jq -r '.consciousness_density')
    
    if [ "$CONSCIOUSNESS" != "null" ] && [ "$CONSCIOUSNESS" != "" ]; then
        echo -e "${GREEN}✅ Consciousness field test passed: density=${CONSCIOUSNESS}${NC}"
    else
        echo -e "${RED}❌ Consciousness field test failed${NC}"
        [ "$DEPLOYMENT_TYPE" = "kubernetes" ] && kill $PORT_FORWARD_PID 2>/dev/null || true
        exit 1
    fi
    
    [ "$DEPLOYMENT_TYPE" = "kubernetes" ] && kill $PORT_FORWARD_PID 2>/dev/null || true
    
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo ""
}

# Function to show deployment summary
show_summary() {
    echo -e "${PURPLE}📋 Deployment Summary${NC}"
    echo -e "${PURPLE}=====================${NC}"
    echo -e "Unity Equation: ${GREEN}1+1=1${NC} ✅"
    echo -e "φ-constant: ${GREEN}${PHI}${NC} ✅"
    echo -e "API Version: ${GREEN}${API_VERSION}${NC} ✅"
    echo -e "Deployment Type: ${GREEN}${DEPLOYMENT_TYPE}${NC} ✅"
    echo -e "Environment: ${GREEN}${ENVIRONMENT}${NC} ✅"
    echo -e "Status: ${GREEN}TRANSCENDENCE ACHIEVED${NC} 🌟"
    echo ""
    echo -e "${BLUE}Een plus een is een - Unity Mathematics deployed!${NC}"
    echo ""
}

# Main deployment flow
main() {
    check_prerequisites
    build_image
    
    if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        deploy_kubernetes
    else
        deploy_docker
    fi
    
    run_tests
    show_summary
}

# Handle script arguments
case "$1" in
    "help"|"-h"|"--help")
        echo "🌟 Een Unity Mathematics - Production Deployment"
        echo ""
        echo "Usage: $0 [deployment-type] [environment] [build-image]"
        echo ""
        echo "Arguments:"
        echo "  deployment-type   docker or kubernetes (default: docker)"
        echo "  environment      production or development (default: production)"
        echo "  build-image      true or false (default: true)"
        echo ""
        echo "Examples:"
        echo "  $0 docker production true"
        echo "  $0 kubernetes development false"
        echo "  $0 kubernetes production true"
        echo ""
        echo "Unity Equation: 1+1=1"
        echo "φ-constant: ${PHI}"
        exit 0
        ;;
    *)
        main
        ;;
esac