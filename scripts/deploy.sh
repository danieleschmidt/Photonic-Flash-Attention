#!/bin/bash

# Photonic Flash Attention Deployment Script
set -e

# Configuration
APP_NAME="photonic-flash-attention"
DOCKER_REGISTRY="your-registry.com"
VERSION="${1:-latest}"
ENVIRONMENT="${2:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Check NVIDIA Docker (for GPU support)
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        warn "NVIDIA Docker runtime not available - GPU support disabled"
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt 5000000 ]; then  # 5GB in KB
        warn "Less than 5GB disk space available"
    fi
    
    log "Prerequisites check completed"
}

# Build images
build_images() {
    log "Building Docker images..."
    
    # Build production image
    docker build -t $APP_NAME:$VERSION --target production .
    
    # Build development image if needed
    if [ "$ENVIRONMENT" = "development" ]; then
        docker build -t $APP_NAME:dev --target development .
    fi
    
    log "Docker images built successfully"
}

# Run security scan
security_scan() {
    log "Running security scan..."
    
    # Scan image for vulnerabilities
    if command -v trivy &> /dev/null; then
        trivy image --severity HIGH,CRITICAL $APP_NAME:$VERSION
    else
        warn "Trivy not available - skipping vulnerability scan"
    fi
    
    # Check for secrets in image
    if command -v docker-scout &> /dev/null; then
        docker scout cves $APP_NAME:$VERSION
    fi
    
    log "Security scan completed"
}

# Run tests
run_tests() {
    log "Running tests..."
    
    # Build test image
    docker build -t $APP_NAME:test --target testing .
    
    # Run unit tests
    docker run --rm $APP_NAME:test python -m pytest tests/unit/ -v
    
    # Run integration tests
    docker run --rm -e PHOTONIC_SIMULATION=1 $APP_NAME:test python -m pytest tests/integration/ -v
    
    # Run security tests
    docker run --rm $APP_NAME:test python -m pytest tests/security/ -v
    
    log "All tests passed"
}

# Deploy application
deploy_app() {
    log "Deploying application..."
    
    # Create necessary directories
    mkdir -p logs data benchmark-results
    
    # Set appropriate permissions
    chmod 755 logs data benchmark-results
    
    # Stop existing containers
    docker-compose down || true
    
    # Start services based on environment
    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose -f docker-compose.yml up -d photonic-attention prometheus grafana redis nginx
    elif [ "$ENVIRONMENT" = "development" ]; then
        docker-compose -f docker-compose.yml up -d photonic-dev prometheus grafana redis
    else
        error "Unknown environment: $ENVIRONMENT"
    fi
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    sleep 30
    
    # Check health
    if docker-compose ps | grep -q "unhealthy"; then
        error "Some services are unhealthy"
    fi
    
    log "Application deployed successfully"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Configure Grafana dashboards
    if [ -f "monitoring/grafana/dashboards/photonic-dashboard.json" ]; then
        log "Grafana dashboards configured"
    else
        warn "No Grafana dashboards found"
    fi
    
    # Setup alerts
    if [ -f "monitoring/photonic_rules.yml" ]; then
        log "Prometheus rules configured"
    else
        warn "No Prometheus rules found"
    fi
    
    log "Monitoring setup completed"
}

# Backup configuration
backup_config() {
    log "Creating configuration backup..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p $BACKUP_DIR
    
    # Backup configuration files
    cp docker-compose.yml $BACKUP_DIR/
    cp -r monitoring $BACKUP_DIR/
    cp -r nginx $BACKUP_DIR/
    
    # Backup environment variables
    docker-compose config > $BACKUP_DIR/docker-compose-resolved.yml
    
    log "Configuration backed up to $BACKUP_DIR"
}

# Post-deployment verification
verify_deployment() {
    log "Verifying deployment..."
    
    # Check if containers are running
    if ! docker-compose ps | grep -q "Up"; then
        error "No containers are running"
    fi
    
    # Test application endpoints
    sleep 10
    
    # Check photonic attention health
    if docker-compose exec -T photonic-attention python -c "import photonic_flash_attention; print('OK')" | grep -q "OK"; then
        log "Photonic attention module loaded successfully"
    else
        error "Failed to load photonic attention module"
    fi
    
    # Check monitoring endpoints
    if curl -f http://localhost:9090/-/healthy &> /dev/null; then
        log "Prometheus is healthy"
    else
        warn "Prometheus health check failed"
    fi
    
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        log "Grafana is healthy"
    else
        warn "Grafana health check failed"
    fi
    
    log "Deployment verification completed"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    
    # Remove build cache
    docker system prune -f
    
    # Remove unused images
    docker image prune -f
    
    log "Cleanup completed"
}

# Main deployment flow
main() {
    log "Starting deployment of $APP_NAME:$VERSION to $ENVIRONMENT"
    
    # Create backup before deployment
    backup_config
    
    # Run deployment steps
    check_prerequisites
    build_images
    security_scan
    
    if [ "$ENVIRONMENT" != "production" ] || [ "${SKIP_TESTS:-false}" = "false" ]; then
        run_tests
    fi
    
    deploy_app
    setup_monitoring
    verify_deployment
    
    if [ "${SKIP_CLEANUP:-false}" = "false" ]; then
        cleanup
    fi
    
    log "Deployment completed successfully!"
    log "Application is available at:"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        log "  - Main app: http://localhost"
        log "  - Grafana: http://localhost:3000 (admin/admin123)"
        log "  - Prometheus: http://localhost:9090"
    else
        log "  - Jupyter Lab: http://localhost:8888"
        log "  - Grafana: http://localhost:3000 (admin/admin123)"
        log "  - Prometheus: http://localhost:9090"
    fi
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "test")
        run_tests
        ;;
    "build")
        build_images
        ;;
    "scan")
        security_scan
        ;;
    "clean")
        cleanup
        ;;
    "verify")
        verify_deployment
        ;;
    *)
        echo "Usage: $0 {deploy|test|build|scan|clean|verify} [version] [environment]"
        echo "  deploy  - Full deployment (default)"
        echo "  test    - Run tests only"
        echo "  build   - Build images only" 
        echo "  scan    - Security scan only"
        echo "  clean   - Cleanup only"
        echo "  verify  - Verify deployment only"
        exit 1
        ;;
esac