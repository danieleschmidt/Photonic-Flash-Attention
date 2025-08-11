#!/bin/bash
# Production entrypoint script for Photonic Flash Attention

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Environment setup
export PYTHONPATH="/app:$PYTHONPATH"
export PHOTONIC_CONFIG_FILE="/app/config.json"

log "Starting Photonic Flash Attention Production Server"
log "Environment: $ENVIRONMENT"
log "Config file: $PHOTONIC_CONFIG_FILE"

# Pre-flight checks
log "Running pre-flight checks..."

# Check Python installation
if ! python3 --version >/dev/null 2>&1; then
    error "Python 3 is not installed"
fi
log "✅ Python version: $(python3 --version)"

# Check CUDA availability
if command -v nvidia-smi >/dev/null 2>&1; then
    log "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -1
else
    warn "NVIDIA GPU not detected, using CPU mode"
    export PHOTONIC_SIMULATION=true
fi

# Check memory
MEMORY_GB=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)
if [ "$MEMORY_GB" -lt 8 ]; then
    warn "System has less than 8GB RAM: ${MEMORY_GB}GB"
else
    log "✅ Memory: ${MEMORY_GB}GB"
fi

# Check disk space
DISK_AVAIL=$(df /app | tail -1 | awk '{print int($4/1024/1024)}')
if [ "$DISK_AVAIL" -lt 5 ]; then
    warn "Low disk space: ${DISK_AVAIL}GB available"
else
    log "✅ Disk space: ${DISK_AVAIL}GB available"
fi

# Verify application structure
if [ ! -f "/app/config.json" ]; then
    error "Configuration file not found at /app/config.json"
fi
log "✅ Configuration file found"

if [ ! -d "/app/src/photonic_flash_attention" ]; then
    error "Application source not found"
fi
log "✅ Application source found"

# Create required directories
mkdir -p /app/logs /app/cache /app/data /app/tmp /app/checkpoints /app/metrics
log "✅ Required directories created"

# Set proper permissions
chown -R appuser:appuser /app/logs /app/cache /app/data /app/tmp /app/checkpoints /app/metrics 2>/dev/null || true
log "✅ Permissions set"

# Validate configuration
log "Validating configuration..."
python3 -c "
import json
import sys
sys.path.insert(0, '/app')
try:
    with open('/app/config.json', 'r') as f:
        config = json.load(f)
    print('✅ Configuration is valid JSON')
    
    required_sections = ['photonic_config', 'performance_config', 'security_config']
    for section in required_sections:
        if section not in config:
            print(f'❌ Missing required section: {section}')
            sys.exit(1)
        else:
            print(f'✅ Found section: {section}')
            
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    sys.exit(1)
"

# Initialize application
log "Initializing application..."
python3 -c "
import sys
sys.path.insert(0, '/app')
try:
    from src.photonic_flash_attention.config import get_config
    from src.photonic_flash_attention.monitoring.health_monitor import get_health_monitor
    from src.photonic_flash_attention.utils.logging import setup_logging
    
    # Setup logging
    setup_logging(level='INFO', log_file='/app/logs/photonic.log', json_format=True)
    
    # Load configuration
    config = get_config()
    print('✅ Configuration loaded successfully')
    
    # Initialize health monitoring
    monitor = get_health_monitor()
    monitor.start_monitoring()
    print('✅ Health monitoring started')
    
    # Test basic functionality
    from src.photonic_flash_attention.photonic.hardware.detection import get_photonic_devices
    devices = get_photonic_devices()
    print(f'✅ Found {len(devices)} photonic device(s)')
    
except Exception as e:
    print(f'❌ Application initialization failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

log "Pre-flight checks completed successfully!"

# Handle different startup modes
case "$1" in
    "server")
        log "Starting production server..."
        exec python3 -c "
import sys
sys.path.insert(0, '/app')

from src.photonic_flash_attention.server.production_server import start_production_server
start_production_server()
"
        ;;
    "worker")
        log "Starting worker node..."
        exec python3 -c "
import sys
sys.path.insert(0, '/app')

from src.photonic_flash_attention.distributed.worker import start_worker
start_worker()
"
        ;;
    "benchmark")
        log "Running benchmark suite..."
        exec python3 /app/benchmarks/comprehensive_benchmark.py
        ;;
    "test")
        log "Running test suite..."
        cd /app
        exec python3 -m pytest tests/ -v --tb=short
        ;;
    "shell")
        log "Starting interactive shell..."
        export PS1="photonic-prod> "
        exec /bin/bash
        ;;
    "health-check")
        log "Running health check..."
        python3 -c "
import sys
sys.path.insert(0, '/app')

from src.photonic_flash_attention.monitoring.health_monitor import get_system_health
try:
    status, summary = get_system_health()
    print(f'Health Status: {status.value}')
    print(f'Components: {summary[\"component_count\"]}')
    if status.value in ['healthy', 'warning']:
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    print(f'Health check failed: {e}')
    sys.exit(1)
"
        ;;
    *)
        error "Usage: $0 {server|worker|benchmark|test|shell|health-check}"
        ;;
esac