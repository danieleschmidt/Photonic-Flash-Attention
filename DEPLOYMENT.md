# Production Deployment Guide

## 🚀 Photonic Flash Attention - Production Deployment

This guide covers the complete production deployment of the Photonic Flash Attention system.

### Prerequisites

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **Git** >= 2.25
- **Linux/Unix system** (Ubuntu 20.04+ recommended)
- **NVIDIA GPU** (optional, for hardware acceleration)
- **Python 3.9+** (for development)

### Quick Deployment

```bash
# Clone the repository
git clone https://github.com/terragon-labs/photonic-flash-attention.git
cd photonic-flash-attention

# Run deployment script
./scripts/deploy.sh

# Access the application
open http://localhost:8080    # Main application
open http://localhost:8081    # Monitoring dashboard
```

### Manual Deployment Steps

#### 1. Environment Setup

```bash
# Create environment file
cp .env.example .env

# Configure environment variables
export PHOTONIC_LOG_LEVEL=INFO
export PHOTONIC_SIMULATION=1
export POSTGRES_PASSWORD=your_secure_password
```

#### 2. Build and Deploy

```bash
# Build Docker images
docker-compose build

# Start production services
docker-compose --profile production up -d

# Verify deployment
docker-compose ps
docker-compose logs -f
```

#### 3. Health Checks

```bash
# Check service health
curl http://localhost:8080/api/health
curl http://localhost:8081/api/health

# View logs
docker-compose logs photonic-attention
docker-compose logs monitoring
```

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│  Main App       │────│  Photonic HW    │
│   (Nginx)       │    │  (Python)       │    │  (Simulation)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ├─────────────────┐    │                ┌─────────────────┐
         │                 │    │                │  Monitoring     │
┌─────────────────┐       │    │                │  (Dashboard)    │
│  Grafana        │       │    │                └─────────────────┘
│  (Port 3000)    │       │    │
└─────────────────┘       │    │                ┌─────────────────┐
                          │    │                │  Database       │
┌─────────────────┐       │    └────────────────│  (PostgreSQL)   │
│  Prometheus     │       │                     └─────────────────┘
│  (Port 9090)    │       │
└─────────────────┘       │                     ┌─────────────────┐
                          └─────────────────────│  Cache          │
                                                │  (Redis)        │
                                                └─────────────────┘
```

### Service Configuration

#### Main Application
- **Port**: 8080
- **Health**: `/api/health`
- **Metrics**: `/metrics`
- **Resources**: 4 CPU, 8GB RAM

#### Monitoring Dashboard
- **Port**: 8081
- **Dashboard**: `/`
- **API**: `/api/metrics`
- **Resources**: 1 CPU, 2GB RAM

#### Database (PostgreSQL)
- **Port**: 5432
- **Database**: `photonic_metrics`
- **User**: `photonic`
- **Resources**: 2 CPU, 4GB RAM

#### Cache (Redis)
- **Port**: 6379
- **Max Memory**: 256MB
- **Policy**: `allkeys-lru`

### Monitoring and Alerting

#### Grafana Dashboards
- **System Metrics**: CPU, Memory, Disk, Network
- **Application Metrics**: Latency, Throughput, Errors
- **Photonic Metrics**: Optical Power, Temperature, Efficiency
- **Business Metrics**: Requests/sec, Device Usage

#### Prometheus Alerts
- High CPU usage (>80%)
- High memory usage (>85%)
- Application errors (>1%)
- Photonic device temperature (>70°C)
- Service downtime

#### Log Aggregation
- **Format**: JSON structured logs
- **Retention**: 30 days
- **Location**: `/app/logs/`
- **Rotation**: 10MB per file, 5 files max

### Security Considerations

#### Network Security
- All external traffic through Nginx reverse proxy
- Internal services on private Docker network
- HTTPS enabled with Let's Encrypt certificates
- Rate limiting and DDoS protection

#### Application Security
- No root processes in containers
- Read-only filesystem where possible
- Secrets managed via environment variables
- Regular security scanning with Trivy

#### Data Security
- Database encryption at rest
- Backup encryption
- Audit logging enabled
- GDPR compliance measures

### Backup and Recovery

#### Automated Backups
```bash
# Database backup (daily at 2 AM)
0 2 * * * docker-compose exec postgres pg_dump -U photonic photonic_metrics > /backups/db_$(date +%Y%m%d).sql

# Configuration backup (weekly)
0 2 * * 0 tar -czf /backups/config_$(date +%Y%m%d).tar.gz docker-compose.yml monitoring/ nginx/

# Application data backup (daily)
0 3 * * * rsync -av /app/data/ /backups/data/
```

#### Recovery Procedures
```bash
# Restore database
docker-compose exec postgres psql -U photonic -d photonic_metrics < /backups/db_20231215.sql

# Restore configuration
tar -xzf /backups/config_20231215.tar.gz

# Restart services
docker-compose restart
```

### Performance Tuning

#### System Optimization
- **Kernel**: Use Linux 5.4+ for better container performance
- **CPU**: Enable all CPU cores for Docker
- **Memory**: Allocate 16GB+ for production
- **Storage**: Use SSD for database and logs

#### Application Tuning
```bash
# Environment variables for production
PHOTONIC_LOG_LEVEL=WARNING
PHOTONIC_ENABLE_PROFILING=false
PHOTONIC_CACHE_SIZE=1000
PHOTONIC_MAX_CONCURRENT_REQUESTS=10
```

#### Database Optimization
```sql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET effective_cache_size = '4GB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
ALTER SYSTEM SET max_connections = 200;
SELECT pg_reload_conf();
```

### Scaling

#### Horizontal Scaling
```yaml
# docker-compose.override.yml for scaling
services:
  photonic-attention:
    deploy:
      replicas: 3
    
  nginx:
    depends_on:
      - photonic-attention
    environment:
      - UPSTREAM_SERVERS=photonic-attention:8080
```

#### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/

# Scale deployment
kubectl scale deployment photonic-flash-attention --replicas=5

# Monitor scaling
kubectl get pods -w
```

### Troubleshooting

#### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Increase container memory limits
   docker-compose up --scale photonic-attention=2
   ```

2. **Slow Performance**
   ```bash
   # Check resource usage
   docker stats
   
   # Enable GPU acceleration
   docker-compose -f docker-compose.gpu.yml up -d
   ```

3. **Service Discovery Issues**
   ```bash
   # Check network connectivity
   docker network ls
   docker network inspect photonic-network
   ```

#### Log Analysis
```bash
# Application logs
docker-compose logs -f photonic-attention | grep ERROR

# System resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Network issues
docker-compose exec photonic-attention netstat -tuln
```

### Maintenance

#### Regular Tasks
- **Weekly**: Update base images and redeploy
- **Monthly**: Review and rotate logs
- **Quarterly**: Security audit and penetration testing
- **Yearly**: Disaster recovery testing

#### Update Procedure
```bash
# Pull latest changes
git pull origin main

# Build new images
docker-compose build --no-cache

# Rolling update
docker-compose up -d --force-recreate
```

### Support

#### Monitoring Endpoints
- **Health Check**: `GET /api/health`
- **Metrics**: `GET /metrics` (Prometheus format)
- **Status**: `GET /api/status` (Detailed system info)

#### Contact Information
- **Support Email**: support@terragonlabs.ai
- **Documentation**: https://docs.terragonlabs.ai/photonic-flash-attention
- **Issues**: https://github.com/terragon-labs/photonic-flash-attention/issues

#### Emergency Contacts
- **On-call Engineer**: +1-555-PHOTONIC
- **DevOps Team**: devops@terragonlabs.ai
- **Security Team**: security@terragonlabs.ai

---

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2023-12-15 | Initial production release |
| 0.1.1 | 2023-12-20 | Performance improvements |
| 0.2.0 | 2024-01-15 | New monitoring features |

### License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

### Acknowledgments

Built with ❤️ by the Terragon Labs team using autonomous SDLC execution.