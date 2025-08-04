# Quick Start - Een Unity Mathematics Docker Deployment

## Prerequisites
- Docker Desktop installed and running
- Git repository cloned

## One-Command Deployment

From the project root, run:

```bash
cd deployment
docker compose up --build -d
```

## Access Your Services

Once running, access:

- **Main API**: http://localhost:8000
- **Dashboard**: http://localhost:8050
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Nginx**: http://localhost:80

## Quick Commands

```bash
# Check status
docker compose ps

# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild everything
docker compose down -v && docker compose up --build -d
```

## What's Included

✅ FastAPI backend with Unity Mathematics endpoints
✅ Streamlit dashboard with consciousness field visualization
✅ PostgreSQL database
✅ Redis cache
✅ Prometheus monitoring
✅ Grafana dashboards
✅ Nginx reverse proxy
✅ Health checks and metrics

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /api/unity/status` - Unity system status
- `POST /api/unity/calculate` - Unity calculations
- `GET /api/consciousness/field` - Consciousness field data

Your Een Unity Mathematics system is now ready! 🧮✨ 