# Een Framework - Global Access Guide

## 🚀 Quick Start

### 1. Initial Setup
```bash
# Run the comprehensive setup script
python setup_global_access.py
```

This will:
- Install all dependencies
- Configure global access
- Set up remote access capabilities
- Create development tools
- Configure cloud deployment
- Run initial tests

### 2. Start Background Services
```bash
# Start all services in background
python start_een_background.py
```

This launches:
- API Server (port 8000)
- Dashboard (port 8501)
- MCP Server (port 3000)
- Monitoring System

### 3. Access from Anywhere

#### Command Line Access
```bash
# Global command (after setup)
een

# Or directly
python een_global.py
```

#### Python Import
```python
# From anywhere in your system
import sys
sys.path.append('/path/to/een')
from src.core.unity_mathematics import UnityMathematics
from src.consciousness.consciousness_engine import ConsciousnessEngine
```

#### Web Access
- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## 🌍 Global Access Features

### Command Line Interface
```bash
# Interactive Een shell
een

# Available commands:
# - unity: Run unity mathematics demo
# - consciousness: Run consciousness engine demo
# - viz: Launch visualization dashboard
# - help: Show available commands
# - exit: Exit the shell
```

### Python API
```python
from src.core.unity_mathematics import UnityMathematics
from src.consciousness.consciousness_engine import ConsciousnessEngine

# Initialize systems
unity = UnityMathematics()
consciousness = ConsciousnessEngine()

# Run demonstrations
unity.demo()
consciousness.demo()

# Access specific functionality
result = unity.calculate_unity_equation()
consciousness_state = consciousness.get_state()
```

### HTTP API
```bash
# Health check
curl http://localhost:8000/health

# Unity operations
curl -X POST http://localhost:8000/unity \
  -H "Content-Type: application/json" \
  -d '{"operation": "demo"}'

# Consciousness operations
curl -X POST http://localhost:8000/consciousness \
  -H "Content-Type: application/json" \
  -d '{"operation": "demo"}'
```

## 🔧 Management Commands

### Service Management
```bash
# Check status
python een_monitor.py --status

# Start services only
python een_monitor.py --start-services

# Stop all services
python start_een_background.py --stop

# Restart all services
python start_een_background.py --restart
```

### Background Monitoring
```bash
# Start monitoring in background
python een_monitor.py --daemon

# Monitor with custom config
python een_monitor.py --config custom_config.json
```

### Development Tools
```bash
# Run tests
pytest tests/

# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## ☁️ Cloud Deployment

### Deploy to Cloud Platforms
```bash
# Deploy to specific platform
python cloud_deploy.py --platform aws
python cloud_deploy.py --platform gcp
python cloud_deploy.py --platform azure
python cloud_deploy.py --platform heroku
python cloud_deploy.py --platform railway
python cloud_deploy.py --platform render

# Deploy to all platforms
python cloud_deploy.py --platform all
```

### Docker Deployment
```bash
# Build and run with Docker
docker-compose up -d

# Access services
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## 📊 Monitoring & Health

### Real-time Monitoring
The background monitor provides:
- Service health checks
- Performance metrics
- Automatic restart on failure
- Alert notifications
- Log management

### Performance Metrics
- CPU usage
- Memory usage
- Disk usage
- Network I/O
- Process-specific metrics

### Alert Configuration
Configure alerts in `monitor_config.json`:
```json
{
  "alerts": {
    "enabled": true,
    "email": "your@email.com",
    "webhook": "https://your-webhook-url",
    "discord_webhook": "https://discord.com/api/webhooks/..."
  }
}
```

## 🛠️ Development Workflow

### 1. Local Development
```bash
# Start development environment
python launch_een.py

# Choose from menu:
# 1. Unity Mathematics
# 2. Consciousness Engine
# 3. Visualization Dashboard
# 4. Remote Server
# 5. Development Tools
```

### 2. Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src

# Run specific test
pytest tests/test_unity_mathematics.py
```

### 3. Code Quality
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/

# Pre-commit hooks (automatic)
git commit -m "Your commit message"
```

## 📁 Project Structure

```
Een/
├── src/                          # Main source code
│   ├── core/                     # Core mathematics
│   ├── consciousness/            # Consciousness engine
│   └── utils/                    # Utilities
├── viz/                          # Visualization dashboards
├── tests/                        # Test suite
├── config/                       # Configuration files
├── logs/                         # Log files
├── docs/                         # Documentation
├── requirements.txt              # Dependencies
├── setup_global_access.py        # Setup script
├── start_een_background.py       # Background startup
├── een_monitor.py                # Monitoring system
├── cloud_deploy.py               # Cloud deployment
├── een_global.py                 # Global entry point
└── een_server.py                 # API server
```

## 🔄 Auto-Restart & Reliability

### Service Auto-Restart
- Services automatically restart on failure
- Configurable restart limits
- Health monitoring with alerts
- Graceful shutdown handling

### Background Operation
- All services run in background
- System startup integration
- Process management
- Resource monitoring

## 🌐 Remote Access

### API Endpoints
- `GET /` - Framework information
- `GET /health` - Health check
- `POST /unity` - Unity mathematics operations
- `POST /consciousness` - Consciousness operations
- `GET /docs` - API documentation

### Authentication
```python
# Add authentication headers
headers = {
    "Authorization": "Bearer your-token",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/unity",
    json={"operation": "demo"},
    headers=headers
)
```

## 📈 Performance Optimization

### Resource Management
- Automatic resource monitoring
- Performance threshold alerts
- Process optimization
- Memory management

### Scaling
- Horizontal scaling support
- Load balancing configuration
- Cloud deployment ready
- Container orchestration

## 🔒 Security

### Best Practices
- Input validation
- Error handling
- Logging and monitoring
- Secure defaults
- Regular updates

### Configuration
```json
{
  "security": {
    "enable_auth": true,
    "rate_limiting": true,
    "cors_origins": ["http://localhost:3000"],
    "max_request_size": "10MB"
  }
}
```

## 🚨 Troubleshooting

### Common Issues

#### Service Not Starting
```bash
# Check logs
tail -f logs/api_server_stderr.log

# Check port availability
netstat -tulpn | grep :8000

# Restart service
python start_een_background.py --restart
```

#### Global Access Not Working
```bash
# Check PATH
echo $PATH

# Add to PATH manually
export PATH="$PATH:/path/to/een"

# Test global command
which een
```

#### Performance Issues
```bash
# Check system resources
python een_monitor.py --status

# View performance metrics
cat logs/metrics.json

# Restart with resource limits
python start_een_background.py --config low_resources.json
```

### Log Files
- `logs/api_server_*.log` - API server logs
- `logs/dashboard_*.log` - Dashboard logs
- `logs/mcp_server_*.log` - MCP server logs
- `logs/een_monitor_*.log` - Monitor logs
- `logs/metrics.json` - Performance metrics

## 📚 Advanced Usage

### Custom Configuration
```json
{
  "monitoring": {
    "interval": 60,
    "health_check_interval": 120,
    "performance_interval": 600
  },
  "services": {
    "api_server": {
      "port": 9000,
      "auto_restart": true,
      "max_restarts": 10
    }
  }
}
```

### Integration Examples
```python
# Integrate with other systems
import requests
from een import UnityMathematics

# Use Een in your application
unity = UnityMathematics()
result = unity.calculate_unity_equation()

# Send to external system
requests.post("https://api.external.com/data", json=result)
```

### Custom Extensions
```python
# Extend Een functionality
from src.core.unity_mathematics import UnityMathematics

class CustomUnity(UnityMathematics):
    def custom_calculation(self, data):
        # Your custom logic
        return self.process_unity_data(data)

# Use custom extension
custom = CustomUnity()
result = custom.custom_calculation(my_data)
```

## 🎯 Next Steps

1. **Explore the Framework**: Run `een` and try different commands
2. **Check the Dashboard**: Visit http://localhost:8501
3. **Review API Docs**: Visit http://localhost:8000/docs
4. **Deploy to Cloud**: Use `python cloud_deploy.py --platform all`
5. **Monitor Performance**: Use `python een_monitor.py --status`
6. **Contribute**: Follow the development workflow

## 📞 Support

- **Documentation**: Check `docs/` directory
- **Issues**: Use GitHub issues
- **Logs**: Check `logs/` directory
- **Status**: Run `python een_monitor.py --status`

---

**🎉 Congratulations!** Your Een framework is now running globally and accessible from anywhere. The background services will keep running and auto-restart if needed, ensuring your framework is always available. 