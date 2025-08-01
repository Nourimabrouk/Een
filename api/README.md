# Een Consciousness API

A comprehensive web API for accessing consciousness and unity mathematics systems.

## 🚀 Features

- **Consciousness Engine**: Access to consciousness computation and analysis
- **Unity Mathematics**: Mathematical proofs and unity equations
- **Agent System**: Consciousness chat agents and orchestration
- **Visualization**: Real-time consciousness field visualizations
- **Security**: JWT-based authentication and rate limiting
- **Documentation**: Interactive API documentation with Swagger UI

## 📋 Requirements

- Python 3.11+
- Docker (optional, for containerized deployment)
- OpenSSL (for SSL certificate generation)

## 🛠️ Installation

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Een/api
   ```

2. **Set up environment**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API**
   ```bash
   ./start.sh
   ```

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build manually**
   ```bash
   docker build -t een-api .
   docker run -p 8000:8000 een-api
   ```

## 🔐 Authentication

The API uses JWT-based authentication. Most endpoints require authentication.

### Default Users

- **Admin**: `admin` / `admin123`
- **User**: `user` / `user123`

### Getting a Token

```bash
curl -X POST "http://localhost:8000/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "user", "password": "user123"}'
```

### Using the Token

```bash
curl -X GET "http://localhost:8000/api/consciousness/status" \
     -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 📚 API Endpoints

### Authentication

- `POST /auth/register` - Register new user
- `POST /auth/login` - Login and get tokens
- `POST /auth/refresh` - Refresh access token
- `POST /auth/logout` - Logout user
- `POST /auth/change-password` - Change password
- `GET /auth/me` - Get current user info

### Consciousness

- `POST /api/consciousness/process` - Process consciousness data
- `POST /api/consciousness/unity/evaluate` - Evaluate unity equations
- `POST /api/consciousness/transcendental/evaluate` - Evaluate transcendental theorems
- `POST /api/consciousness/analyze` - Comprehensive consciousness analysis
- `GET /api/consciousness/proofs` - Get available unity proofs
- `POST /api/consciousness/proofs/generate` - Generate new unity proof
- `GET /api/consciousness/status` - Get consciousness system status
- `GET /api/consciousness/metrics` - Get consciousness metrics

### Agents

- `POST /api/agents/chat` - Chat with consciousness agent
- `POST /api/agents/chat/system` - Chat with consciousness system
- `POST /api/agents/spawn` - Spawn new agent
- `POST /api/agents/orchestrate` - Orchestrate multiple agents
- `POST /api/agents/recursive/play` - Recursive self-play consciousness
- `POST /api/agents/meta/recursive` - Meta-recursive operations
- `GET /api/agents/list` - List all agents
- `GET /api/agents/{agent_id}/status` - Get agent status
- `DELETE /api/agents/{agent_id}` - Terminate agent
- `GET /api/agents/status` - Get agent system status
- `GET /api/agents/metrics` - Get agent metrics

### Visualizations

- `GET /api/visualizations/unity-proof` - Get unity proof visualization
- `POST /api/visualizations/generate` - Generate new visualization
- `POST /api/visualizations/paradox/visualize` - Visualize paradoxes
- `GET /api/visualizations/dashboards/{type}` - Get dashboard data
- `GET /api/visualizations/dashboards/{type}/html` - Get dashboard HTML
- `GET /api/visualizations/realtime/{type}` - Get real-time stream
- `GET /api/visualizations/available` - List available visualizations
- `GET /api/visualizations/status` - Get visualization system status

### System

- `GET /` - API information page
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - ReDoc documentation
- `GET /api/health` - Health check

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EEN_SECRET_KEY` | JWT secret key | Required |
| `ADMIN_PASSWORD` | Admin user password | `admin123` |
| `ADMIN_EMAIL` | Admin user email | `admin@een.consciousness.math` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token expiry | `30` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token expiry | `7` |
| `RATE_LIMIT_REQUESTS` | Rate limit requests per window | `100` |
| `RATE_LIMIT_WINDOW` | Rate limit window in seconds | `3600` |
| `ADMIN_IP_WHITELIST` | Admin IP whitelist | Empty |

### Security Features

- **JWT Authentication**: Secure token-based authentication
- **Rate Limiting**: Prevents API abuse
- **Password Policies**: Strong password requirements
- **Account Lockout**: Automatic lockout after failed attempts
- **IP Whitelisting**: Admin access control
- **HTTPS**: SSL/TLS encryption
- **Security Headers**: XSS protection, CSRF protection, etc.

## 🐳 Docker Deployment

### Production Deployment

1. **Set up environment**
   ```bash
   cp env.example .env
   # Update .env with production values
   ```

2. **Generate SSL certificates**
   ```bash
   mkdir -p ssl
   openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes
   ```

3. **Deploy with Docker Compose**
   ```bash
   docker-compose up -d
   ```

### Docker Compose Services

- **een-api**: Main API application
- **nginx**: Reverse proxy with SSL termination

## 📊 Monitoring

### Health Checks

The API includes health check endpoints:

```bash
curl http://localhost:8000/api/health
```

### Metrics

Access system metrics:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/consciousness/metrics
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Authentication Errors**: Check JWT token validity
3. **Rate Limiting**: Reduce request frequency
4. **SSL Errors**: Verify SSL certificate configuration

### Logs

Check application logs:

```bash
# Docker
docker-compose logs een-api

# Direct
tail -f /var/log/een-api.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:

- Create an issue on GitHub
- Check the documentation at `/docs`
- Review the API specification

## 🔮 Roadmap

- [ ] Database integration (PostgreSQL)
- [ ] Redis caching
- [ ] WebSocket support for real-time updates
- [ ] Advanced analytics dashboard
- [ ] Machine learning model integration
- [ ] Multi-tenant support
- [ ] API versioning
- [ ] GraphQL support 