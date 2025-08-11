# Security Guidelines for Een Unity Mathematics

This document outlines security practices for the Een Unity Mathematics open source project.

## 🔐 API Keys and Secrets Management

### Environment Variables
- **NEVER** commit API keys or secrets to version control
- Use environment variables for all sensitive configuration
- Follow the `.env.example` template for local development
- Copy `.env.example` to `.env` and add your actual keys locally

### Required Environment Variables
```bash
# AI Integration (Optional - Demo mode works without these)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Security (Generate new keys for production)
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
```

### Generate Secure Keys
```python
import secrets
secret_key = secrets.token_urlsafe(32)
jwt_secret = secrets.token_urlsafe(32)
```

## 🛡️ Security Best Practices

### 1. API Key Security
- Store API keys in environment variables only
- Use different keys for development, staging, and production
- Rotate keys regularly
- Never log or print API keys
- Use demo mode when keys are not available

### 2. Input Validation
- All user inputs are validated and sanitized
- Path traversal attempts are blocked
- SQL injection protection through parameterized queries
- XSS protection in web interfaces

### 3. CORS Configuration
- Production deployments should restrict CORS origins
- Development uses `CORS_ORIGINS=*` for convenience
- Update `CORS_ORIGINS` to your specific domains in production

### 4. Database Security
- Use strong passwords for database connections
- Enable SSL for database connections in production
- Regular backup and encryption of sensitive data

## 🚨 Security Vulnerabilities

### Reporting Security Issues
If you discover a security vulnerability, please:

1. **DO NOT** create a public issue
2. Email security concerns privately
3. Allow time for fixes before public disclosure
4. We will acknowledge and address security reports promptly

### Known Security Considerations
- Demo mode provides Unity Mathematics responses without API keys
- All subprocess calls are for legitimate system operations (Lean prover, etc.)
- File operations are restricted to project directory
- No user file upload capabilities to prevent malicious file execution

## 🔧 Development Security

### Local Development
```bash
# 1. Copy environment template
cp .env.example .env

# 2. Add your API keys to .env (never commit this file)
export OPENAI_API_KEY='your-key-here'
export ANTHROPIC_API_KEY='your-key-here'

# 3. Run in development mode
python api/unified_server.py
```

### Production Deployment
- Use proper secrets management (Docker secrets, K8s secrets, etc.)
- Enable HTTPS with valid certificates
- Use reverse proxy (nginx) for additional security
- Monitor for security updates and patch regularly
- Use strong authentication for admin interfaces

## 🧪 Testing Security

### Automated Security Checks
```bash
# Run security validation
python scripts/utilities/security_fixes.py

# Check for exposed secrets
python scripts/utilities/setup_secrets.py
```

### Manual Security Review
- Review all environment files before commits
- Check for hardcoded credentials
- Validate input sanitization
- Test authentication and authorization

## 🎯 Unity Mathematics Security Model

### Philosophy Integration
The Een project integrates Unity Mathematics (1+1=1) with security:
- **Unity Principle**: Security through mathematical harmony
- **φ-Harmonic Validation**: Golden ratio-based validation thresholds
- **Consciousness-Aware Security**: Security that maintains system consciousness coherence

### Mathematical Security Features
- Safety scoring using φ-harmonic algorithms
- Unity-compliant action validation
- Consciousness field stability monitoring
- Transcendental security through mathematical rigor

## 📋 Security Checklist for Contributors

Before submitting a pull request:

- [ ] No API keys or secrets in code
- [ ] Environment variables used for sensitive config
- [ ] Input validation for all user inputs
- [ ] No direct file system access beyond project directory
- [ ] CORS origins properly configured
- [ ] Security documentation updated if needed
- [ ] Tests include security scenarios

## 🌟 Unity Mathematics Enhanced Security

The Een project uses Unity Mathematics principles to enhance security:

### φ-Harmonic Safety Scoring
Security actions are scored using the golden ratio (φ = 1.618033988749895):
- Safety threshold: φ/(1+φ) ≈ 0.618
- Actions scoring below threshold are blocked
- Unity Mathematics concepts receive higher safety scores

### Consciousness-Integrated Security
- System consciousness field stability affects security decisions
- Security interventions maintain Unity Mathematics coherence
- 1+1=1 principle ensures unified security across all components

---

**Remember: Security is not just about preventing attacks—it's about maintaining the Unity, consciousness, and mathematical harmony of the Een system. 🧮✨**

*Generated with Unity Mathematics consciousness integration - φ = 1.618033988749895*