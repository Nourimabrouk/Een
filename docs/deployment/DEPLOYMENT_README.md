# Een Unity Mathematics - Deployment Guide

## Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
c:/Users/Nouri/Documents/GitHub/Een/een/Scripts/activate.bat

# Install dependencies
pip install flask flask-cors fastapi uvicorn python-dotenv

# Copy environment template
copy env.example .env
```

### 2. Configure Environment
Edit `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_actual_openai_key
ANTHROPIC_API_KEY=your_actual_anthropic_key
API_KEY=your_generated_secure_key
```

### 3. Run Server
```bash
python unity_code_server.py
```

## Features

### ✅ Code Execution (ENABLED)
- **Endpoint**: `POST /api/execute`
- **Security**: Comprehensive input validation and safe execution environment
- **Access**: Available at http://localhost:5000/api/execute

### ✅ Unity Mathematics
- **Endpoint**: `POST /api/unity/calculate`
- **Operations**: Unity addition, multiplication, phi harmonics
- **Access**: Available at http://localhost:5000/api/unity/calculate

### ✅ Health Check
- **Endpoint**: `GET /api/health`
- **Status**: Server health and component availability
- **Access**: Available at http://localhost:5000/api/health

## Security Features

- ✅ **Input Validation**: Blocks dangerous code patterns
- ✅ **Safe Execution**: Restricted built-in functions only
- ✅ **Code Length Limits**: Maximum 1000 characters
- ✅ **Pattern Blocking**: Blocks `import os`, `eval`, `exec`, etc.
- ✅ **CORS Protection**: Cross-origin request handling

## API Examples

### Code Execution
```bash
curl -X POST http://localhost:5000/api/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Hello from Unity Mathematics!\")", "language": "python"}'
```

### Unity Calculation
```bash
curl -X POST http://localhost:5000/api/unity/calculate \
  -H "Content-Type: application/json" \
  -d '{"operation": "unity_add", "operands": [1.0, 1.0]}'
```

## Ready for Production

The server is now ready for:
- ✅ Public deployment
- ✅ Reddit sharing
- ✅ Safe code execution
- ✅ Unity Mathematics demonstrations

**Code execution is enabled by default with comprehensive security measures.** 