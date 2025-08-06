#!/usr/bin/env python3
"""
Een Unity Mathematics Framework - Critical Security Fixes
Automated script to fix the most critical security vulnerabilities
"""

import shutil
from pathlib import Path


def backup_file(file_path):
    """Create a backup of the original file"""
    backup_path = str(file_path) + ".backup"
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backed up: {file_path} -> {backup_path}")


def fix_admin_password():
    """Fix the hardcoded admin password in security.py"""
    file_path = Path("api/security.py")
    if not file_path.exists():
        print("❌ api/security.py not found")
        return False

    backup_file(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace hardcoded admin password with secure generation
    old_pattern = r'admin_password = os\.getenv\("ADMIN_PASSWORD", "admin123"\)'
    new_pattern = (
        'admin_password = os.getenv("ADMIN_PASSWORD", secrets.token_urlsafe(32))'
    )

    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Fixed hardcoded admin password")
        return True
    else:
        print("⚠️  Admin password pattern not found - may already be fixed")
        return False


def fix_cors_configuration():
    """Fix CORS configuration to be more restrictive"""
    file_path = Path("api/main.py")
    if not file_path.exists():
        print("❌ api/main.py not found")
        return False

    backup_file(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace overly permissive CORS configuration
    old_cors = """app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)"""

    new_cors = """app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000", 
        "https://nourimabrouk.github.io",
        "https://your-production-domain.com"  # Replace with your actual domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)"""

    if old_cors in content:
        content = content.replace(old_cors, new_cors)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Fixed CORS configuration")
        return True
    else:
        print("⚠️  CORS configuration pattern not found - may already be fixed")
        return False


def fix_trusted_host_configuration():
    """Fix trusted host configuration"""
    file_path = Path("api/main.py")
    if not file_path.exists():
        print("❌ api/main.py not found")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace overly permissive trusted host configuration
    old_trusted = """app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]  # Configure appropriately for production
)"""

    new_trusted = """app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=[
        "localhost",
        "127.0.0.1",
        "your-production-domain.com"  # Replace with your actual domain
    ]
)"""

    if old_trusted in content:
        content = content.replace(old_trusted, new_trusted)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Fixed trusted host configuration")
        return True
    else:
        print("⚠️  Trusted host configuration pattern not found - may already be fixed")
        return False


def add_security_headers():
    """Add security headers middleware"""
    file_path = Path("api/main.py")
    if not file_path.exists():
        print("❌ api/main.py not found")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if security headers middleware already exists
    if '@app.middleware("http")' in content and "add_security_headers" in content:
        print("⚠️  Security headers middleware already exists")
        return False

    # Add security headers middleware after the existing middleware
    security_middleware = '''
@app.middleware("http")
async def add_security_headers(request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response
'''

    # Insert after the security middleware section
    if "# Security" in content:
        content = content.replace("# Security", "# Security" + security_middleware)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Added security headers middleware")
        return True
    else:
        print("⚠️  Could not find insertion point for security headers")
        return False


def fix_dummy_api_key():
    """Remove dummy API key from validation script"""
    file_path = Path("validation/validate_ai_integration.py")
    if not file_path.exists():
        print("❌ validation/validate_ai_integration.py not found")
        return False

    backup_file(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace dummy API key with proper error handling
    old_pattern = (
        r'os\.environ\["OPENAI_API_KEY"\] = "sk-test-dummy-key-for-validation"'
    )
    new_pattern = 'raise ValueError("OPENAI_API_KEY environment variable required for validation")'

    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Removed dummy API key")
        return True
    else:
        print("⚠️  Dummy API key pattern not found - may already be fixed")
        return False


def create_secure_env_template():
    """Create a secure .env template"""
    env_template = """# Een Unity Mathematics Framework - Secure Environment Configuration
# Copy this file to .env and fill in your actual values

# =============================================================================
# CRITICAL SECURITY SETTINGS
# =============================================================================

# Generate a secure admin password (run: python -c "import secrets; print(secrets.token_urlsafe(32))")
ADMIN_PASSWORD=your_secure_admin_password_here_min_32_chars

# Generate a secure JWT secret (run: python -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_SECRET_KEY=your_super_secret_jwt_key_here_min_32_chars

# Generate a secure API key (run: python -c "import secrets; print(secrets.token_urlsafe(32))")
API_KEY=your_secure_api_key_here

# =============================================================================
# API Configuration
# =============================================================================

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ASSISTANT_ID=your_assistant_id_here
OPENAI_MODEL=gpt-4o-mini

# Anthropic API (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# =============================================================================
# Security Configuration
# =============================================================================

# JWT Configuration
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# API Security
REQUIRE_AUTH=true
API_KEY_REQUIRED=true

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST_SIZE=10
MAX_REQUEST_SIZE=10485760

# CORS Configuration - UPDATE THESE FOR YOUR DOMAIN
CORS_ORIGINS=http://localhost:3000,http://localhost:8000,https://nourimabrouk.github.io,https://your-production-domain.com
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,https://your-production-domain.com

# =============================================================================
# Environment
# =============================================================================

ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1

# =============================================================================
# Features
# =============================================================================

ENABLE_CODE_EXECUTION=false  # SECURITY: Keep disabled in production
ENABLE_STREAMING=true
ENABLE_CITATIONS=true
ENABLE_VOICE=false
"""

    with open(".env.secure.template", "w", encoding="utf-8") as f:
        f.write(env_template)

    print("✅ Created secure .env template: .env.secure.template")


def main():
    """Run all security fixes"""
    print("🔒 Een Unity Mathematics Framework - Security Fixes")
    print("=" * 60)

    fixes_applied = 0

    # Apply security fixes
    if fix_admin_password():
        fixes_applied += 1

    if fix_cors_configuration():
        fixes_applied += 1

    if fix_trusted_host_configuration():
        fixes_applied += 1

    if add_security_headers():
        fixes_applied += 1

    if fix_dummy_api_key():
        fixes_applied += 1

    create_secure_env_template()

    print("\n" + "=" * 60)
    print(f"✅ Applied {fixes_applied} security fixes")
    print("\n🔧 Next Steps:")
    print("1. Update .env.secure.template with your actual values")
    print("2. Copy .env.secure.template to .env")
    print(
        '3. Generate secure passwords using: python -c "import secrets; print(secrets.token_urlsafe(32))"'
    )
    print("4. Update CORS_ORIGINS and ALLOWED_ORIGINS with your actual domains")
    print("5. Test the application thoroughly")
    print("6. Deploy with HTTPS")
    print("\n⚠️  IMPORTANT: This fixes the most critical issues, but you should still:")
    print("   - Review all console.log statements in JavaScript files")
    print("   - Sanitize innerHTML usage in frontend code")
    print("   - Implement proper input validation")
    print("   - Set up monitoring and logging")
    print("   - Regular security audits")


if __name__ == "__main__":
    main()
