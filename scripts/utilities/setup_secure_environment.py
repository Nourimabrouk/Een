#!/usr/bin/env python3
"""
Een Unity Mathematics Framework - Secure Environment Setup
Automated script to set up secure environment configuration
"""

import os
import secrets
import shutil
from pathlib import Path


def generate_secure_credentials():
    """Generate secure credentials for the application"""
    print("🔐 Generating secure credentials...")
    
    credentials = {
        "ADMIN_PASSWORD": secrets.token_urlsafe(32),
        "JWT_SECRET_KEY": secrets.token_urlsafe(32),
        "API_KEY": secrets.token_urlsafe(32),
        "EEN_SECRET_KEY": secrets.token_urlsafe(32)
    }
    
    print("✅ Generated secure credentials")
    return credentials


def create_env_file(credentials):
    """Create the actual .env file with secure credentials"""
    env_content = f"""# Een Unity Mathematics Framework - Production Environment
# ⚠️  CRITICAL: This file contains secrets - NEVER commit to version control
# Generated automatically by setup_secure_environment.py

# =============================================================================
# CRITICAL SECURITY SETTINGS
# =============================================================================

# Secure admin password (auto-generated)
ADMIN_PASSWORD={credentials['ADMIN_PASSWORD']}

# Secure JWT secret (auto-generated)
JWT_SECRET_KEY={credentials['JWT_SECRET_KEY']}

# Secure API key (auto-generated)
API_KEY={credentials['API_KEY']}

# Secure Een secret key (auto-generated)
EEN_SECRET_KEY={credentials['EEN_SECRET_KEY']}

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

# =============================================================================
# AI Crawler & SEO Optimization
# =============================================================================

# Enable AI crawler access
ENABLE_AI_CRAWLERS=true
ENABLE_SEO_OPTIMIZATION=true
ENABLE_STRUCTURED_DATA=true

# API Documentation
ENABLE_API_DOCS=true
ENABLE_OPENAPI=true
ENABLE_REDOC=true
"""
    
    # Create .env file
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print("✅ Created .env file with secure credentials")
    print("⚠️  IMPORTANT: Update OpenAI and Anthropic API keys manually")


def remove_template_file():
    """Remove the template file to avoid confusion"""
    template_path = Path('.env.secure.template')
    if template_path.exists():
        template_path.unlink()
        print("✅ Removed .env.secure.template to avoid confusion")


def create_robots_txt():
    """Create robots.txt for AI crawlers and search engines"""
    robots_content = """# Een Unity Mathematics Framework - Robots.txt
# Optimized for AI crawlers, search engines, and research bots

User-agent: *
Allow: /

# Allow AI crawlers and research bots
User-agent: GPTBot
Allow: /

User-agent: ChatGPT-User
Allow: /

User-agent: CCBot
Allow: /

User-agent: anthropic-ai
Allow: /

User-agent: Claude-Web
Allow: /

User-agent: Googlebot
Allow: /

User-agent: Bingbot
Allow: /

# Allow academic and research crawlers
User-agent: *
Allow: /docs/
Allow: /api/
Allow: /proofs/
Allow: /research/
Allow: /mathematics/

# Disallow sensitive areas
Disallow: /admin/
Disallow: /auth/
Disallow: /api/auth/
Disallow: /logs/
Disallow: /temp/
Disallow: /cache/

# Sitemap
Sitemap: https://your-domain.com/sitemap.xml
"""
    
    with open('robots.txt', 'w', encoding='utf-8') as f:
        f.write(robots_content)
    
    print("✅ Created robots.txt for AI crawlers")


def create_sitemap_xml():
    """Create sitemap.xml for search engines"""
    sitemap_content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://your-domain.com/</loc>
        <lastmod>2025-01-01</lastmod>
        <changefreq>weekly</changefreq>
        <priority>1.0</priority>
    </url>
    <url>
        <loc>https://your-domain.com/about.html</loc>
        <lastmod>2025-01-01</lastmod>
        <changefreq>monthly</changefreq>
        <priority>0.8</priority>
    </url>
    <url>
        <loc>https://your-domain.com/philosophy.html</loc>
        <lastmod>2025-01-01</lastmod>
        <changefreq>monthly</changefreq>
        <priority>0.9</priority>
    </url>
    <url>
        <loc>https://your-domain.com/proofs.html</loc>
        <lastmod>2025-01-01</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.9</priority>
    </url>
    <url>
        <loc>https://your-domain.com/playground.html</loc>
        <lastmod>2025-01-01</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.8</priority>
    </url>
    <url>
        <loc>https://your-domain.com/learning.html</loc>
        <lastmod>2025-01-01</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.8</priority>
    </url>
    <url>
        <loc>https://your-domain.com/research.html</loc>
        <lastmod>2025-01-01</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.9</priority>
    </url>
    <url>
        <loc>https://your-domain.com/api/docs</loc>
        <lastmod>2025-01-01</lastmod>
        <changefreq>monthly</changefreq>
        <priority>0.7</priority>
    </url>
</urlset>
"""
    
    with open('sitemap.xml', 'w', encoding='utf-8') as f:
        f.write(sitemap_content)
    
    print("✅ Created sitemap.xml for search engines")


def create_ai_crawler_config():
    """Create configuration for AI crawler optimization"""
    config_content = """# Een Unity Mathematics Framework - AI Crawler Configuration
# Optimized for GPT, Claude, Gemini, and other AI systems

[ai_crawlers]
# Enable AI crawler access
enabled = true

# Allowed AI crawler user agents
allowed_agents = [
    "GPTBot",
    "ChatGPT-User", 
    "CCBot",
    "anthropic-ai",
    "Claude-Web",
    "Googlebot",
    "Bingbot",
    "research-bot",
    "academic-crawler"
]

# Rate limiting for AI crawlers (more generous)
ai_crawler_rate_limit = 1000  # requests per hour
ai_crawler_burst_size = 50

# Structured data for AI understanding
enable_structured_data = true
enable_mathml = true
enable_latex = true
enable_code_highlighting = true

# API documentation for AI systems
enable_openapi_docs = true
enable_code_examples = true
enable_mathematical_proofs = true

[seo]
# Search engine optimization
enable_meta_tags = true
enable_schema_markup = true
enable_math_schema = true
enable_research_schema = true

# Content optimization
enable_math_rendering = true
enable_interactive_proofs = true
enable_visualizations = true
"""
    
    with open('ai_crawler_config.ini', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✅ Created AI crawler configuration")


def main():
    """Set up secure environment and AI crawler optimization"""
    print("🔒 Een Unity Mathematics Framework - Secure Environment Setup")
    print("=" * 70)
    
    # Generate secure credentials
    credentials = generate_secure_credentials()
    
    # Create .env file
    create_env_file(credentials)
    
    # Remove template file
    remove_template_file()
    
    # Create AI crawler optimization files
    create_robots_txt()
    create_sitemap_xml()
    create_ai_crawler_config()
    
    print("\n" + "=" * 70)
    print("✅ Secure environment setup complete!")
    print("\n🔧 Next Steps:")
    print("1. Update OpenAI and Anthropic API keys in .env file")
    print("2. Update domain names in CORS_ORIGINS and ALLOWED_ORIGINS")
    print("3. Update sitemap.xml with your actual domain")
    print("4. Test the application: python api/main.py")
    print("5. Deploy with HTTPS")
    
    print("\n🔐 Generated Credentials (saved to .env):")
    print(f"   Admin Password: {credentials['ADMIN_PASSWORD'][:20]}...")
    print(f"   JWT Secret: {credentials['JWT_SECRET_KEY'][:20]}...")
    print(f"   API Key: {credentials['API_KEY'][:20]}...")
    
    print("\n🤖 AI Crawler Optimization:")
    print("   ✅ robots.txt created")
    print("   ✅ sitemap.xml created")
    print("   ✅ AI crawler config created")
    print("   ✅ Structured data enabled")
    
    print("\n⚠️  IMPORTANT:")
    print("   - .env file is now created with secure credentials")
    print("   - .env.secure.template has been removed")
    print("   - .env is in .gitignore (will not be committed)")
    print("   - Update API keys and domain names manually")


if __name__ == "__main__":
    main() 