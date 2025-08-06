#!/usr/bin/env python3
"""
Een Unity Mathematics - Secure Secrets Setup
Automated script to configure API keys and secrets safely
"""

import os
import secrets
import getpass
from pathlib import Path

def generate_secure_key(length=32):
    """Generate a cryptographically secure random key."""
    return secrets.token_hex(length)

def setup_environment():
    """Setup environment variables interactively and securely."""
    print("🔐 Een Unity Mathematics - Secure Setup")
    print("=" * 50)
    print()
    
    env_file = Path(".env")
    
    # Read existing .env or create new
    env_vars = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
    
    print("📝 Configure your API keys (press Enter to skip):")
    print()
    
    # OpenAI API Key
    current_openai = env_vars.get('OPENAI_API_KEY', 'your-key-here')
    if current_openai in ['your-key-here', 'sk-proj-your-key-here']:
        openai_key = getpass.getpass("🤖 Enter your OpenAI API key (hidden input): ").strip()
        if openai_key:
            env_vars['OPENAI_API_KEY'] = openai_key
        else:
            print("   ⚠️  Skipping OpenAI key - AI features will be disabled")
            env_vars['OPENAI_API_KEY'] = 'sk-disabled'
    else:
        print(f"   ✅ OpenAI key already configured: {current_openai[:8]}...")
    
    # Anthropic API Key  
    current_anthropic = env_vars.get('ANTHROPIC_API_KEY', 'your-key-here')
    if current_anthropic in ['your-key-here', 'sk-ant-your-key-here']:
        anthropic_key = getpass.getpass("🧠 Enter your Anthropic API key (hidden input): ").strip()
        if anthropic_key:
            env_vars['ANTHROPIC_API_KEY'] = anthropic_key
        else:
            print("   ⚠️  Skipping Anthropic key - Claude features will be disabled")
            env_vars['ANTHROPIC_API_KEY'] = 'sk-disabled'
    else:
        print(f"   ✅ Anthropic key already configured: {current_anthropic[:8]}...")
    
    print()
    print("🔑 Generating secure secrets...")
    
    # Generate secure secrets if needed
    if env_vars.get('SECRET_KEY', '').endswith('change-in-production'):
        env_vars['SECRET_KEY'] = generate_secure_key(32)
        print("   ✅ Generated new SECRET_KEY")
    
    if env_vars.get('JWT_SECRET_KEY', '').endswith('change-in-production'):
        env_vars['JWT_SECRET_KEY'] = generate_secure_key(32) 
        print("   ✅ Generated new JWT_SECRET_KEY")
    
    # Set other defaults
    defaults = {
        'API_HOST': '0.0.0.0',
        'API_PORT': '8000',
        'API_WORKERS': '1',
        'DEBUG': 'false',
        'DATABASE_URL': 'sqlite:///./een_unity.db',
        'REDIS_URL': 'redis://localhost:6379',
        'JWT_ALGORITHM': 'HS256',
        'ACCESS_TOKEN_EXPIRE_MINUTES': '30',
        'PHI_VALUE': '1.618033988749895',
        'UNITY_CONSTANT': '1',
        'CONSCIOUSNESS_LEVEL': '1.618',
        'CACHE_TTL': '3600',
        'MAX_WORKERS': '4',
        'TIMEOUT': '300',
        'ENABLE_METRICS': 'true',
        'METRICS_PORT': '9090',
        'ENV': 'production',
        'LOG_LEVEL': 'INFO',
        'CORS_ORIGINS': '*',
        'AUTO_OPEN_BROWSER': 'true',
        'ENABLE_GPU': 'true',
    }
    
    for key, value in defaults.items():
        if key not in env_vars:
            env_vars[key] = value
    
    # Write .env file
    print()
    print("💾 Writing configuration to .env file...")
    
    with open(env_file, 'w') as f:
        f.write("# Een Unity Mathematics Configuration\n")
        f.write("# Generated automatically - DO NOT commit to version control\n\n")
        
        f.write("# API Server Configuration\n")
        f.write(f"API_HOST={env_vars['API_HOST']}\n")
        f.write(f"API_PORT={env_vars['API_PORT']}\n")
        f.write(f"API_WORKERS={env_vars['API_WORKERS']}\n")
        f.write(f"DEBUG={env_vars['DEBUG']}\n\n")
        
        f.write("# AI Integration Keys\n")
        f.write(f"OPENAI_API_KEY={env_vars['OPENAI_API_KEY']}\n")
        f.write(f"ANTHROPIC_API_KEY={env_vars['ANTHROPIC_API_KEY']}\n\n")
        
        f.write("# Database Configuration\n")
        f.write(f"DATABASE_URL={env_vars['DATABASE_URL']}\n")
        f.write(f"REDIS_URL={env_vars['REDIS_URL']}\n\n")
        
        f.write("# Security\n")
        f.write(f"SECRET_KEY={env_vars['SECRET_KEY']}\n")
        f.write(f"JWT_SECRET_KEY={env_vars['JWT_SECRET_KEY']}\n")
        f.write(f"JWT_ALGORITHM={env_vars['JWT_ALGORITHM']}\n")
        f.write(f"ACCESS_TOKEN_EXPIRE_MINUTES={env_vars['ACCESS_TOKEN_EXPIRE_MINUTES']}\n\n")
        
        f.write("# Unity Mathematics Constants\n")
        f.write(f"PHI_VALUE={env_vars['PHI_VALUE']}\n")
        f.write(f"UNITY_CONSTANT={env_vars['UNITY_CONSTANT']}\n")
        f.write(f"CONSCIOUSNESS_LEVEL={env_vars['CONSCIOUSNESS_LEVEL']}\n\n")
        
        f.write("# Performance Settings\n")
        f.write(f"CACHE_TTL={env_vars['CACHE_TTL']}\n")
        f.write(f"MAX_WORKERS={env_vars['MAX_WORKERS']}\n")
        f.write(f"TIMEOUT={env_vars['TIMEOUT']}\n\n")
        
        f.write("# Monitoring\n")
        f.write(f"ENABLE_METRICS={env_vars['ENABLE_METRICS']}\n")
        f.write(f"METRICS_PORT={env_vars['METRICS_PORT']}\n\n")
        
        f.write("# Application Settings\n")
        f.write(f"ENV={env_vars['ENV']}\n")
        f.write(f"LOG_LEVEL={env_vars['LOG_LEVEL']}\n")
        f.write(f"CORS_ORIGINS={env_vars['CORS_ORIGINS']}\n")
        f.write(f"AUTO_OPEN_BROWSER={env_vars['AUTO_OPEN_BROWSER']}\n")
        f.write(f"ENABLE_GPU={env_vars['ENABLE_GPU']}\n")
    
    print("   ✅ Configuration saved to .env")
    
    # Update gitignore
    gitignore_file = Path(".gitignore")
    gitignore_content = ""
    if gitignore_file.exists():
        gitignore_content = gitignore_file.read_text()
    
    if ".env" not in gitignore_content:
        with open(gitignore_file, 'a') as f:
            f.write("\n# Environment variables (contains secrets)\n")
            f.write(".env\n")
            f.write(".env.local\n")
            f.write("*.env\n")
        print("   ✅ Updated .gitignore to protect secrets")
    
    print()
    print("🌟 Setup complete!")
    print("=" * 50)
    print("Your Een Unity Mathematics platform is now configured with:")
    print(f"  • OpenAI Integration: {'✅ Enabled' if not env_vars['OPENAI_API_KEY'].endswith('disabled') else '❌ Disabled'}")
    print(f"  • Anthropic Integration: {'✅ Enabled' if not env_vars['ANTHROPIC_API_KEY'].endswith('disabled') else '❌ Disabled'}")
    print("  • Secure secrets: ✅ Generated")
    print("  • Configuration: ✅ Saved to .env")
    print("  • Security: ✅ Protected with .gitignore")
    print()
    print("🚀 Ready to launch! Run: python launch.py")
    print()

def check_existing_setup():
    """Check if setup is already complete."""
    env_file = Path(".env")
    if not env_file.exists():
        return False
    
    content = env_file.read_text()
    has_real_keys = (
        'sk-' in content and 
        not 'your-key-here' in content and 
        not 'change-in-production' in content
    )
    
    return has_real_keys

if __name__ == "__main__":
    if check_existing_setup():
        print("✅ Een Unity Mathematics is already configured!")
        print("🚀 Run 'python launch.py' to start the platform")
        print()
        response = input("Reconfigure anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup skipped.")
            exit(0)
    
    setup_environment()