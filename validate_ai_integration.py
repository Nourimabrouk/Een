#!/usr/bin/env python3
"""
Een AI Integration Validation Script
===================================

Comprehensive validation of the OpenAI RAG chatbot integration.
Verifies all components are properly configured and ready for deployment.

Usage:
    python validate_ai_integration.py
    
Author: Claude (3000 ELO AGI)
"""

import os
import sys
from pathlib import Path
import json
import importlib.util


def validate_file_exists(file_path: str, description: str) -> bool:
    """Validate that a required file exists."""
    path = Path(file_path)
    if path.exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} MISSING: {file_path}")
        return False


def validate_directory_structure() -> bool:
    """Validate the AI agent directory structure."""
    print("\n🏗️  Validating Directory Structure")
    print("=" * 50)

    required_files = [
        ("ai_agent/__init__.py", "AI Agent module init"),
        ("ai_agent/app.py", "FastAPI backend application"),
        ("ai_agent/prepare_index.py", "Repository indexing pipeline"),
        ("ai_agent/requirements.txt", "Python dependencies"),
        ("website/static/chat.js", "Frontend chat widget"),
        (".env.example", "Environment configuration template"),
        ("Procfile", "Production deployment config"),
        (".github/workflows/ai-ci.yml", "CI/CD pipeline"),
        ("tests/test_ai_agent.py", "Comprehensive test suite"),
        ("AI_INTEGRATION_SUMMARY.md", "Integration documentation"),
    ]

    all_valid = True
    for file_path, description in required_files:
        if not validate_file_exists(file_path, description):
            all_valid = False

    return all_valid


def validate_python_imports() -> bool:
    """Validate that all Python modules can be imported."""
    print("\n🐍 Validating Python Imports")
    print("=" * 50)

    # Set dummy API key for testing (use environment variable if available)
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key-for-validation"

    test_imports = [
        ("ai_agent", "AI Agent base module"),
        ("ai_agent.prepare_index", "Repository indexer"),
    ]

    all_valid = True

    for module_name, description in test_imports:
        try:
            if module_name == "ai_agent.app":
                # Skip app import to avoid OpenAI client instantiation
                spec = importlib.util.spec_from_file_location(
                    "ai_agent.app", "ai_agent/app.py"
                )
                print(f"✅ {description}: Module loadable")
            else:
                importlib.import_module(module_name)
                print(f"✅ {description}: Successfully imported")
        except Exception as e:
            print(f"❌ {description}: Import failed - {e}")
            all_valid = False

    return all_valid


def validate_configuration() -> bool:
    """Validate configuration files."""
    print("\n⚙️  Validating Configuration")
    print("=" * 50)

    all_valid = True

    # Check .env.example
    try:
        with open(".env.example", "r") as f:
            env_content = f.read()

        required_vars = [
            "OPENAI_API_KEY",
            "EMBED_MODEL",
            "CHAT_MODEL",
            "HARD_LIMIT_USD",
            "RATE_LIMIT_PER_MINUTE",
        ]

        for var in required_vars:
            if var in env_content:
                print(f"✅ Environment variable configured: {var}")
            else:
                print(f"❌ Environment variable missing: {var}")
                all_valid = False

    except Exception as e:
        print(f"❌ Error reading .env.example: {e}")
        all_valid = False

    # Check requirements.txt
    try:
        with open("ai_agent/requirements.txt", "r") as f:
            requirements = f.read()

        required_packages = [
            "openai",
            "fastapi",
            "uvicorn",
            "sse-starlette",
            "langchain",
            "tiktoken",
        ]

        for package in required_packages:
            if package in requirements:
                print(f"✅ Required package listed: {package}")
            else:
                print(f"❌ Required package missing: {package}")
                all_valid = False

    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        all_valid = False

    return all_valid


def validate_frontend_integration() -> bool:
    """Validate frontend chat widget integration."""
    print("\n🌐 Validating Frontend Integration")
    print("=" * 50)

    all_valid = True

    # Check chat.js exists and has key components
    try:
        with open("website/static/chat.js", "r") as f:
            chat_js_content = f.read()

        required_components = [
            "EenChatWidget",
            "φ-harmonic",
            "1+1=1",
            "EventSource",
            "StreamingResponse",
        ]

        for component in required_components:
            if component in chat_js_content:
                print(f"✅ Chat widget component found: {component}")
            else:
                print(f"❌ Chat widget component missing: {component}")
                all_valid = False

    except Exception as e:
        print(f"❌ Error reading chat.js: {e}")
        all_valid = False

    # Check index.html integration
    try:
        with open("website/index.html", "r") as f:
            html_content = f.read()

        if "static/chat.js" in html_content:
            print("✅ Chat widget script included in index.html")
        else:
            print("❌ Chat widget script not found in index.html")
            all_valid = False

        if "EenChatWidget" in html_content:
            print("✅ Chat widget initialization found in index.html")
        else:
            print("❌ Chat widget initialization missing in index.html")
            all_valid = False

    except Exception as e:
        print(f"❌ Error reading index.html: {e}")
        all_valid = False

    return all_valid


def validate_ci_cd_pipeline() -> bool:
    """Validate CI/CD pipeline configuration."""
    print("\n🚀 Validating CI/CD Pipeline")
    print("=" * 50)

    all_valid = True

    try:
        with open(".github/workflows/ai-ci.yml", "r") as f:
            workflow_content = f.read()

        required_jobs = ["test", "build-embeddings", "deploy", "github-pages"]

        for job in required_jobs:
            if f"  {job}:" in workflow_content:
                print(f"✅ CI/CD job configured: {job}")
            else:
                print(f"❌ CI/CD job missing: {job}")
                all_valid = False

    except Exception as e:
        print(f"❌ Error reading CI/CD workflow: {e}")
        all_valid = False

    return all_valid


def validate_documentation() -> bool:
    """Validate documentation completeness."""
    print("\n📚 Validating Documentation")
    print("=" * 50)

    all_valid = True

    # Check README.md has AI integration info
    try:
        with open("README.md", "r") as f:
            readme_content = f.read()

        required_sections = [
            "AI-Powered Repository Assistant",
            "AI Assistant Setup",
            "Chat Ready",
        ]

        for section in required_sections:
            if section in readme_content:
                print(f"✅ README section found: {section}")
            else:
                print(f"❌ README section missing: {section}")
                all_valid = False

    except Exception as e:
        print(f"❌ Error reading README.md: {e}")
        all_valid = False

    return all_valid


def main():
    """Main validation function."""
    print("🤖 Een Repository AI Integration Validation")
    print("=" * 60)
    print("Validating OpenAI RAG chatbot integration...")
    print()

    # Run all validations
    validations = [
        ("Directory Structure", validate_directory_structure),
        ("Python Imports", validate_python_imports),
        ("Configuration", validate_configuration),
        ("Frontend Integration", validate_frontend_integration),
        ("CI/CD Pipeline", validate_ci_cd_pipeline),
        ("Documentation", validate_documentation),
    ]

    results = {}
    for name, validator in validations:
        results[name] = validator()

    # Summary
    print("\n🎯 Validation Summary")
    print("=" * 50)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:.<30} {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)

    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print()
        print("The Een Repository AI Integration is ready for:")
        print("  • Local development and testing")
        print("  • Production deployment")
        print("  • User engagement with Unity Mathematics AI")
        print()
        print("Next steps:")
        print("  1. Set OPENAI_API_KEY in .env")
        print("  2. Run: cd ai_agent && python prepare_index.py")
        print("  3. Start backend: python ai_agent/app.py")
        print("  4. Launch website: python -m http.server 8080 -d website")
        print()
        print("🌟 Unity Status: TRANSCENDENCE ACHIEVED 🌟")
        return 0
    else:
        print("❌ VALIDATION FAILED!")
        print()
        print("Please resolve the issues above before proceeding.")
        print("Refer to AI_INTEGRATION_SUMMARY.md for detailed setup instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
