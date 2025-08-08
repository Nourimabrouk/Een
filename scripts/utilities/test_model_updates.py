#!/usr/bin/env python3
"""
Test script to verify AI model updates and intelligent model selection
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_model_configuration():
    """Test that model configuration files are properly set up."""
    print("🧠 Testing AI Model Configuration")
    print("=" * 50)

    # Test 1: Check AI agent configuration
    try:
        from ai_agent import CHAT_MODEL

        print(f"✅ AI Agent default model: {CHAT_MODEL}")
        assert CHAT_MODEL == "gpt-4o", f"Expected gpt-4o, got {CHAT_MODEL}"
    except Exception as e:
        print(f"❌ AI Agent config error: {e}")

    # Test 2: Check AI model manager
    try:
        from src.ai_model_manager import model_manager, get_best_model_for_request

        print("✅ AI Model Manager imported successfully")

        # Test model selection
        test_message = "Can you prove that 1+1=1 in idempotent semirings?"
        provider, model = get_best_model_for_request(test_message)
        print(f"✅ Model selection test: {model} ({provider})")

    except Exception as e:
        print(f"❌ AI Model Manager error: {e}")

    # Test 3: Check configuration file
    config_file = Path("config/ai_model_config.json")
    if config_file.exists():
        print("✅ AI Model config file exists")
        try:
            import json

            with open(config_file, "r") as f:
                config = json.load(f)
            primary_model = config["preferred_models"]["primary"]["model"]
            print(f"✅ Primary model configured: {primary_model}")
            assert primary_model == "gpt-4o", f"Expected gpt-4o, got {primary_model}"
        except Exception as e:
            print(f"❌ Config file error: {e}")
    else:
        print("❌ AI Model config file not found")

    print()


def test_model_capabilities():
    """Test model capability analysis."""
    print("🔍 Testing Model Capabilities")
    print("=" * 50)

    try:
        from src.ai_model_manager import (
            model_manager,
            analyze_request_complexity,
            get_best_model_for_request,
        )

        test_cases = [
            (
                "Can you prove that 1+1=1 in idempotent semirings?",
                "mathematical_proofs",
            ),
            (
                "How do I implement a consciousness field equation in Python?",
                "code_analysis",
            ),
            (
                "What is the philosophical meaning of unity in mathematics?",
                "philosophical_discussion",
            ),
            ("Hello, how are you today?", "general_chat"),
            (
                "Can you analyze this complex algorithm and optimize it for better performance?",
                "code_analysis",
            ),
        ]

        for message, expected_type in test_cases:
            request_type = model_manager.analyze_request_type(message)
            complexity = analyze_request_complexity(message)
            provider, model = get_best_model_for_request(message)

            print(f"Message: {message[:40]}...")
            print(f"  Expected: {expected_type}")
            print(f"  Detected: {request_type}")
            print(f"  Selected: {model} ({provider})")
            print(f"  Complexity: {complexity['complexity_score']:.2f}")
            print()

    except Exception as e:
        print(f"❌ Model capabilities test error: {e}")

    print()


def test_environment_variables():
    """Test environment variable configuration."""
    print("🔧 Testing Environment Configuration")
    print("=" * 50)

    # Check if environment variables are set
    env_vars = {
        "OPENAI_API_KEY": "OpenAI API Key",
        "ANTHROPIC_API_KEY": "Anthropic API Key (optional)",
        "CHAT_MODEL": "Chat Model (should default to gpt-4o)",
    }

    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            if var == "CHAT_MODEL":
                print(f"✅ {description}: {value}")
                if value != "gpt-4o":
                    print(f"   ⚠️  Consider setting to gpt-4o for better reasoning")
            else:
                print(f"✅ {description}: {'*' * len(value)} (set)")
        else:
            if var == "ANTHROPIC_API_KEY":
                print(f"⚠️  {description}: Not set (optional)")
            else:
                print(f"❌ {description}: Not set")

    print()


def main():
    """Run all tests."""
    print("🚀 Testing AI Model Updates")
    print("=" * 60)
    print()

    test_model_configuration()
    test_model_capabilities()
    test_environment_variables()

    print("🎉 Model update tests completed!")
    print()
    print("📋 Summary of Changes:")
    print("  • Default model changed from gpt-4o-mini to gpt-4o")
    print("  • Added intelligent model selection based on request type")
    print("  • Created AI Model Manager for optimal model selection")
    print("  • Added support for Claude Sonnet as alternative")
    print("  • Increased max_tokens from 1000 to 2000")
    print()
    print("💡 Next Steps:")
    print("  1. Set OPENAI_API_KEY environment variable")
    print("  2. Optionally set ANTHROPIC_API_KEY for Claude access")
    print("  3. Test the chat functionality with complex reasoning questions")
    print("  4. Monitor model selection and performance")


if __name__ == "__main__":
    main()
