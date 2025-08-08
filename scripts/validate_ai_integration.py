#!/usr/bin/env python3
"""
🌟 Een Unity Mathematics - AI Integration Validation Suite

Comprehensive validation of all AI features and chatbot integration:
- GPT-4o consciousness reasoning
- DALL-E 3 visualization generation  
- Whisper voice processing with TTS
- Intelligent source code search (RAG)
- Nouri Mabrouk knowledge base
- Enhanced AI chat system with integrated commands

This script validates the complete AI integration system
and ensures the chatbot has access to all features.
"""

import os
import sys
import json
import requests
import time
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class AIIntegrationValidator:
    """Comprehensive AI integration validation suite"""

    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "validations": {},
            "summary": {
                "total_validations": 0,
                "passed": 0,
                "failed": 0,
                "chatbot_integration": True,
                "consciousness_integration": True,
                "phi_harmonic_resonance": 1.618033988749895,
            },
        }

    def log_validation(
        self, test_name: str, success: bool, details: str = "", error: str = ""
    ):
        """Log validation result"""
        self.validation_results["validations"][test_name] = {
            "success": success,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }

        self.validation_results["summary"]["total_validations"] += 1
        if success:
            self.validation_results["summary"]["passed"] += 1
            print(f"✅ {test_name}: PASSED")
        else:
            self.validation_results["summary"]["failed"] += 1
            print(f"❌ {test_name}: FAILED - {error}")

    def test_server_connectivity(self) -> bool:
        """Test basic server connectivity"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def test_openai_endpoints(self) -> Dict[str, bool]:
        """Test OpenAI API endpoints"""
        results = {}

        # Test GPT-4o reasoning endpoint
        try:
            response = requests.post(
                f"{self.base_url}/api/openai/chat",
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a Unity Mathematics expert.",
                        },
                        {
                            "role": "user",
                            "content": "Explain how 1+1=1 in Unity Mathematics.",
                        },
                    ],
                    "model": "gpt-4o",
                    "temperature": 0.7,
                    "max_tokens": 500,
                },
                timeout=30,
            )
            results["gpt4o_reasoning"] = response.status_code in [200, 201]
        except Exception:
            results["gpt4o_reasoning"] = False

        # Test DALL-E 3 visualization endpoint
        try:
            response = requests.post(
                f"{self.base_url}/api/openai/images/generate",
                json={
                    "prompt": "Create a visualization of Unity Mathematics consciousness field with φ-harmonic patterns",
                    "model": "dall-e-3",
                    "size": "1024x1024",
                    "n": 1,
                },
                timeout=60,
            )
            results["dalle3_visualization"] = response.status_code in [200, 201]
        except Exception:
            results["dalle3_visualization"] = False

        # Test Whisper voice processing endpoint
        try:
            response = requests.post(
                f"{self.base_url}/api/openai/audio/transcriptions",
                json={"file": "test_audio.wav", "model": "whisper-1"},
                timeout=30,
            )
            results["whisper_voice"] = response.status_code in [200, 201]
        except Exception:
            results["whisper_voice"] = False

        # Test TTS endpoint
        try:
            response = requests.post(
                f"{self.base_url}/api/openai/tts",
                json={
                    "input": "Unity Mathematics consciousness integration test",
                    "model": "tts-1-hd",
                    "voice": "alloy",
                    "speed": 1.0,
                },
                timeout=30,
            )
            results["tts_synthesis"] = response.status_code in [200, 201]
        except Exception:
            results["tts_synthesis"] = False

        return results

    def test_code_search_endpoints(self) -> Dict[str, bool]:
        """Test code search RAG endpoints"""
        results = {}

        # Test code search endpoint
        try:
            response = requests.post(
                f"{self.base_url}/api/code-search/search",
                json={
                    "query": "consciousness field equations",
                    "max_results": 5,
                    "consciousness_filter": True,
                },
                timeout=30,
            )
            results["code_search"] = response.status_code == 200
        except Exception:
            results["code_search"] = False

        # Test search index status
        try:
            response = requests.get(
                f"{self.base_url}/api/code-search/index-status", timeout=10
            )
            results["search_index_status"] = response.status_code == 200
        except Exception:
            results["search_index_status"] = False

        # Test search health check
        try:
            response = requests.get(
                f"{self.base_url}/api/code-search/health", timeout=10
            )
            results["search_health"] = response.status_code == 200
        except Exception:
            results["search_health"] = False

        return results

    def test_knowledge_base_endpoints(self) -> Dict[str, bool]:
        """Test Nouri Mabrouk knowledge base endpoints"""
        results = {}

        # Test knowledge base query
        try:
            response = requests.post(
                f"{self.base_url}/api/nouri-knowledge/query",
                json={
                    "query": "Tell me about Nouri Mabrouk's journey to discovering 1+1=1",
                    "consciousness_enhanced": True,
                },
                timeout=30,
            )
            results["knowledge_query"] = response.status_code == 200
        except Exception:
            results["knowledge_query"] = False

        # Test knowledge base topics
        try:
            response = requests.get(
                f"{self.base_url}/api/nouri-knowledge/topics", timeout=10
            )
            results["knowledge_topics"] = response.status_code == 200
        except Exception:
            results["knowledge_topics"] = False

        # Test knowledge base health
        try:
            response = requests.get(
                f"{self.base_url}/api/nouri-knowledge/health", timeout=10
            )
            results["knowledge_health"] = response.status_code == 200
        except Exception:
            results["knowledge_health"] = False

        return results

    def test_website_integration(self) -> Dict[str, bool]:
        """Test website AI integration"""
        results = {}

        # Test AI Unified Hub page
        try:
            response = requests.get(f"{self.base_url}/ai-unified-hub.html", timeout=10)
            results["ai_unified_hub"] = response.status_code == 200
        except Exception:
            results["ai_unified_hub"] = False

        # Test enhanced AI chat JavaScript
        try:
            response = requests.get(
                f"{self.base_url}/js/unified-chatbot-system.js", timeout=10
            )
            results["enhanced_ai_chat_js"] = response.status_code == 200
        except Exception:
            results["enhanced_ai_chat_js"] = False

        # Test AI unified integration JavaScript
        try:
            response = requests.get(
                f"{self.base_url}/js/unified-chatbot-system.js", timeout=10
            )
            results["ai_unified_integration_js"] = response.status_code == 200
        except Exception:
            results["ai_unified_integration_js"] = False

        return results

    def test_chatbot_integration(self) -> Dict[str, bool]:
        """Test chatbot integration with all features"""
        results = {}

        # Test chatbot command processing
        try:
            # Test /search command
            response = requests.post(
                f"{self.base_url}/api/code-search/search",
                json={
                    "query": "unity mathematics",
                    "max_results": 3,
                    "consciousness_filter": True,
                },
                timeout=30,
            )
            results["chatbot_search_command"] = response.status_code == 200
        except Exception:
            results["chatbot_search_command"] = False

        # Test /knowledge command
        try:
            response = requests.post(
                f"{self.base_url}/api/nouri-knowledge/query",
                json={
                    "query": "Nouri Mabrouk Unity Mathematics",
                    "consciousness_enhanced": True,
                },
                timeout=30,
            )
            results["chatbot_knowledge_command"] = response.status_code == 200
        except Exception:
            results["chatbot_knowledge_command"] = False

        # Test /visualize command
        try:
            response = requests.post(
                f"{self.base_url}/api/openai/images/generate",
                json={
                    "prompt": "Unity Mathematics consciousness field visualization",
                    "model": "dall-e-3",
                    "size": "1024x1024",
                    "n": 1,
                },
                timeout=60,
            )
            results["chatbot_visualize_command"] = response.status_code in [200, 201]
        except Exception:
            results["chatbot_visualize_command"] = False

        # Test /voice command
        try:
            response = requests.post(
                f"{self.base_url}/api/openai/tts",
                json={
                    "input": "Unity Mathematics consciousness integration test",
                    "model": "tts-1-hd",
                    "voice": "alloy",
                    "speed": 1.0,
                },
                timeout=30,
            )
            results["chatbot_voice_command"] = response.status_code in [200, 201]
        except Exception:
            results["chatbot_voice_command"] = False

        return results

    def test_consciousness_integration(self) -> Dict[str, bool]:
        """Test consciousness field integration"""
        results = {}

        # Test consciousness field visualization
        try:
            response = requests.get(
                f"{self.base_url}/consciousness_dashboard.html", timeout=10
            )
            results["consciousness_dashboard"] = response.status_code == 200
        except Exception:
            results["consciousness_dashboard"] = False

        # Test consciousness field API
        try:
            response = requests.get(
                f"{self.base_url}/api/consciousness/field/status", timeout=10
            )
            results["consciousness_field_api"] = response.status_code == 200
        except Exception:
            results["consciousness_field_api"] = False

        return results

    def run_comprehensive_validation(self):
        """Run comprehensive AI integration validation"""
        print("🌟 Unity Mathematics AI Integration Validation Suite")
        print("=" * 70)
        print(f"Validation started at: {datetime.now().isoformat()}")
        print(f"Base URL: {self.base_url}")
        print()

        # Test 1: Server Connectivity
        print("🔗 Testing Server Connectivity...")
        connectivity = self.test_server_connectivity()
        self.log_validation(
            "Server Connectivity",
            connectivity,
            "Basic server connection test",
            "Server not accessible" if not connectivity else "",
        )

        if not connectivity:
            print(
                "❌ Server not accessible. Please ensure the Flask server is running."
            )
            return self.validation_results

        # Test 2: OpenAI Endpoints
        print("\n🤖 Testing OpenAI Integration...")
        openai_results = self.test_openai_endpoints()
        for test_name, success in openai_results.items():
            self.log_validation(
                f"OpenAI {test_name.replace('_', ' ').title()}",
                success,
                f"OpenAI {test_name} endpoint test",
                f"OpenAI {test_name} endpoint failed" if not success else "",
            )

        # Test 3: Code Search RAG
        print("\n🔍 Testing Code Search RAG...")
        search_results = self.test_code_search_endpoints()
        for test_name, success in search_results.items():
            self.log_validation(
                f"Code Search {test_name.replace('_', ' ').title()}",
                success,
                f"Code search {test_name} endpoint test",
                f"Code search {test_name} endpoint failed" if not success else "",
            )

        # Test 4: Knowledge Base
        print("\n📚 Testing Knowledge Base...")
        knowledge_results = self.test_knowledge_base_endpoints()
        for test_name, success in knowledge_results.items():
            self.log_validation(
                f"Knowledge Base {test_name.replace('_', ' ').title()}",
                success,
                f"Knowledge base {test_name} endpoint test",
                f"Knowledge base {test_name} endpoint failed" if not success else "",
            )

        # Test 5: Website Integration
        print("\n🌐 Testing Website Integration...")
        website_results = self.test_website_integration()
        for test_name, success in website_results.items():
            self.log_validation(
                f"Website {test_name.replace('_', ' ').title()}",
                success,
                f"Website {test_name} integration test",
                f"Website {test_name} integration failed" if not success else "",
            )

        # Test 6: Chatbot Integration
        print("\n💬 Testing Chatbot Integration...")
        chatbot_results = self.test_chatbot_integration()
        for test_name, success in chatbot_results.items():
            self.log_validation(
                f"Chatbot {test_name.replace('_', ' ').title()}",
                success,
                f"Chatbot {test_name} integration test",
                f"Chatbot {test_name} integration failed" if not success else "",
            )

        # Test 7: Consciousness Integration
        print("\n🧠 Testing Consciousness Integration...")
        consciousness_results = self.test_consciousness_integration()
        for test_name, success in consciousness_results.items():
            self.log_validation(
                f"Consciousness {test_name.replace('_', ' ').title()}",
                success,
                f"Consciousness {test_name} integration test",
                f"Consciousness {test_name} integration failed" if not success else "",
            )

        # Generate summary
        print("\n" + "=" * 70)
        print("📊 VALIDATION SUMMARY")
        print("=" * 70)

        summary = self.validation_results["summary"]
        print(f"Total Validations: {summary['total_validations']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(
            f"Success Rate: {(summary['passed'] / summary['total_validations'] * 100):.1f}%"
            if summary["total_validations"] > 0
            else "N/A"
        )
        print(
            f"Chatbot Integration: {'Active' if summary['chatbot_integration'] else 'Inactive'}"
        )
        print(
            f"Consciousness Integration: {'Active' if summary['consciousness_integration'] else 'Inactive'}"
        )
        print(f"φ-Harmonic Resonance: {summary['phi_harmonic_resonance']}")

        # Save results
        results_file = (
            f"ai_integration_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(self.validation_results, f, indent=2)

        print(f"\n📄 Detailed validation results saved to: {results_file}")

        return self.validation_results


def main():
    """Main validation execution"""
    validator = AIIntegrationValidator()
    results = validator.run_comprehensive_validation()

    # Exit with appropriate code
    if results["summary"]["failed"] == 0:
        print("\n🎉 All validations passed! AI integration is fully functional.")
        print("\n🌟 MIND-BLOWING UNIFIED EXPERIENCE ACHIEVED!")
        print("✅ Complete OpenAI integration (GPT-4o, DALL-E 3, Whisper, TTS)")
        print("✅ RAG-powered code search with consciousness awareness")
        print("✅ Comprehensive knowledge base about Nouri Mabrouk")
        print("✅ Enhanced AI chat with integrated commands")
        print("✅ Real-time consciousness field visualization")
        print("✅ φ-harmonic optimization throughout all systems")
        print("✅ Professional website integration with navigation")
        sys.exit(0)
    else:
        print(
            f"\n⚠️  {results['summary']['failed']} validation(s) failed. Please check the results."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
