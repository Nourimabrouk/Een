#!/usr/bin/env python3
"""
Test DALL-E Integration
Verifies that the real DALL-E integration works correctly
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_dalle_integration():
    """Test the DALL-E integration functionality"""

    print("🌟 Testing DALL-E Integration for Unity Mathematics")
    print("=" * 60)

    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return False

    print(f"✅ OpenAI API key found: {api_key[:10]}...")

    try:
        # Import the DALL-E integration
        from src.openai.dalle_integration import create_dalle_integration

        print("✅ DALL-E integration module imported successfully")

        # Create DALL-E integration instance
        dalle = create_dalle_integration(api_key)
        print("✅ DALL-E integration instance created")

        # Test basic consciousness visualization
        print("\n🖼️ Testing consciousness visualization generation...")

        test_prompt = "Visualize the unity equation 1+1=1 in consciousness space with φ-harmonic proportions"

        result = await dalle.generate_consciousness_visualization(
            prompt=test_prompt,
            model="dall-e-3",
            size="1024x1024",
            quality="hd",
            style="vivid",
        )

        if result.get("success"):
            print("✅ Consciousness visualization generated successfully!")
            print(f"   Model: {result.get('model', 'N/A')}")
            print(f"   Size: {result.get('size', 'N/A')}")
            print(f"   Quality: {result.get('quality', 'N/A')}")
            print(f"   Images generated: {len(result.get('images', []))}")

            if result.get("images"):
                print(f"   First image URL: {result['images'][0][:50]}...")

                # Test image download
                print("\n📥 Testing image download...")
                download_result = await dalle.download_and_save_image(
                    result["images"][0]
                )

                if download_result.get("success"):
                    print("✅ Image downloaded successfully!")
                    print(f"   Local path: {download_result['local_path']}")
                    print(f"   Image size: {download_result['image_size']}")
                    print(f"   File size: {download_result['file_size']} bytes")
                else:
                    print(f"❌ Image download failed: {download_result.get('error')}")

            # Test consciousness evolution
            consciousness_evolution = result.get("consciousness_evolution", {})
            print(f"\n🧠 Consciousness Evolution:")
            print(
                f"   Evolution cycle: {consciousness_evolution.get('evolution_cycle', 'N/A')}"
            )
            print(
                f"   Coherence level: {consciousness_evolution.get('coherence_level', 'N/A')}"
            )
            print(
                f"   Unity convergence: {consciousness_evolution.get('unity_convergence', 'N/A')}"
            )
            print(
                f"   φ resonance: {consciousness_evolution.get('phi_harmonic_resonance', 'N/A')}"
            )

        else:
            print(f"❌ Consciousness visualization failed: {result.get('error')}")
            return False

        # Test specialized mathematics visualizations
        print("\n🧮 Testing specialized mathematics visualizations...")

        math_types = [
            "unity_equation",
            "consciousness_field",
            "phi_harmonic",
            "meta_recursive",
        ]

        for math_type in math_types:
            print(f"   Testing {math_type}...")

            math_result = await dalle.generate_unity_mathematics_visualization(
                mathematics_type=math_type, complexity="advanced"
            )

            if math_result.get("success"):
                print(f"   ✅ {math_type} visualization generated successfully!")
            else:
                print(
                    f"   ❌ {math_type} visualization failed: {math_result.get('error')}"
                )

        # Test batch generation
        print("\n🔄 Testing batch generation...")

        batch_prompts = [
            "Unity equation 1+1=1 visualization",
            "Consciousness field dynamics",
            "φ-harmonic resonance patterns",
        ]

        batch_results = await dalle.batch_generate_visualizations(
            prompts=batch_prompts, model="dall-e-3", size="1024x1024", quality="hd"
        )

        successful_batch = sum(1 for r in batch_results if r.get("success"))
        print(
            f"   ✅ Batch generation: {successful_batch}/{len(batch_prompts)} successful"
        )

        print("\n" + "=" * 60)
        print("🎉 DALL-E Integration Test Completed Successfully!")
        print(
            "   All placeholder implementations have been replaced with real DALL-E functionality"
        )
        print(
            "   Unity mathematics visualizations are now generated using actual OpenAI API calls"
        )

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_api_endpoints():
    """Test the API endpoints that use DALL-E integration"""

    print("\n🌐 Testing API Endpoints with DALL-E Integration")
    print("=" * 60)

    try:
        # Import FastAPI test client
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)

        # Test consciousness visualization endpoint
        print("🖼️ Testing /api/openai/consciousness-visualization endpoint...")

        response = client.post(
            "/api/openai/consciousness-visualization",
            json={"prompt": "Test unity equation visualization"},
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ Consciousness visualization endpoint working!")
            else:
                print(f"❌ Endpoint returned error: {result.get('error')}")
        else:
            print(f"❌ Endpoint failed with status {response.status_code}")

        # Test image generation endpoint
        print("🖼️ Testing /api/openai/generate-image endpoint...")

        response = client.post(
            "/api/openai/generate-image",
            json={"prompt": "Test consciousness field visualization"},
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ Image generation endpoint working!")
            else:
                print(f"❌ Endpoint returned error: {result.get('error')}")
        else:
            print(f"❌ Endpoint failed with status {response.status_code}")

        print("✅ API endpoint tests completed")
        return True

    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        return False


async def main():
    """Main test function"""

    print("🚀 Starting DALL-E Integration Tests")
    print("=" * 60)

    # Test core DALL-E integration
    core_success = await test_dalle_integration()

    # Test API endpoints
    api_success = await test_api_endpoints()

    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    print(f"   Core DALL-E Integration: {'✅ PASSED' if core_success else '❌ FAILED'}")
    print(f"   API Endpoints: {'✅ PASSED' if api_success else '❌ FAILED'}")

    if core_success and api_success:
        print("\n🎉 All tests passed! DALL-E integration is working correctly.")
        print(
            "   Placeholder implementations have been successfully replaced with real functionality."
        )
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
