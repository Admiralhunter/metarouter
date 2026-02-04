#!/usr/bin/env python
"""Simple test script for the router."""

import asyncio
import httpx


async def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
        print(f"Health: {response.json()}")
        return response.status_code == 200


async def test_models():
    """Test models listing."""
    print("\nTesting models endpoint...")
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/v1/models")
        data = response.json()
        print(f"Found {len(data['data'])} models:")
        for model in data["data"][:5]:  # Show first 5
            print(f"  - {model['id']}")
        return response.status_code == 200


async def test_completion():
    """Test chat completion."""
    print("\nTesting chat completion...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Say hello in exactly 5 words"}
                ],
                "stream": False,
            },
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Selected model: {data['model']}")
            print(f"Response: {data['choices'][0]['message']['content']}")
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("MetaRouter - Quick Test")
    print("=" * 60)
    print("\nMake sure:")
    print("1. LM Studio is running on http://localhost:1234")
    print("2. phi-4 is loaded in LM Studio")
    print("3. Router is running on http://localhost:8000")
    print("=" * 60)

    try:
        results = []
        results.append(("Health Check", await test_health()))
        results.append(("Models List", await test_models()))
        results.append(("Chat Completion", await test_completion()))

        print("\n" + "=" * 60)
        print("Test Results:")
        print("=" * 60)
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} - {name}")

        all_passed = all(result[1] for result in results)
        if all_passed:
            print("\n🎉 All tests passed!")
        else:
            print("\n⚠️  Some tests failed. Check the router logs.")

    except httpx.ConnectError:
        print("\n❌ Could not connect to router on http://localhost:8000")
        print("Make sure the router is running!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
