"""Example: Streaming responses with MetaRouter."""

import httpx
import json


async def stream_chat_completion(message: str):
    """Stream a chat completion from MetaRouter."""
    url = "http://localhost:8000/v1/chat/completions"

    payload = {
        "messages": [{"role": "user", "content": message}],
        "stream": True
    }

    print(f"Query: {message}")
    print("Response: ", end="", flush=True)

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        if data["choices"][0]["delta"].get("content"):
                            content = data["choices"][0]["delta"]["content"]
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue

    print("\n" + "=" * 60 + "\n")


async def main():
    """Run streaming examples."""
    print("MetaRouter Streaming Examples")
    print("=" * 60 + "\n")

    # Example 1: Simple query
    await stream_chat_completion("Tell me a short joke")

    # Example 2: Code generation
    await stream_chat_completion("Write a Python function to reverse a string")

    # Example 3: Explanation
    await stream_chat_completion("Explain how HTTP works in 3 sentences")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
