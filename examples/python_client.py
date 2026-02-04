"""Example: Using MetaRouter with Python OpenAI client."""

from openai import OpenAI

# Point to MetaRouter instead of LM Studio directly
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required for local routing
)

# Example 1: Simple chat
print("Example 1: Simple Greeting")
print("=" * 60)
response = client.chat.completions.create(
    model="gpt-4",  # Model parameter is ignored - router selects best model
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ]
)
print(f"Response: {response.choices[0].message.content}")
print(f"Model used: {response.model}\n")

# Example 2: Code generation
print("Example 2: Code Generation")
print("=" * 60)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
    ]
)
print(f"Response: {response.choices[0].message.content}")
print(f"Model used: {response.model}\n")

# Example 3: Complex reasoning
print("Example 3: Complex Reasoning")
print("=" * 60)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Explain the concept of quantum entanglement in simple terms"}
    ]
)
print(f"Response: {response.choices[0].message.content}")
print(f"Model used: {response.model}\n")

# Example 4: Streaming
print("Example 4: Streaming Response")
print("=" * 60)
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Count to 10"}
    ],
    stream=True
)

print("Streaming: ", end="", flush=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

# Example 5: Multi-turn conversation
print("Example 5: Multi-turn Conversation")
print("=" * 60)
messages = [
    {"role": "user", "content": "What is Python?"},
]

response1 = client.chat.completions.create(model="gpt-4", messages=messages)
print(f"User: What is Python?")
print(f"Assistant: {response1.choices[0].message.content}")
print(f"Model used: {response1.model}\n")

# Continue conversation
messages.append({"role": "assistant", "content": response1.choices[0].message.content})
messages.append({"role": "user", "content": "Give me a simple code example"})

response2 = client.chat.completions.create(model="gpt-4", messages=messages)
print(f"User: Give me a simple code example")
print(f"Assistant: {response2.choices[0].message.content}")
print(f"Model used: {response2.model}\n")
