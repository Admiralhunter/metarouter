#!/bin/bash
# MetaRouter API Examples using curl

BASE_URL="http://localhost:8000"

echo "MetaRouter curl Examples"
echo "================================"

# Example 1: Health check
echo -e "\n1. Health Check"
echo "--------------------------------"
curl -s "${BASE_URL}/health" | jq '.'

# Example 2: List available models
echo -e "\n2. List Models"
echo "--------------------------------"
curl -s "${BASE_URL}/v1/models" | jq '.data[] | {id, owned_by}'

# Example 3: Simple chat completion
echo -e "\n3. Simple Chat"
echo "--------------------------------"
curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Say hello in 5 words"}
    ]
  }' | jq '{model, response: .choices[0].message.content}'

# Example 4: Code generation
echo -e "\n4. Code Generation"
echo "--------------------------------"
curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a Python function to check if a number is prime"}
    ]
  }' | jq '{model, response: .choices[0].message.content}'

# Example 5: Reasoning query
echo -e "\n5. Reasoning Query"
echo "--------------------------------"
curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain the theory of relativity in simple terms"}
    ]
  }' | jq '{model, response: .choices[0].message.content}'

# Example 6: Streaming response
echo -e "\n6. Streaming Response"
echo "--------------------------------"
echo "Streaming response for: 'Count to 5'"
curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Count to 5"}
    ],
    "stream": true
  }' | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
      data="${line#data: }"
      if [[ $data != "[DONE]" ]]; then
        content=$(echo "$data" | jq -r '.choices[0].delta.content // empty')
        if [[ -n $content ]]; then
          echo -n "$content"
        fi
      fi
    fi
  done
echo -e "\n"

# Example 7: Multi-turn conversation
echo -e "\n7. Multi-turn Conversation"
echo "--------------------------------"
curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is Python?"},
      {"role": "assistant", "content": "Python is a high-level programming language."},
      {"role": "user", "content": "Give me a Hello World example"}
    ]
  }' | jq '{model, response: .choices[0].message.content}'

# Example 8: With parameters
echo -e "\n8. Chat with Parameters"
echo "--------------------------------"
curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a creative story about a robot"}
    ],
    "temperature": 0.9,
    "max_tokens": 150
  }' | jq '{model, response: .choices[0].message.content, usage}'

echo -e "\n================================"
echo "Examples completed!"
