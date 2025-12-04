# Best Practices for GenAI Development

This guide covers best practices for building production-ready GenAI applications.

## Security Best Practices

### 1. API Key Management

**❌ Never do this:**
```python
# Hardcoding API keys
client = OpenAI(api_key="sk-abc123...")
```

**✅ Always do this:**
```python
# Use environment variables
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

### 2. Input Validation

Always validate and sanitize user inputs:

```python
def validate_input(user_input: str, max_length: int = 1000) -> str:
    """Validate and sanitize user input"""
    if not user_input or not isinstance(user_input, str):
        raise ValueError("Input must be a non-empty string")
    
    if len(user_input) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length}")
    
    # Remove potentially harmful content
    sanitized = user_input.strip()
    return sanitized
```

### 3. Rate Limiting

Implement rate limiting to prevent abuse:

```python
import time
from functools import wraps

def rate_limit(calls_per_minute: int):
    """Decorator to rate limit function calls"""
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator
```

## Cost Optimization

### 1. Choose the Right Model

- **GPT-4**: Most capable, but expensive. Use for complex tasks.
- **GPT-3.5-turbo**: Fast and cost-effective. Good for most tasks.
- **GPT-3.5-turbo-16k**: For longer context needs.

### 2. Control Token Usage

```python
# Set appropriate max_tokens
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=150,  # Limit response length
    temperature=0.7
)
```

### 3. Cache Responses

Cache frequently requested information:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_ai_response(prompt: str) -> str:
    """Cache AI responses for identical prompts"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

## Prompt Engineering Best Practices

### 1. Be Specific and Clear

**❌ Vague prompt:**
```python
"Tell me about health"
```

**✅ Specific prompt:**
```python
"Explain the top 3 benefits of regular cardiovascular exercise 
for adults aged 40-60, in 150 words or less."
```

### 2. Use System Messages

```python
messages = [
    {
        "role": "system",
        "content": "You are a medical AI assistant. Provide accurate, "
                   "evidence-based information. Always include disclaimers "
                   "about consulting healthcare professionals."
    },
    {
        "role": "user",
        "content": user_question
    }
]
```

### 3. Provide Structure

```python
prompt = """
Task: Summarize the following medical report
Format: Bullet points
Length: Maximum 5 points
Focus: Key findings and recommendations

Report:
{report_text}
"""
```

### 4. Use Few-Shot Learning

Provide examples of desired input/output:

```python
messages = [
    {"role": "system", "content": "Classify medical symptoms by urgency level."},
    {"role": "user", "content": "Mild headache"},
    {"role": "assistant", "content": "Low urgency"},
    {"role": "user", "content": "Chest pain with shortness of breath"},
    {"role": "assistant", "content": "High urgency - seek immediate care"},
    {"role": "user", "content": user_symptom}
]
```

## Error Handling

### 1. Robust Error Handling

```python
import time
from openai import OpenAI, OpenAIError, RateLimitError, APIError

def make_api_call_with_retry(prompt: str, max_retries: int = 3):
    """Make API call with exponential backoff retry"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
            
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 1  # Exponential backoff
                print(f"Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
                
        except APIError as e:
            print(f"API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise
                
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
```

### 2. Validate Responses

```python
def validate_ai_response(response: str) -> bool:
    """Validate AI response meets requirements"""
    if not response:
        return False
    
    if len(response) < 10:  # Too short
        return False
    
    # Add domain-specific validation
    if "I cannot" in response or "I don't know" in response:
        return False
    
    return True
```

## Code Organization

### 1. Separate Configuration

```python
# config.py
from dataclasses import dataclass

@dataclass
class LLMConfig:
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500
    
@dataclass
class AppConfig:
    llm: LLMConfig = LLMConfig()
    debug: bool = False
```

### 2. Use Type Hints

```python
from typing import List, Dict, Optional

def generate_response(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7
) -> Optional[str]:
    """Generate AI response with type hints"""
    pass
```

### 3. Create Reusable Components

```python
class AIAssistant:
    """Reusable AI assistant class"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def ask(self, question: str, system_message: str = None) -> str:
        """Ask a question and get a response"""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": question})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content
```

## Testing

### 1. Unit Tests

```python
import pytest
from unittest.mock import Mock, patch

def test_ai_assistant_response():
    """Test AI assistant returns valid response"""
    with patch('openai.OpenAI') as mock_openai:
        # Mock the response
        mock_response = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        assistant = AIAssistant()
        result = assistant.ask("Test question")
        
        assert result == "Test response"
```

### 2. Integration Tests

Test with real API calls (use test environment):

```python
def test_api_integration():
    """Integration test with real API"""
    if os.getenv("SKIP_INTEGRATION_TESTS"):
        pytest.skip("Integration tests disabled")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say 'test'"}],
        max_tokens=10
    )
    
    assert response.choices[0].message.content
```

## Monitoring and Logging

### 1. Log API Calls

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_api_call(prompt: str, response: str, tokens_used: int):
    """Log API usage for monitoring"""
    logger.info(f"API Call - Tokens: {tokens_used}")
    logger.debug(f"Prompt: {prompt[:100]}...")
    logger.debug(f"Response: {response[:100]}...")
```

### 2. Track Costs

```python
class CostTracker:
    """Track API usage costs"""
    
    # Pricing as of example date (check current pricing)
    PRICING = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
        "gpt-4": {"input": 0.03, "output": 0.06}
    }
    
    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens = 0
    
    def add_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Add usage and calculate cost"""
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] + 
                output_tokens * pricing["output"]) / 1000
        self.total_cost += cost
        self.total_tokens += input_tokens + output_tokens
    
    def get_report(self) -> dict:
        """Get usage report"""
        return {
            "total_cost": f"${self.total_cost:.4f}",
            "total_tokens": self.total_tokens
        }
```

## Documentation

### 1. Document Your Functions

```python
def process_medical_query(query: str, patient_context: dict) -> str:
    """
    Process a medical query with patient context.
    
    Args:
        query: The patient's question or concern
        patient_context: Dictionary containing patient information
            - age: Patient age
            - conditions: List of existing conditions
            - medications: List of current medications
    
    Returns:
        AI-generated response with medical information
        
    Raises:
        ValueError: If query is empty or invalid
        APIError: If the API call fails
        
    Example:
        >>> context = {"age": 45, "conditions": ["diabetes"]}
        >>> response = process_medical_query("What foods should I avoid?", context)
    """
    pass
```

## Compliance and Ethics

### 1. Healthcare Applications

- Always include medical disclaimers
- Don't replace professional medical advice
- Comply with HIPAA (or relevant regulations)
- Ensure data privacy and security

### 2. Bias and Fairness

- Test for bias in responses
- Use diverse examples in prompts
- Validate outputs for fairness
- Document limitations

### 3. Transparency

- Be clear about AI-generated content
- Explain capabilities and limitations
- Provide human oversight for critical decisions

## Performance Optimization

### 1. Async Processing

```python
import asyncio
from openai import AsyncOpenAI

async def process_multiple_queries(queries: List[str]) -> List[str]:
    """Process multiple queries concurrently"""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def process_one(query: str) -> str:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content
    
    results = await asyncio.gather(*[process_one(q) for q in queries])
    return results
```

### 2. Streaming Responses

```python
def stream_response(prompt: str):
    """Stream responses for better UX"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
```

## Checklist for Production

Before deploying to production:

- [ ] All API keys are in environment variables
- [ ] Input validation is implemented
- [ ] Error handling covers all cases
- [ ] Rate limiting is in place
- [ ] Costs are monitored and capped
- [ ] Logging is configured
- [ ] Tests are passing (unit and integration)
- [ ] Documentation is complete
- [ ] Security review completed
- [ ] Compliance requirements met
- [ ] Performance is acceptable
- [ ] Monitoring and alerts configured

## Additional Resources

- [OpenAI Best Practices](https://platform.openai.com/docs/guides/production-best-practices)
- [LangChain Production Guide](https://python.langchain.com/docs/guides/productionization/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

Remember: Building production GenAI applications requires careful attention to security, costs, reliability, and ethics. Always prioritize user safety and data privacy.
