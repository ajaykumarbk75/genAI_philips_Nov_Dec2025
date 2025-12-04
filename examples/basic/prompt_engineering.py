"""
Prompt Engineering Example

This example demonstrates different prompting techniques and best practices.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def basic_prompt_example(client):
    """Example of a basic prompt"""
    print("1. BASIC PROMPT")
    print("-" * 50)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Write a product description"}
        ],
        max_tokens=100
    )
    print(response.choices[0].message.content)
    print()

def detailed_prompt_example(client):
    """Example of a detailed, structured prompt"""
    print("2. DETAILED PROMPT WITH CONTEXT")
    print("-" * 50)
    
    prompt = """
    Write a professional product description for:
    Product: Smart Health Monitor Watch
    Target Audience: Health-conscious professionals aged 30-50
    Key Features: Heart rate monitoring, sleep tracking, stress level detection
    Tone: Professional yet approachable
    Length: 2-3 sentences
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional marketing copywriter."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    print(response.choices[0].message.content)
    print()

def few_shot_prompt_example(client):
    """Example of few-shot prompting"""
    print("3. FEW-SHOT PROMPTING")
    print("-" * 50)
    
    messages = [
        {"role": "system", "content": "You classify customer feedback as positive, negative, or neutral."},
        {"role": "user", "content": "The product exceeded my expectations!"},
        {"role": "assistant", "content": "positive"},
        {"role": "user", "content": "Delivery was very slow."},
        {"role": "assistant", "content": "negative"},
        {"role": "user", "content": "The watch is great but battery life could be better."}
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=50
    )
    print(response.choices[0].message.content)
    print()

def main():
    """Run all prompt examples"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("PROMPT ENGINEERING EXAMPLES")
    print("=" * 50)
    print()
    
    try:
        basic_prompt_example(client)
        detailed_prompt_example(client)
        few_shot_prompt_example(client)
        
        print("\nKey Takeaways:")
        print("- Be specific about what you want")
        print("- Provide context and constraints")
        print("- Use examples to guide the model (few-shot)")
        print("- Set the right system message for tone/role")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure you have set your OPENAI_API_KEY in the .env file")

if __name__ == "__main__":
    main()
