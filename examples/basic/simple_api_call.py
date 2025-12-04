"""
Basic OpenAI API Example

This example demonstrates how to make a simple API call to OpenAI's GPT models.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """
    Simple example of using OpenAI API to generate text
    """
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Define the prompt
    prompt = "Explain what Generative AI is in simple terms."
    
    print(f"Prompt: {prompt}\n")
    print("Response:")
    print("-" * 50)
    
    try:
        # Make the API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        # Extract and print the response
        answer = response.choices[0].message.content
        print(answer)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure you have set your OPENAI_API_KEY in the .env file")

if __name__ == "__main__":
    main()
