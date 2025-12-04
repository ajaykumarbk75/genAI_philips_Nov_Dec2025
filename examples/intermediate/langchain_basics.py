"""
LangChain Basic Example

This example demonstrates using LangChain for building LLM applications.
"""

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def simple_chain_example():
    """Example of a simple LangChain chain"""
    print("SIMPLE LANGCHAIN CHAIN")
    print("-" * 50)
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant specializing in {domain}."),
        ("user", "{question}")
    ])
    
    # Create the chain
    chain = prompt | llm | StrOutputParser()
    
    # Use the chain
    result = chain.invoke({
        "domain": "healthcare technology",
        "question": "What are the benefits of wearable health monitors?"
    })
    
    print(result)
    print()

def multi_step_chain_example():
    """Example of a multi-step chain"""
    print("MULTI-STEP CHAIN")
    print("-" * 50)
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # First chain: Generate a topic
    topic_prompt = ChatPromptTemplate.from_messages([
        ("system", "You suggest interesting topics about {subject}."),
        ("user", "Suggest one specific topic.")
    ])
    
    topic_chain = topic_prompt | llm | StrOutputParser()
    
    # Second chain: Explain the topic
    explanation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert educator."),
        ("user", "Explain this topic in simple terms: {topic}")
    ])
    
    explanation_chain = explanation_prompt | llm | StrOutputParser()
    
    # Execute the chains
    topic = topic_chain.invoke({"subject": "AI in healthcare"})
    print(f"Generated Topic: {topic}\n")
    
    explanation = explanation_chain.invoke({"topic": topic})
    print(f"Explanation: {explanation}")
    print()

def main():
    """Run all LangChain examples"""
    print("LANGCHAIN EXAMPLES")
    print("=" * 50)
    print()
    
    try:
        simple_chain_example()
        multi_step_chain_example()
        
        print("\nKey Concepts:")
        print("- LangChain simplifies LLM application development")
        print("- Chains connect prompts, models, and outputs")
        print("- Templates make prompts reusable and maintainable")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure you have:")
        print("1. Installed all requirements (pip install -r requirements.txt)")
        print("2. Set your OPENAI_API_KEY in the .env file")

if __name__ == "__main__":
    main()
