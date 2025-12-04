"""
Conversation Memory Example

This example demonstrates how to maintain conversation context using LangChain.
"""

import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

def conversation_with_memory():
    """Example of maintaining conversation context"""
    print("CONVERSATION WITH MEMORY")
    print("-" * 50)
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create memory
    memory = ConversationBufferMemory()
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # First interaction
    print("\n1st Message:")
    response1 = conversation.predict(input="Hi! My name is John and I work at Philips.")
    print(f"AI: {response1}\n")
    
    # Second interaction - AI should remember the name
    print("2nd Message:")
    response2 = conversation.predict(input="What company did I say I work at?")
    print(f"AI: {response2}\n")
    
    # Third interaction - AI should remember both
    print("3rd Message:")
    response3 = conversation.predict(input="What's my name?")
    print(f"AI: {response3}\n")
    
    print("\nConversation History:")
    print("-" * 50)
    print(memory.buffer)

def main():
    """Run conversation memory example"""
    print("CONVERSATION MEMORY EXAMPLE")
    print("=" * 50)
    
    try:
        conversation_with_memory()
        
        print("\n\nKey Concepts:")
        print("- Memory allows the AI to remember previous messages")
        print("- ConversationBufferMemory stores the entire conversation")
        print("- This enables context-aware responses")
        print("- Essential for building chatbots and assistants")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure you have:")
        print("1. Installed all requirements")
        print("2. Set your OPENAI_API_KEY in the .env file")

if __name__ == "__main__":
    main()
