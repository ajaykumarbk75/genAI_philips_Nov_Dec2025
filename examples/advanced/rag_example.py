"""
Advanced: RAG (Retrieval-Augmented Generation) Example

This example demonstrates a simple RAG pattern where we retrieve relevant
information from documents and use it to augment the LLM's response.
"""

import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Sample healthcare knowledge base
KNOWLEDGE_BASE = """
Wearable Health Monitors:
Wearable health monitors are devices that track various health metrics continuously.
Common features include heart rate monitoring, sleep tracking, step counting, and
stress level detection. These devices can help detect irregular heart rhythms,
monitor fitness progress, and provide early warnings for potential health issues.

Philips Healthcare Technology:
Philips is a leader in health technology, offering solutions ranging from diagnostic
imaging to patient monitoring systems. Their innovations focus on improving patient
outcomes through connected care solutions and data-driven insights.

AI in Medical Diagnostics:
Artificial Intelligence in medical diagnostics can analyze medical images, identify
patterns, and assist healthcare professionals in making more accurate diagnoses.
AI systems can process large amounts of data quickly and identify subtle patterns
that might be missed by human observation alone.

Telemedicine Benefits:
Telemedicine allows patients to consult with healthcare providers remotely using
digital communication tools. Benefits include improved access to care, reduced
travel time and costs, and the ability to monitor chronic conditions more effectively.
"""

class SimpleRAG:
    """Simple RAG implementation for demonstration"""
    
    def __init__(self):
        """Initialize the RAG system"""
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.documents = self._prepare_documents()
    
    def _prepare_documents(self) -> List[Dict[str, str]]:
        """Split knowledge base into chunks"""
        # Simple splitting by paragraphs
        chunks = [chunk.strip() for chunk in KNOWLEDGE_BASE.split('\n\n') if chunk.strip()]
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": i,
                "content": chunk
            })
        return documents
    
    def _retrieve_relevant_docs(self, query: str, top_k: int = 2) -> List[str]:
        """
        Simple retrieval based on keyword matching.
        In production, use vector embeddings for better results.
        """
        query_lower = query.lower()
        scored_docs = []
        
        for doc in self.documents:
            content_lower = doc["content"].lower()
            # Simple scoring: count query words in document
            score = sum(word in content_lower for word in query_lower.split())
            scored_docs.append((score, doc["content"]))
        
        # Sort by score and return top_k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc[1] for doc in scored_docs[:top_k]]
    
    def answer_question(self, question: str) -> str:
        """Answer a question using RAG"""
        # Retrieve relevant documents
        relevant_docs = self._retrieve_relevant_docs(question)
        
        # Prepare context
        context = "\n\n".join(relevant_docs)
        
        # Create prompt with context
        prompt = f"""Based on the following information, please answer the question.
If the information provided doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        # Get response from LLM
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)
        
        return response.content, relevant_docs

def demo_rag():
    """Demonstrate RAG system"""
    print("RETRIEVAL-AUGMENTED GENERATION (RAG) DEMO")
    print("=" * 60)
    print()
    
    rag = SimpleRAG()
    
    questions = [
        "What features do wearable health monitors typically have?",
        "How can AI help with medical diagnostics?",
        "What are the advantages of telemedicine?",
        "Tell me about Philips healthcare technology"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 60)
        
        answer, retrieved_docs = rag.answer_question(question)
        
        print(f"\nAnswer: {answer}")
        print(f"\n📚 Retrieved {len(retrieved_docs)} relevant document(s)")
        
        if i < len(questions):
            print("\n" + "=" * 60)

def main():
    """Run the RAG demonstration"""
    try:
        demo_rag()
        
        print("\n\n" + "=" * 60)
        print("KEY CONCEPTS:")
        print("=" * 60)
        print("""
1. RAG combines retrieval and generation
2. Retrieval finds relevant information from a knowledge base
3. Generation uses that information to create informed responses
4. This reduces hallucinations and provides source-grounded answers
5. Production RAG systems use vector embeddings for better retrieval

IMPROVEMENTS FOR PRODUCTION:
- Use vector databases (Pinecone, Weaviate, Chroma)
- Implement semantic search with embeddings
- Add citation tracking
- Implement re-ranking of retrieved documents
- Add caching for efficiency
- Handle larger document collections
        """)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure you have:")
        print("1. Installed all requirements")
        print("2. Set your OPENAI_API_KEY in the .env file")

if __name__ == "__main__":
    main()
