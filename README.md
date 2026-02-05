# Neura-Dynamics-RAG-Project
🤖 Policy Question Answering Assistant
An intelligent RAG-based chatbot that answers company policy questions with natural conversation abilities and full context awareness.
✨ Features
🧠 Intelligent Conversation

LLM-Based Intent Classification: Automatically detects whether a question needs policy lookup or can be answered directly
Full Conversation Continuity: Remembers up to 10 conversation exchanges
Pronoun Resolution: Understands "it", "that", "they" from context
Follow-up Question Handling: Maintains topic flow across multiple questions

📚 Smart RAG System

Semantic Search: Uses OpenAI embeddings with FAISS vector store
Context-Enhanced Retrieval: Combines current and previous questions for better results
Relevance Scoring: Detects when information isn't available in documents
No Hallucination: Only provides answers from actual policy documents

💬 User-Friendly Interface

Clean UI: Professional design with custom styling
Conditional Source Display: Only shows sources when relevant
Conversation History: Sidebar with full chat history
Debug Mode: For developers to inspect intent classification

🎯 Multi-Intent Support
Handles various types of user inputs:

Greetings (hi, hello)
Small talk (what can you do?)
Policy questions (refunds, cancellations, shipping)
Emotional support (basic empathy)
Off-topic detection


🎬 Demo
User: "What's your refund policy?"
Bot: "We offer full refunds within 30 days of purchase for most products..."

User: "What about digital products?"
Bot: "For digital products, the refund window is 14 days..."

User: "How do I request it?"
Bot: "To request a refund, contact support@company.com with your order number..."
🛠️ Tech Stack

Frontend: Streamlit
LLM: OpenAI GPT-4o-mini
Embeddings: OpenAI text-embedding-3-small
Vector Store: FAISS
Language: Python 3.8+

📦 Installation
Prerequisites

Python 3.8 or higher
OpenAI API key


Using the Assistant

Ask a question in the text input field
View the answer in the response box
Check sources (expands automatically for detailed policy answers)
Review history in the sidebar
Enable debug mode to see intent classification details

Example Queries

"What is your refund policy?"
"How long do I have to return an item?"
"Can I cancel my order?"
"What about shipping for international orders?"
"How do I request a refund?"
