import os
import streamlit as st
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Tuple

# ==============================
# Load Environment Variables
# ==============================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==============================
# Configuration
# ==============================
DATA_DIR = "policies"
CHUNK_SIZE = 300
TOP_K = 3
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# ==============================
# Intelligent Intent Classification
# ==============================
def classify_intent(text: str, history: List[Dict] = None) -> Dict[str, any]:
    """
    Use LLM to intelligently classify user intent and determine
    whether the query requires policy lookup.
    Consider conversation history for context.
    """
    
    # Build history context for classification
    history_text = ""
    if history and len(history) > 0:
        recent = history[-2:]  # Last 2 exchanges
        history_text = "\n".join([f"Previous Q: {h['question']}\nPrevious A: {h['answer'][:100]}..." for h in recent])
        history_text = f"\nRecent conversation:\n{history_text}\n"
    
    classification_prompt = f"""Analyze this user message and classify it, considering any conversation history.
{history_text}
Current user message: "{text}"

IMPORTANT: If the current message is a follow-up question (e.g., "what about digital products?", "how long?", "can I do that?"), 
and it refers to a topic from the conversation history, classify it as POLICY_QUESTION even if it seems vague on its own.

Classify into ONE of these categories:
1. GREETING - Simple greetings (hi, hello, hey)
2. SMALL_TALK - Questions about the assistant itself or general chat
3. EMOTIONAL - User expressing emotions or seeking emotional support
4. POLICY_QUESTION - Actual question about company policies (refunds, cancellations, shipping, returns, etc.)
5. OFF_TOPIC - Questions completely unrelated to company policies

Respond in this EXACT format:
CATEGORY: [category name]
REQUIRES_RAG: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [brief explanation, mention if this is a follow-up question]"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": classification_prompt}],
        temperature=0,
        max_tokens=150
    )
    
    result = response.choices[0].message.content
    
    # Parse response
    lines = result.strip().split('\n')
    intent = {
        'category': 'POLICY_QUESTION',  # default
        'requires_rag': True,
        'confidence': 'MEDIUM',
        'reasoning': ''
    }
    
    for line in lines:
        if line.startswith('CATEGORY:'):
            intent['category'] = line.split(':', 1)[1].strip()
        elif line.startswith('REQUIRES_RAG:'):
            intent['requires_rag'] = 'YES' in line.upper()
        elif line.startswith('CONFIDENCE:'):
            intent['confidence'] = line.split(':', 1)[1].strip()
        elif line.startswith('REASONING:'):
            intent['reasoning'] = line.split(':', 1)[1].strip()
    
    return intent

# ==============================
# Document Utilities
# ==============================
def load_documents() -> List[str]:
    if not os.path.exists(DATA_DIR):
        return []
    documents = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    documents.append(content)
    return documents

def chunk_text(text: str, chunk_size: int) -> List[str]:
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

def embed_texts(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return np.array([e.embedding for e in response.data]).astype("float32")

# ==============================
# Vector Store
# ==============================
@st.cache_resource
def build_vector_store():
    documents = load_documents()
    if not documents:
        return None, []
    
    chunks = []
    for doc in documents:
        chunks.extend(chunk_text(doc, CHUNK_SIZE))
    
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index, chunks

# ==============================
# Enhanced Prompt Engineering
# ==============================
CONTEXT_AWARE_PROMPT = """You are an intelligent company policy assistant with natural conversation abilities.

Context from policy documents:
{context}

Recent conversation history:
{history}

Current question: {question}

Instructions:
1. IMPORTANT: Consider the conversation history to understand context and pronouns (it, that, they, etc.)
2. If the question refers to a previous topic (e.g., "what about...", "how long...", "can I..."), use the conversation history to understand what they're referring to
3. Provide a natural, conversational answer using both the retrieved context AND conversation history
4. Preserve exact company policy terminology (e.g., specific timeframes, conditions, dollar amounts)
5. Paraphrase explanations naturally while keeping policy terms precise
6. Be concise and direct - avoid unnecessary formatting
7. If the question is a follow-up, acknowledge the connection naturally (e.g., "For refunds, you have..." when they previously asked about returns)
8. If information is missing, say so clearly and suggest what the user might do

Respond naturally as if having a professional conversation with someone who may be asking follow-up questions."""

SIMPLE_RESPONSE_PROMPT = """You are a helpful, friendly company policy assistant.

Question: {question}

Provide a brief, natural response appropriate to the question type.
Be warm and professional. Keep it conversational."""

# ==============================
# RAG Pipeline with Context Awareness
# ==============================
def retrieve_context(question: str, index, chunks: List[str], history: List[Dict] = None) -> Tuple[str, float]:
    """
    Retrieve context and return relevance score.
    Enhance query with conversation history for better retrieval.
    """
    
    # Enhance query with recent context for better retrieval
    enhanced_query = question
    if history and len(history) > 0:
        # Get the last question to understand context
        last_exchange = history[-1]
        # Create an enhanced query that includes context
        enhanced_query = f"{last_exchange['question']} {question}"
    
    q_embedding = embed_texts([enhanced_query])
    distances, indices = index.search(q_embedding, TOP_K)
    
    # Calculate relevance score (lower distance = higher relevance)
    avg_distance = np.mean(distances[0])
    relevance_score = 1 / (1 + avg_distance)  # Normalize to 0-1
    
    context = "\n\n".join(chunks[i] for i in indices[0])
    return context, relevance_score

def generate_response(question: str, intent: Dict, index, chunks: List[str], history: List[Dict] = None) -> Tuple[str, bool]:
    """
    Generate response based on intent classification.
    Returns: (response_text, show_sources)
    Uses conversation history for context continuity.
    """
    
    # Build formatted history for prompts
    history_text = ""
    if history and len(history) > 0:
        history_entries = []
        for i, exchange in enumerate(history[-3:]):  # Last 3 exchanges
            history_entries.append(f"Q{i+1}: {exchange['question']}")
            history_entries.append(f"A{i+1}: {exchange['answer']}")
        history_text = "\n".join(history_entries)
    
    # Handle non-policy questions without RAG
    if intent['category'] == 'GREETING':
        # Check if this is a greeting after previous conversation
        if history and len(history) > 0:
            return "Hello again! What else would you like to know about our policies?", False
        return "Hello! I'm here to help you with questions about our company policies, including refunds, cancellations, and shipping. What would you like to know?", False
    
    elif intent['category'] == 'SMALL_TALK':
        prompt = SIMPLE_RESPONSE_PROMPT.format(question=question)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content, False
    
    elif intent['category'] == 'EMOTIONAL':
        return "I appreciate you sharing that with me. While I'm here primarily to help with policy questions, I'm glad to assist you when you're ready. Is there anything related to our policies I can help clarify?", False
    
    elif intent['category'] == 'OFF_TOPIC':
        return "I'm specifically designed to help with company policy questions about refunds, cancellations, and shipping. I'm not able to assist with other topics. Is there a policy question I can help you with?", False
    
    # Handle policy questions with RAG
    elif intent['category'] == 'POLICY_QUESTION':
        if index is None:
            return "I apologize, but I don't have access to policy documents at the moment. Please contact customer support for assistance.", False
        
        context, relevance = retrieve_context(question, index, chunks, history)
        
        # If relevance is very low, the answer might not be in the docs
        if relevance < 0.3:
            return "I couldn't find specific information about that in our policy documents. For the most accurate answer, I recommend contacting our customer support team directly.", False
        
        prompt = CONTEXT_AWARE_PROMPT.format(
            context=context,
            history=history_text,
            question=question
        )
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        # Determine if we should show sources (only for complex/detailed answers)
        show_sources = len(answer) > 200 or any(keyword in answer.lower() for keyword in 
                                                 ['refund', 'days', 'policy', 'condition', 'cancellation', 'shipping'])
        
        return answer, show_sources
    
    # Fallback
    return "I'm not sure how to help with that. Could you ask a question about our refund, cancellation, or shipping policies?", False

# ==============================
# Session State Management
# ==============================
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'last_question' not in st.session_state:
    st.session_state.last_question = None

def add_to_history(question: str, answer: str):
    # Only add if this is a new question (not a rerun from UI interactions)
    if st.session_state.last_question != question:
        st.session_state.conversation_history.append({
            'question': question,
            'answer': answer
        })
        st.session_state.last_question = question
        
        # Keep only last 10 exchanges for better context
        if len(st.session_state.conversation_history) > 10:
            st.session_state.conversation_history.pop(0)

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(
    page_title="Policy Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .response-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🤖 Policy Question Answering Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask me anything about refunds, cancellations, or shipping policies</div>', unsafe_allow_html=True)

# Build vector store
index, chunks = build_vector_store()

# Input
question = st.text_input("💬 Ask your question", placeholder="e.g., What is your refund policy?")

if question:
    with st.spinner("🤔 Thinking..."):
        # Get conversation history
        history = st.session_state.conversation_history
        
        # Classify intent with history context
        intent = classify_intent(question, history)
        
        # Generate response with history
        answer, show_sources = generate_response(question, intent, index, chunks, history)
        
        # Add to history
        add_to_history(question, answer)
        
        # Display answer in a nice format
        st.markdown('<div class="response-box">', unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show sources only when relevant (for policy questions with substantive answers)
        if show_sources and intent['category'] == 'POLICY_QUESTION' and index is not None:
            with st.expander("📄 View source context from policy documents"):
                context, _ = retrieve_context(question, index, chunks, history)
                st.text(context)
        
        # Debug info (optional, can be removed in production)
        if st.checkbox("🔍 Show debug info", value=False):
            st.json({
                'intent_category': intent['category'],
                'requires_rag': intent['requires_rag'],
                'confidence': intent['confidence'],
                'reasoning': intent['reasoning'],
                'show_sources': show_sources,
                'conversation_length': len(st.session_state.conversation_history)
            })

# Sidebar with conversation history
with st.sidebar:
    st.header("💬 Conversation History")
    if st.session_state.conversation_history:
        for i, exchange in enumerate(reversed(st.session_state.conversation_history)):
            with st.expander(f"Q: {exchange['question'][:50]}...", expanded=False):
                st.write("**Question:**", exchange['question'])
                st.write("**Answer:**", exchange['answer'][:200] + "..." if len(exchange['answer']) > 200 else exchange['answer'])
    else:
        st.write("No conversation history yet.")
    
    if st.button("🗑️ Clear History"):
        st.session_state.conversation_history = []
        st.session_state.last_question = None
        st.rerun()