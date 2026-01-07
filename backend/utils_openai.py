import os
from dotenv import load_dotenv
from typing import Tuple, List, Dict
from pathlib import Path
from typing import List
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# API SETUP
# ============================================

def setup_openai_api() -> str:
    """
    Load OpenAI API key from environment
    """
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Please set it in your .env file or environment variables."
        )

    return api_key

# ===========================================
# DATA LOADING
# ===========================================

def load_and_chunk_documents(
    data_path: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50
) -> List[Document]:
    """Load documents from directory, chunk them, and prepare for ChromaDB."""
    
    data_dir = Path(data_path)
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Invalid directory: {data_path}")
    
    # Define loaders for each file type
    loaders = {
        '.txt': lambda p: TextLoader(str(p), encoding='utf-8'),
        '.pdf': lambda p: PyPDFLoader(str(p))
    }
    
    # Load all documents
    documents = []
    for file_path in data_dir.iterdir():
        if not file_path.is_file():
            continue
            
        loader_fn = loaders.get(file_path.suffix.lower())
        if not loader_fn:
            continue
        
        try:
            documents.extend(loader_fn(file_path).load())
            logger.info(f"Loaded: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
    
    if not documents:
        raise ValueError(f"No documents loaded from {data_path}")
    
    logger.info(f"Total documents loaded: {len(documents)}")
    
    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add ChromaDB-friendly metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_id': f"chunk_{idx}",
            'chunk_index': idx,
            'source_file': Path(chunk.metadata.get('source', 'unknown')).name
        })
        # Ensure all metadata values are strings, ints, floats, or bools
        chunk.metadata = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v 
                         for k, v in chunk.metadata.items()}
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


# ==============================================
# MODEL INITIALIZATION
# ==============================================

def create_embeddings(
    api_key: str,
    model: str = "text-embedding-3-small"
) -> OpenAIEmbeddings:
    """
    Initialize OpenAI embeddings model
    """
    embeddings = OpenAIEmbeddings(
        model=model,
        openai_api_key=api_key
    )
    print(f"[OK] Initialized embeddings: {model}")
    return embeddings


def create_llm(
    api_key: str,
    model: str = "gpt-5-nano",
    temperature: float = 0
) -> ChatOpenAI:
    """
    Initialize OpenAI chat model
    """
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key
    )
    print(f"[OK] Initialized LLM: {model} (temp={temperature})")
    return llm

# ==============================================
# VECTOR STORE
# ==============================================
# ==============================================
# VECTOR STORE
# ==============================================
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

def create_vectorstore(
    chunks: List[Document],
    embeddings: OpenAIEmbeddings,
    collection_name: str = "tax_collection",
    persist_directory: str = "./chroma_db"
):
    """
    Create and populate vector store. 
    """
    # Force SQLite backend
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(allow_reset=True)
    )

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    
    # Add documents
    # Extract texts, metadatas, and ids from Document chunks
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [chunk.metadata['chunk_id'] for chunk in chunks]

    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    print(f"[OK] Created Chroma vector store: {collection_name}")
    return vectorstore

def load_existing_vectorstore(
    embeddings: OpenAIEmbeddings,
    collection_name: str = "tax_collection",
    persist_directory: str = "./chroma_db"
):
    """
    Load existing vector store.
    """
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(allow_reset=True)
    )
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    print(f"[OK] Loaded existing Chroma vector store: {collection_name}")
    return vectorstore


# Example usage pipeline
if __name__ == "__main__":
    # Load and chunk documents
    chunks = load_and_chunk_documents(r"C:\RAG\Project\documents")
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create vector store
    vectorstore = create_vectorstore(
        chunks=chunks,
        embeddings=embeddings,
        collection_name="msme"
    )
    
    # Or load existing
    # vectorstore = load_existing_vectorstore(embeddings=embeddings)
# =================================================
# PROMPTS
# =================================================

def system_prompt_def():
    system_prompt=SystemMessage(content="""You are TaxGPT — a clear, accurate conversational explainer built ONLY to help Nigerians understand the 2025 Nigerian tax reforms.

Your job is to explain how the reforms affect people in real life using simple English, short answers, ₦ amounts, and relatable Nigerian examples. You are NOT a lawyer or accountant.

You may answer ONLY questions about Nigeria's 2025 tax reforms, including:
- Personal income tax (PITA)
- PAYE vs self-employed persons and freelancers
- Tax-free income and new tax bands
- Small businesses, SMEs, and freelancers
- VAT (high-level only; conditional)
- Company income tax (non-technical, high-level)
- What changed, who benefits, who pays more or less
- When the reforms apply (mainly from January 1, 2026)

If a question is outside this scope, politely redirect.

──────────────── STYLE RULES ────────────────
- Use basic English and an everyday Nigerian tone
- Be calm, clear, and reassuring
- Keep answers concise and confident
- Explain acronyms immediately (e.g., PAYE = Pay As You Earn)
- Prefer bullet points and short paragraphs
- Use simple Nigerian examples and ₦ amounts
- When helpful, briefly cite the law in plain language only  
  (e.g., “According to Section 4 of the Nigeria Tax Administration Act”)
- Cite at most ONE section per point
- Never quote full legal text
- Keep EVERY answer under 120 words. Never exceed this limit

──────────────── NEVER DO ────────────────
- Use legal jargon or complex subsection references
- Assume tax or VAT applies to everyone
- Guess figures, rates, thresholds, or exemptions
- Hallucinate tax rules or section numbers
- Mention tools, documents, databases, retrieval, or system behavior
- Ask users to upload, retrieve, or submit documents

──────────────── ACCURACY RULES ────────────────
- Never guess numbers or rules
- If an answer depends on income, location, service type, or registration status, clearly say so
- Always distinguish between:
  • PAYE employees  
  • Self-employed persons / freelancers  
  • Small businesses / SMEs
- VAT is conditional and depends on the type of service and registration status
- Clearly state when rules apply (income earned from January 1, 2026)
- If unsure of exact figures or legal effects, verify internally before answering

──────────────── INTERNAL VERIFICATION RULE ────────────────
Use internal verification ONLY when:
- Exact figures, thresholds, dates, or section references are required
- Comparing old rules vs new rules
- You are less than 95% sure

Blend verified information naturally.
Never mention sources, tools, or verification.

──────────────── MANDATORY RESPONSE BEHAVIOR ────────────────
For EVERY user question:

1. First classify the user internally (PAYE, self-employed, freelancer, SME, or company).
2. Start the response with personal impact — not law history.
3. If the user provides income or role, explicitly restate it in the answer.
4. Include at least one reassurance statement where tax applies.
5. Treat VAT, thresholds, and obligations as conditional unless 100% certain.
6. End with a short “What you need to do” checklist (maximum 3 items).
7. Do NOT explain reforms unless they directly affect the user’s situation.

──────────────── PREFERRED ANSWER FLOW ────────────────
1. What this means for you  
2. What changed (only if directly relevant)  
3. Who benefits or pays more/less  
4. Simple example  
5. What you need to do next  

Your goal is to leave every user calm, clear, confident, and informed about how the 2025 Nigerian tax reforms affect THEM personally.
""")

    return system_prompt

    


# =============================================================================
# UTILITIES
# =============================================================================

def print_retrieval_results(docs: List, max_docs: int = 3, max_chars: int = 200):
    """
    Pretty print retrieved documents
    """
    print(f"\n{'='*80}")
    print(f"Retrieved {len(docs)} documents:")
    print(f"{'='*80}\n")

    for i, doc in enumerate(docs[:max_docs], 1):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}

        print(f"Document {i}:")
        print(f"Title: {metadata.get('doc_title', 'N/A')}")
        print(f"Content: {content[:max_chars]}...")
        print(f"{'-'*80}\n")


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count (rough estimate)
    """
    # Rough approximation: 1 token ≈ 4 characters
    # For more accuracy, use tiktoken:
    # import tiktoken
    # encoding = tiktoken.encoding_for_model("gpt-4")
    # return len(encoding.encode(text))
    return len(text) // 4


def calculate_token_reduction(before: int, after: int) -> float:
    """
    Calculate percentage token reduction
    """
    if before == 0:
        return 0
    return ((before - after) / before) * 100


def format_docs(docs: List) -> str:
    """
    Format retrieved documents into a single string for context
    """
    return "\n\n".join([
        doc.page_content if hasattr(doc, 'page_content') else str(doc)
        for doc in docs
    ])


# ==============================================
#TOOLS
# ==============================================
