# Import utilities
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from datetime import datetime


from typing import Literal
import os
import sys

# Ensure backend directory is in path for imports if needed, though relative imports should work if run as module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils_openai import (
    setup_openai_api,
    create_embeddings,
    create_llm,
    create_vectorstore,
    system_prompt_def,
)

# Load API key
api_key = setup_openai_api()
print("✅ API key loaded successfully!")

# Create chat model
llm = create_llm(api_key, temperature=0)

# Create embeddings model
embeddings = create_embeddings(api_key)

# Connect to existing vector store

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

from utils_openai import load_existing_vectorstore

vectorstore = load_existing_vectorstore(
    embeddings=embeddings,
    persist_directory=CHROMA_PATH,
    collection_name="tax_collection"
)
print("Vector store loaded!")

@tool
def retrieve_documents(query: str) -> str:
    """
    Search the Nigerian 2025 tax reform knowledge base for accurate, authoritative information.

Use this tool ONLY when:
- Exact figures, thresholds, rates, dates, or exemptions are required
- Legal authority or section references are needed
- The question compares old vs new tax rules
- You are less than fully certain of the correct tax rule

Do NOT use this tool for:
- General explanations or summaries
- High-level “what does this mean for me?” questions
- Opinions, greetings, small talk, or simple calculations

Never guess or invent tax rules, figures, or section numbers.
If accuracy matters and you are unsure, retrieve first.

Query guidance:
- Be specific and focused
- Include relevant terms like “Nigeria Tax Act 2025”, “Nigeria Tax Administration Act”, or the tax type involved
- Search for clarity, not completeness

Return only the most relevant excerpts needed to answer accurately.
"""
    # Use MMR (Maximum Marginal Relevance) for diverse results
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )
    
    # Retrieve documents
    results = retriever.invoke(query)
    
    if not results:
        return "No relevant documents found."
    
    # Format results
    formatted = "\n\n---\n\n".join(
        f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
        for i, doc in enumerate(results)
    )
    
    return formatted

print(" Retrieval tool created")



@tool
def get_current_date() -> str:
    """
    PURPOSE:
    --------
    Returns the current date in ISO format (YYYY-MM-DD) so the agent can reason
    correctly about time-sensitive statements.

    WHY THIS TOOL EXISTS:
    ---------------------
    This tool prevents the agent from speaking as if key dates (e.g. January 1, 2026)
    are always in the future. Nigerian tax reform explanations depend heavily on
    whether a rule:
      - is upcoming,
      - has started applying, or
      - already applies to income earned.

    Without this tool, the agent may default to static phrasing like:
      "This will start in January 2026"
    even when January 2026 has already passed.

    WHEN THE AGENT MUST USE THIS TOOL:
    ---------------------------------
    The agent MUST call this tool BEFORE mentioning or implying anything related to:
      - Current date
      - Current year
      - "Now", "currently", "as at today"
      - Whether January 1, 2026 is in the future or past
      - Whether new tax rules are already applying or will apply later
      - Statements like:
          • "from next year"
          • "has started"
          • "will begin"
          • "is now in effect"
          • "has not started yet"

    WHAT THE TOOL RETURNS:
    ---------------------
    A string representing today's date in ISO format:
        YYYY-MM-DD

    HOW THE AGENT SHOULD USE THE OUTPUT:
    -----------------------------------
    - Compare the returned date with key reform dates (e.g. 2026-01-01).
    - Adjust language accordingly:
        • If today < 2026-01-01 → say "will apply from January 1, 2026"
        • If today >= 2026-01-01 → say "now applies to income earned from January 1, 2026"
    - NEVER expose the raw date value to the user unless explicitly asked.
    - NEVER mention that a tool was used.

    IMPORTANT CONSTRAINTS:
    ----------------------
    - This tool provides date awareness ONLY.
    - It does NOT provide legal interpretation.
    - It does NOT validate tax rules.
    - It must be silently used in the background.

    FAILURE TO USE THIS TOOL:
    -------------------------
    If the agent discusses timing without using this tool, it risks:
      - Giving outdated guidance
      - Incorrectly framing user obligations
      - Losing user trust

    RETURN VALUE:
    -------------
    str: Current date in YYYY-MM-DD format.
    """
    return datetime.now().strftime("%Y-%m-%d")

print(" Current date tool created")



#System Prompt
system_prompt = system_prompt_def()
print(" System prompt configured")

#Bind tools to LLM
tools = [retrieve_documents, get_current_date] 
llm_with_tools = llm.bind_tools(tools)

def assistant(state: MessagesState) -> dict:
    """
Assistant node that decides whether to answer directly or retrieve
authoritative information from the Nigerian 2025+ tax knowledge base.

Retrieve whenever accuracy is critical (figures, thresholds, dates,
section references, or old vs new rule comparisons), especially for
post-cutoff tax law.

Never guess or rely on internal knowledge when unsure.
"""

    messages = [system_prompt] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """
Decide whether to call tools or finish the response.

Route to tools whenever accuracy is critical or uncertainty exists
(e.g. tax figures, dates, section references, or post-cutoff rule changes).
Only finish when the response can be confidently given without retrieval.
"""

    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'tools'
    return '__end__'

print(" Agent nodes defined")

#Build graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node('assistant', assistant)
builder.add_node('tools', ToolNode(tools))

# Define edges
builder.add_edge(START, 'assistant')
builder.add_conditional_edges(
    'assistant',
    should_continue,
    {'tools': 'tools', '__end__': END},
)
builder.add_edge('tools', 'assistant')

# Add memory
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
agent = graph

print(" Agentic RAG system compiled")