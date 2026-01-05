# Import utilities
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from typing import Literal
import os

from utils_openai import (
    setup_openai_api,
    create_embeddings,
    create_llm,
    load_and_chunk_documents,
    create_vectorstore,
    system_prompt_def,
)


# Load API key
api_key = setup_openai_api()
print("✅ API key loaded successfully!")

#WORKFLOW
"""
Load API Key
initialize llm
load documents
chunk documents (plan to use recursive splitter then semantic splitter)
embed
persist to chromadb
define tools (retriever(plan to use mmr), tax tool, calculator)
define system prompt
bind tools to llm
define assitant and conditional nodes
build stategraph
define query agent
test agent
"""

# Create chat model
llm = create_llm(api_key, temperature=0)

#Load Documents and chunk
#I will define a document loader and chunking function later
chunks = load_and_chunk_documents(
    data_path=r"C:\Users\owner\Desktop\TAX\TAX_Documents",
    chunk_size=1000,
    chunk_overlap=10)

# Create embeddings model
embeddings = create_embeddings(api_key)
print("\nModels initialized!")


# Create vector store
vectorstore = create_vectorstore(
        chunks=chunks,
        embeddings=embeddings,
        collection_name="tax_collection")
print("Vector store created and persisted!")

#Do you think we should define the tools in utils_openai.py? or just define it here

#Tools
@tool
def retrieve_documents(query: str) -> str:
    """
    Search for relevant documents in the knowledge base.
    
    Use this tool when you need information from the document collection
    to answer the user's question. Do NOT use this for:
    - General knowledge questions
    - Greetings or small talk
    - Simple calculations
    
    Args:
        query: The search query describing what information is needed
        
    Returns:
        Relevant document excerpts that can help answer the question
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
        f"Document {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    )
    
    return formatted

print(" Retrieval tool created")


#System Prompt
system_prompt = system_prompt_def()
print(" System prompt configured")

#Bind tools to LLM
tools = [retrieve_documents] # I will add the tax calculator tool once we decided on how it works
llm_with_tools = llm.bind_tools(tools)

def assistant(state: MessagesState) -> dict:
    """Assistant node - decides whether to retrieve or answer directly."""
    messages = [system_prompt] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """Decide whether to call tools or finish."""
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
agent = builder.compile(checkpointer=memory)

print(" Agentic RAG system compiled")


# Define Query Agent
def query_agent( thread_id: str = "default"):
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print('Goodbye...')
            break
        if not query:
            continue

        print(f"\n{'='*90}")
        print(f"Query: {query}")
        print(f"{'='*90}")

        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )

        #Check if Retrieval was used
        used_retrieval = any(
            isinstance(message, AIMessage) and message.tool_calls
            for message in result["messages"]
        )

        final_answer = result["messages"][-1].content
        print(f"Agent: {final_answer}")
        print(f"Decision: {'RETRIEVED' if used_retrieval else 'ANSWERED DIRECTLY'}") # We can remove this later
        print(f"\n{'='*90}\n")

query_agent(thread_id="test2")


# def query_agent(query: str, thread_id: str = "default"):
    
#     print(f"\n{'='*70}")
#     print(f"Query: {query}")
#     print(f"{'='*70}")

#     result = agent.invoke(
#         {"messages": [HumanMessage(content=query)]},
#         config={"configurable": {"thread_id": thread_id}}
#     )

#     #Check if Retrieval was used
#     used_retrieval = any(
#         isinstance(message, AIMessage) and message.tool_calls
#         for message in result["messages"]
#     )

#     final_answer = result["messages"][-1].content
#     print(f"Agent: {final_answer}")
#     print(f"Decision: {'RETRIEVED' if used_retrieval else 'ANSWERED DIRECTLY'}") # We can remove this later
#     print(f"\n{'='*70}\n")

# query_agent("What does is the current tax bracket according to the Nigerian tax law 2025?", thread_id="test1")