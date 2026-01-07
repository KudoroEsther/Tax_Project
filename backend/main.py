from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import uuid
from datetime import datetime
from agent_logic import agent as graph # importing the compiled graph
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Tax Reform Assistant API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for thread metadata (since LangGraph Checkpointer handles message history)
# Format: { "thread_id": { "title": "...", "updated_at": "..." } }
threads_db = {}

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Tax Reform Assistant API is running"}

@app.get("/threads")
def get_threads():
    # Return sorted threads by updated_at
    sorted_threads = sorted(
        [{"id": k, **v} for k, v in threads_db.items()],
        key=lambda x: x['updated_at'],
        reverse=True
    )
    return sorted_threads

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        thread_id = request.thread_id
        is_new = False
        
        if not thread_id:
            thread_id = str(uuid.uuid4())
            is_new = True
            
        # Convert Pydantic messages to LangChain messages
        # We only really need to pass the *new* message if we are using thread_id with checkpointer,
        # but the frontend sends history. With LangGraph MemorySaver and thread_id, 
        # we can just pass the latest user message and let the graph handle state.
        # However, to support the current frontend structure effectively, let's extract the last user message.
        
        if not request.messages:
             raise HTTPException(status_code=400, detail="No messages provided")
             
        last_user_msg = request.messages[-1]
        if last_user_msg['role'] != 'user':
             raise HTTPException(status_code=400, detail="Last message must be from user")

        input_message = HumanMessage(content=last_user_msg['content'])
        
        # Invoke agent with thread_id
        config = {"configurable": {"thread_id": thread_id}}
        
        # For the very first message, we might want to generate a title
        if is_new or thread_id not in threads_db:
             title = last_user_msg['content'][:30] + "..."
             threads_db[thread_id] = {
                 "title": title,
                 "created_at": datetime.now().isoformat(),
                 "updated_at": datetime.now().isoformat()
             }

        # Update timestamp
        threads_db[thread_id]['updated_at'] = datetime.now().isoformat()

        # Run Graph
        # We pass ONLY the new message because MemorySaver recalls the rest
        result = graph.invoke({"messages": [input_message]}, config=config)
        
        # Get last message from result
        last_msg = result['messages'][-1]
        response_text = last_msg.content
        
        return {"response": response_text, "thread_id": thread_id}
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
