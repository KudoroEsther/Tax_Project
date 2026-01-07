# Nigerian Tax Reform Bills 2024 Q&A Assistant

An Agentic RAG-powered AI Assistant that helps Nigerians understand the new tax reform bills.

## 📋 Project Overview

This project implements:
- **Agentic RAG** using LangGraph for intelligent document retrieval
- **Vector Database** (ChromaDB) for semantic search
- **FastAPI Backend** with conversation memory
- **React Frontend** with ChatGPT-style interface

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+
- OpenAI API Key

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create/edit `.env` file in the root directory:
```
OPENAI_API_KEY=your_key_here
```

### 3. Start Backend
```bash
cd backend
py -3.11 -m uvicorn main:app --reload
# Or if you only have one python version:
# python -m uvicorn main:app --reload
```
Backend runs at: `http://localhost:8000`

### 4. Start Frontend
Open a new terminal:
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at: `http://localhost:5173`

## 📁 Project Structure

```
Tax_Project/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── agent_logic.py       # LangGraph agent implementation
│   └── utils_openai.py      # Utility functions
├── frontend/
│   └── src/
│       ├── App.jsx          # Main React component
│       └── index.css        # Tailwind styles
├── chroma_db/               # Vector database (pre-populated)
├── .env                     # API keys (not in git)
└── requirements.txt         # Python dependencies
```

## ✨ Features

### Backend
- **Conditional Retrieval**: Only searches documents when needed
- **Conversation Memory**: Maintains context across messages
- **Session Management**: Multiple chat threads support
- **Source Citation**: References document sources

### Frontend
- **ChatGPT-style UI**: Dark theme with sidebar
- **Chat History**: View and switch between conversations
- **New Chat**: Start fresh conversations
- **Real-time Updates**: Streaming-ready interface

## 🎯 Key Endpoints

- `GET /` - Health check
- `GET /threads` - List all chat sessions
- `POST /chat` - Send message and get response
  ```json
  {
    "messages": [{"role": "user", "content": "question"}],
    "thread_id": "optional-uuid"
  }
  ```

## 🧪 Testing

1. Start both backend and frontend
2. Open browser to `http://localhost:5173`
3. Try these queries:
   - "Hi" → Should greet without retrieval
   - "What are the tax reform bills about?" → Should retrieve documents
   - "Which states benefit most?" → Should cite sources

## 📝 Notes

- The vector database (`chroma_db`) is pre-populated with tax reform documents
- All conversations are stored in memory (restart clears history)
- The agent uses `gpt-4o` model by default
- Frontend communicates with backend via REST API

## 🎓 Academic Requirements Met

✅ Agentic RAG with conditional retrieval  
✅ Conversation memory using LangGraph  
✅ Source citations  
✅ FastAPI backend  
✅ React frontend with clean UI  
✅ README with setup instructions  

**Note**: Demo video should be created separately as per assignment requirements.
