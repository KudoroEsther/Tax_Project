# TaxGPT - Nigerian Tax Reform Assistant

An Agentic RAG-powered AI Assistant built to help Nigerians navigate and understand the 2024/2025 tax reform bills with precision and clarity.

## 📋 Project Overview

TaxGPT leverages an **Agentic RAG** architecture to provide authoritative answers on Nigerian tax reforms. It intelligently decides when to retrieve information from a curated knowledge base and when to answer directly, ensuring high accuracy for complex tax queries.

### Tech Stack
- **AI Engine**: LangGraph for agentic workflow and conversation state
- **LLM**: OpenAI GPT models
- **Vector Database**: ChromaDB for semantic retrieval of tax documents
- **Backend**: FastAPI (Python)
- **Frontend**: React + Vite + Tailwind CSS

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAI API Key

### 1. Install Dependencies
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Application

**Start Backend:**
```bash
cd backend
python main.py
```
*Backend runs at `http://localhost:8000`*

**Start Frontend:**
```bash
cd frontend
npm run dev
```
*Frontend runs at `http://localhost:5173`*

## ✨ Key Features

- **Agentic Retrieval**: Uses conditional logic to pull data from official tax documents only when needed.
- **Persistent History**: Conversations are saved and can be restored from the sidebar.
- **Thread Management**: Start new chats or delete old ones from your history.
- **Dark/Light Mode**: Premium UI with a theme toggle for comfortable reading.
- **Up-to-Date Knowledge**: Aware of name changes like **FIRS** becoming the **NRS (Nigeria Revenue Service)**.
- **Source Citations**: Clearly indicates which documents were used to generate answers.

## 📁 Project Structure

```
Tax_Project/
├── backend/
│   ├── main.py              # FastAPI server & API endpoints
│   ├── agent_logic.py       # LangGraph agent & tool definitions
│   └── utils_openai.py      # OpenAI & VectorStore utilities
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main UI component
│   │   └── index.css        # Global styles & theme tokens
│   └── index.html           # Entry point
├── chroma_db/               # Persistent vector storage
├── .env                     # Configuration keys
└── requirements.txt         # Python dependencies
```

## 🎯 API Reference

- `GET /threads`: Retrieve all active chat sessions.
- `GET /threads/{id}/messages`: Fetch message history for a specific thread.
- `POST /chat`: Submit a message to the agent.
- `DELETE /threads/{id}`: Remove a thread from history.

## 📝 Important Notes

- **FIRS Rebrand**: The system is instructed to refer to the Federal Inland Revenue Service as the **Nigeria Revenue Service (NRS)** per the new reforms.
- **Memory**: The current version uses an in-memory checkpointer (`MemorySaver`). Restarting the backend will clear message history, though thread metadata is currently stored in a volatile `threads_db` dictionary for this MVP.
- **Accuracy**: Always verify AI-generated tax advice with official NRS publications.

## ✅ Project Status

✅ Agentic RAG implementation  
✅ Conversation persistence  
✅ Theme customization  
✅ Thread management (Create/Select/Delete)  
✅ Automated source citations  
