import { useState, useRef, useEffect } from 'react'
import { PlusIcon, ChatBubbleLeftIcon, PaperAirplaneIcon } from '@heroicons/react/24/outline'

function App() {
  const [threads, setThreads] = useState([])
  const [currentThreadId, setCurrentThreadId] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const messagesEndRef = useRef(null)

  // Fetch threads on load
  useEffect(() => {
    fetchThreads()
  }, [])

  const fetchThreads = async () => {
    try {
      const res = await fetch('http://localhost:8000/threads')
      if (res.ok) {
        const data = await res.json()
        setThreads(data)
      }
    } catch (e) {
      console.error("Failed to fetch threads", e)
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const createNewChat = () => {
    setCurrentThreadId(null)
    setMessages([])
    setInput('')
  }

  const selectThread = (threadId) => {
    // Ideally we would fetch message history for this thread from backend
    // But for now, since backend API is stateless regarding GET history (MemorySaver is internal),
    // we might start empty or need an endpoint to sync history.
    // LIMITATION: Simple implementation doesn't sync old messages to frontend yet.
    // For now, let's just switch context ID so new messages go to that thread.
    // To make it real, we'd need GET /threads/{id}/messages. 
    // Let's assume for this MVP we just switch ID and clear screen (imperfect but functional for "new chat" feeling).
    // Or better: Just switch ID.
    setCurrentThreadId(threadId)
    setMessages([]) // Clearing because we can't fetch old ones easily without new endpoint
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const payload = {
        messages: [userMessage], // Only sending new one is sufficient for backend logic now
        thread_id: currentThreadId
      }

      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) throw new Error('Network error')

      const data = await response.json()

      // Update thread ID if it was a new chat
      if (!currentThreadId && data.thread_id) {
        setCurrentThreadId(data.thread_id)
        fetchThreads() // Refresh sidebar
      }

      const botMessage = { role: 'assistant', content: data.response }
      setMessages(prev => [...prev, botMessage])

    } catch (error) {
      console.error(error)
      setMessages(prev => [...prev, { role: 'assistant', content: "Error: Could not reach the assistant." }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100 font-sans overflow-hidden">

      {/* Sidebar */}
      <aside className="w-64 bg-gray-950 flex flex-col border-r border-gray-800">
        <div className="p-4">
          <button
            onClick={createNewChat}
            className="w-full flex items-center gap-3 px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-md transition-colors border border-gray-700 text-sm font-medium"
          >
            <PlusIcon className="w-5 h-5" />
            New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2">
          <div className="space-y-1">
            {threads.map(thread => (
              <button
                key={thread.id}
                onClick={() => selectThread(thread.id)}
                className={`w-full text-left px-3 py-3 rounded-md text-sm truncate flex items-center gap-3 transition-colors ${currentThreadId === thread.id ? 'bg-gray-800 text-white' : 'text-gray-400 hover:bg-gray-900 hover:text-gray-200'
                  }`}
              >
                <ChatBubbleLeftIcon className="w-4 h-4 shrink-0" />
                <span className="truncate">{thread.title}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="p-4 border-t border-gray-800 text-xs text-gray-500">
          Tax Reform Assistant v1.0
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col bg-gray-800 relative">
        {/* Header (Optional mobile helper) */}
        <div className="md:hidden p-4 bg-gray-900 border-b border-gray-800 text-center">
          Tax Assistant
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 md:p-10 space-y-6">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-gray-400 opacity-50">
              <div className="text-4xl mb-4">🇳🇬</div>
              <h2 className="text-xl font-semibold">Nigerian Tax Reform Assistant</h2>
              <p className="mt-2 text-sm text-center max-w-md">
                Ask any question about the 2024 Tax Reform Bills.
              </p>
            </div>
          )}

          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex gap-4 max-w-3xl mx-auto ${msg.role === 'user' ? 'justify-end' : ''}`}
            >
              <div className={`p-1 rounded-sm w-8 h-8 flex items-center justify-center shrink-0 ${msg.role === 'assistant' ? 'bg-green-600' : 'hidden'
                }`}>
                {msg.role === 'assistant' && "AI"}
              </div>

              <div className={`prose prose-invert max-w-none flex-1 leading-7 ${msg.role === 'user' ? 'bg-gray-700 px-4 py-2 rounded-2xl' : ''
                }`}>
                <div className="whitespace-pre-wrap">{msg.content}</div>
              </div>

              <div className={`p-1 rounded-sm w-8 h-8 flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-gray-500' : 'hidden'
                }`}>
                U
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-4 max-w-3xl mx-auto">
              <div className="p-1 rounded-sm w-8 h-8 bg-green-600 flex items-center justify-center shrink-0">AI</div>
              <div className="animate-pulse flex items-center">Thinking...</div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 max-w-3xl mx-auto w-full pb-10">
          <form onSubmit={handleSubmit} className="relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question..."
              className="w-full bg-gray-700 text-white placeholder-gray-400 border border-gray-600 rounded-xl px-4 py-4 pr-12 focus:outline-none focus:border-gray-500 focus:bg-gray-700 shadow-lg"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-green-600 hover:bg-green-700 text-white rounded-md disabled:opacity-50 disabled:bg-gray-600 transition-colors"
            >
              <PaperAirplaneIcon className="w-4 h-4" />
            </button>
          </form>
          <p className="text-center text-xs text-gray-500 mt-2">
            AI can make mistakes. Verify information with official documents.
          </p>
        </div>

      </main>
    </div>
  )
}

export default App
