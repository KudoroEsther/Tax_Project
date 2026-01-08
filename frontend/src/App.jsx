import { useState, useRef, useEffect } from 'react'
import { PlusIcon, ChatBubbleLeftIcon, PaperAirplaneIcon, SunIcon, MoonIcon, TrashIcon } from '@heroicons/react/24/outline'

function App() {
  const [threads, setThreads] = useState([])
  const [currentThreadId, setCurrentThreadId] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [darkMode, setDarkMode] = useState(true)

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

  const selectThread = async (threadId) => {
    setCurrentThreadId(threadId)
    setIsLoading(true)

    try {
      const res = await fetch(`http://localhost:8000/threads/${threadId}/messages`)
      if (res.ok) {
        const data = await res.json()
        setMessages(data.messages || [])
      } else {
        setMessages([])
      }
    } catch (e) {
      console.error("Failed to fetch thread messages", e)
      setMessages([])
    } finally {
      setIsLoading(false)
    }
  }

  const deleteThread = async (e, threadId) => {
    e.stopPropagation() // Prevent selecting thread when clicking delete


    try {
      const res = await fetch(`http://localhost:8000/threads/${threadId}`, {
        method: 'DELETE'
      })

      if (res.ok) {
        if (currentThreadId === threadId) {
          createNewChat()
        }
        fetchThreads()
      }
    } catch (e) {
      console.error("Failed to delete thread", e)
    }
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
        messages: [userMessage],
        thread_id: currentThreadId
      }

      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) throw new Error('Network error')

      const data = await response.json()

      if (!currentThreadId && data.thread_id) {
        setCurrentThreadId(data.thread_id)
        fetchThreads()
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

  const toggleTheme = () => {
    setDarkMode(!darkMode)
  }

  return (
    <div className={`flex h-screen font-sans overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-gray-900 text-gray-100' : 'bg-gray-100 text-gray-900'
      }`}>

      {/* Sidebar */}
      <aside className={`w-64 flex flex-col border-r transition-colors duration-300 ${darkMode ? 'bg-gray-950 border-gray-800' : 'bg-white border-gray-200'
        }`}>
        <div className="p-4">
          <button
            onClick={createNewChat}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-md transition-colors border text-sm font-medium ${darkMode
              ? 'bg-gray-800 hover:bg-gray-700 border-gray-700'
              : 'bg-gray-100 hover:bg-gray-200 border-gray-300'
              }`}
          >
            <PlusIcon className="w-5 h-5" />
            New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2">
          <div className="space-y-1">
            {threads.map(thread => (
              <div key={thread.id} className="group relative">
                <button
                  onClick={() => selectThread(thread.id)}
                  className={`w-full text-left px-3 py-3 rounded-md text-sm truncate flex items-center gap-3 transition-colors ${currentThreadId === thread.id
                    ? (darkMode ? 'bg-gray-800 text-white' : 'bg-gray-200 text-gray-900')
                    : (darkMode ? 'text-gray-400 hover:bg-gray-900 hover:text-gray-200' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900')
                    }`}
                >
                  <ChatBubbleLeftIcon className="w-4 h-4 shrink-0" />
                  <span className="truncate pr-6">{thread.title}</span>
                </button>
                <button
                  onClick={(e) => deleteThread(e, thread.id)}
                  className={`absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-md opacity-0 group-hover:opacity-100 transition-opacity ${darkMode ? 'text-gray-500 hover:text-red-400 hover:bg-gray-800' : 'text-gray-400 hover:text-red-500 hover:bg-gray-200'
                    }`}
                  title="Delete chat"
                >
                  <TrashIcon className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className={`p-4 border-t flex items-center justify-between text-xs ${darkMode ? 'border-gray-800 text-gray-500' : 'border-gray-200 text-gray-500'
          }`}>
          <span>TaxGPT v1.0</span>
          <button
            onClick={toggleTheme}
            className={`p-2 rounded-md transition-colors ${darkMode
              ? 'hover:bg-gray-800 text-gray-400 hover:text-yellow-400'
              : 'hover:bg-gray-200 text-gray-600 hover:text-gray-900'
              }`}
            title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {darkMode ? <SunIcon className="w-5 h-5" /> : <MoonIcon className="w-5 h-5" />}
          </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className={`flex-1 flex flex-col relative transition-colors duration-300 ${darkMode ? 'bg-gray-800' : 'bg-gray-50'
        }`}>
        {/* Header (Mobile) */}
        <div className={`md:hidden p-6 border-b text-center transition-colors duration-300 ${darkMode ? 'bg-gray-900 border-gray-800' : 'bg-white border-gray-200'
          }`}>
          TaxGPT
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 md:p-10 space-y-6">
          {messages.length === 0 && (
            <div className={`h-full flex flex-col items-center justify-center opacity-50 ${darkMode ? 'text-gray-400' : 'text-gray-600'
              }`}>
              <h2 className="text-4xl font-bold">TaxGPT</h2>
              <p className="mt-2 text-sm text-center max-w-md">
                Your AI assistant for Nigerian Tax Reform Bills.
              </p>
            </div>
          )}

          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex gap-4 max-w-3xl mx-auto ${msg.role === 'user' ? 'justify-end' : ''}`}
            >
              <div className={`p-1 rounded-sm w-8 h-8 flex items-center justify-center shrink-0 ${msg.role === 'assistant' ? 'bg-green-600 text-white' : 'hidden'
                }`}>
                {msg.role === 'assistant' && "AI"}
              </div>

              <div className={`prose max-w-none flex-1 leading-7 ${msg.role === 'user'
                ? (darkMode ? 'bg-gray-700 text-white' : 'bg-green-600 text-white') + ' px-4 py-2 rounded-2xl'
                : (darkMode ? 'prose-invert' : '')
                }`}>
                <div className="whitespace-pre-wrap">{msg.content}</div>
              </div>

              <div className={`p-1 rounded-sm w-8 h-8 flex items-center justify-center shrink-0 ${msg.role === 'user' ? (darkMode ? 'bg-gray-500' : 'bg-green-700') + ' text-white' : 'hidden'
                }`}>
                U
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-4 max-w-3xl mx-auto">
              <div className="p-1 rounded-sm w-8 h-8 bg-green-600 flex items-center justify-center shrink-0 text-white">AI</div>
              <div className="animate-pulse flex items-center">Overthinking your message...</div>
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
              className={`w-full border rounded-xl px-4 py-4 pr-12 focus:outline-none shadow-lg transition-colors duration-300 ${darkMode
                ? 'bg-gray-700 text-white placeholder-gray-400 border-gray-600 focus:border-gray-500'
                : 'bg-white text-gray-900 placeholder-gray-500 border-gray-300 focus:border-green-500'
                }`}
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
          <p className={`text-center text-xs mt-2 ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
            AI can make mistakes. Verify information with official documents.
          </p>
        </div>

      </main>
    </div>
  )
}

export default App
