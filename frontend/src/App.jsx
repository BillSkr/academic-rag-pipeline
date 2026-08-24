import React, { useState } from 'react';
import ChatBox from './components/ChatBox';
import ChatInput from './components/ChatInput';
import { Bot } from 'lucide-react';
import './index.css';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'ai',
      text: 'Hello! I am the Academic RAG Pipeline assistant. I can search through research papers to answer your questions. What would you like to know?',
      citations: [],
      statusUpdates: []
    }
  ]);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleSendMessage = async (text) => {
    if (!text.trim() || isProcessing) return;

    const userMessageId = Date.now().toString();
    const aiMessageId = (Date.now() + 1).toString();

    // Add user message and empty AI message
    setMessages((prev) => [
      ...prev,
      { id: userMessageId, role: 'user', text },
      { id: aiMessageId, role: 'ai', text: '', citations: [], statusUpdates: [], isStreaming: true }
    ]);

    setIsProcessing(true);

    try {
      // Build history payload
      const history = messages
        .filter(m => m.id !== 'welcome')
        .map(m => ({ role: m.role, content: m.text }));

      const apiBase = import.meta.env.VITE_API_BASE_URL || '';
      const response = await fetch(`${apiBase}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text, history })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch response');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      
      let aiText = '';
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // SSE messages are separated by double newlines
        let boundary = buffer.indexOf('\n\n');
        
        while (boundary !== -1) {
          const chunk = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          boundary = buffer.indexOf('\n\n');
          
          if (chunk.startsWith('data: ')) {
            const dataStr = chunk.substring(6);
            if (!dataStr) continue;
            
            try {
              const data = JSON.parse(dataStr);
              
              setMessages((prev) => prev.map(msg => {
                if (msg.id !== aiMessageId) return msg;

                const newMsg = { ...msg };

                if (data.token) {
                  aiText += data.token;
                  newMsg.text = aiText;
                }
                
                if (data.status) {
                  if (data.status === 'completed') {
                    newMsg.isStreaming = false;
                    if (data.response) {
                      newMsg.text = data.response; // Final full response
                    }
                    if (data.citations) {
                      newMsg.citations = data.citations;
                    }
                    newMsg.statusUpdates = []; // Clear status updates when done
                  } else {
                    // It's a status update (e.g., Running RetrieverEvaluator...)
                    newMsg.statusUpdates = [data.status];
                  }
                }
                
                return newMsg;
              }));
              
            } catch (e) {
              console.error('Error parsing SSE data:', e, dataStr);
            }
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages((prev) => prev.map(msg => {
        if (msg.id === aiMessageId) {
          return { ...msg, text: 'Sorry, I encountered an error while processing your request.', isStreaming: false };
        }
        return msg;
      }));
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Academic RAG Pipeline</h1>
        <p>Your intelligent research assistant powered by local LLMs</p>
      </header>
      
      <main className="chat-container">
        <ChatBox messages={messages} />
        <ChatInput onSendMessage={handleSendMessage} disabled={isProcessing} />
      </main>
    </div>
  );
}

export default App;
