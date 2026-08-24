import React, { useEffect, useRef } from 'react';
import { Loader2 } from 'lucide-react';
import Citation from './Citation';

const ChatBox = ({ messages }) => {
  const bottomRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="messages-area">
      {messages.map((msg) => (
        <div key={msg.id} className={`message-wrapper ${msg.role}`}>
          <div className="message-bubble">
            {msg.role === 'ai' && msg.statusUpdates && msg.statusUpdates.length > 0 && (
              <div className="agent-status">
                <Loader2 className="status-icon" />
                <span>{msg.statusUpdates[0]}</span>
              </div>
            )}
            
            <div dangerouslySetInnerHTML={{ __html: formatMessageText(msg.text) }} />
            
            {msg.isStreaming && !msg.text && (
              <span className="typing-indicator" style={{ opacity: 0.5 }}>...</span>
            )}

            {msg.citations && msg.citations.length > 0 && (
              <div className="citations-container">
                {msg.citations.map((cite, idx) => (
                  <Citation key={idx} text={cite} />
                ))}
              </div>
            )}
          </div>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
};

// Simple text formatter to handle bolding and basic markdown-like structures
const formatMessageText = (text) => {
  if (!text) return '';
  
  let formatted = text
    // Handle bold text **text**
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // Handle newlines
    .replace(/\n/g, '<br />');

  return formatted;
};

export default ChatBox;
