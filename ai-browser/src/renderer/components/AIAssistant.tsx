import React, { useState, useEffect, useRef } from 'react';
import './AIAssistant.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  type?: 'text' | 'code' | 'image' | 'audio';
}

interface AIAssistantProps {
  onClose: () => void;
  currentUrl: string;
  initialMessage?: string;
}

export const AIAssistant: React.FC<AIAssistantProps> = ({ onClose, currentUrl, initialMessage }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (initialMessage && initialMessage.length > 0) {
      setInput(initialMessage);
      // focus the input when initial message is set
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [initialMessage]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      type: 'text'
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsProcessing(true);

    try {
      const response = await window.electronAPI?.sendChatQuery(input);

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.text,
        type: response.type || 'text'
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error getting AI response:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        type: 'text'
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="ai-assistant glass-panel">
      <div className="ai-header">
        <span className="glow-text">AI Assistant</span>
        <div className="ai-controls">
          <button className="ai-button" onClick={() => setMessages([])}>
            Clear
          </button>
          <button className="ai-button" onClick={onClose}>
            ✕
          </button>
        </div>
      </div>

      <div className="ai-messages">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.role} ${message.type || 'text'}`}
          >
            {message.type === 'code' ? (
              <pre>
                <code>{message.content}</code>
              </pre>
            ) : (
              message.content
            )}
          </div>
        ))}
        {isProcessing && (
          <div className="message assistant">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="ai-input-form" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me anything..."
          disabled={isProcessing}
          className="ai-input"
        />
        <button
          type="submit"
          disabled={isProcessing || !input.trim()}
          className="ai-submit"
        >
          Send
        </button>
      </form>
    </div>
  );
};