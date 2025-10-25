import React, { useState, useEffect, useRef } from 'react';
import './AIAssistant.css';
import { RagService } from '../services/rag-service';
import ModelSelector from './ModelSelector';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  context?: string[];
  initialMessage?: string;
}

export const AIAssistant: React.FC<AIAssistantProps> = ({ initialMessage }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState(initialMessage || '');
  const [isConnected, setIsConnected] = useState(false);
  const [useLocal, setUseLocal] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-focus and select initial message if provided
  useEffect(() => {
    if (initialMessage && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.setSelectionRange(0, initialMessage.length);
    }
  }, [initialMessage]);

  // Check RAG server connection
  useEffect(() => {
    const checkConnection = async () => {
      const connected = await RagService.checkHealth();
      setIsConnected(connected);
    };
    checkConnection();
    // Poll every 30s
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || !isConnected || isProcessing) return;

    const userMessage: Message = {
      role: 'user',
      content: inputValue
    };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsProcessing(true);

    try {
      const response = await RagService.generate({
        prompt: inputValue,
        use_local: useLocal,
        temperature: 0.7,
        max_tokens: 1024,
        top_k: 3
      });

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.text,
        context: response.context
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Generation error:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again or check if the RAG server is running.'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  return (
    <div className="ai-assistant">
      <div className="ai-header">
        <h3>AI Assistant</h3>
        <ModelSelector
          useLocal={useLocal}
          onToggle={setUseLocal}
          isConnected={isConnected}
        />
      </div>

      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-content">{msg.content}</div>
            {msg.context && (
              <div className="message-context">
                <details>
                  <summary>Retrieved Context</summary>
                  {msg.context.map((ctx, i) => (
                    <div key={i} className="context-item">{ctx}</div>
                  ))}
                </details>
              </div>
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

      <form onSubmit={handleSubmit} className="input-form">
        <textarea
          ref={inputRef}
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={isConnected ? "Ask anything..." : "RAG server not connected"}
          disabled={!isConnected || isProcessing}
        />
        <button
          type="submit"
          disabled={!isConnected || isProcessing || !inputValue.trim()}
        >
          Send
        </button>
      </form>
    </div>
  );
};