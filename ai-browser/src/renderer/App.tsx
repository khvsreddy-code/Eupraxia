import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import * as THREE from 'three';
import BackgroundVisualization from './components/BackgroundVisualization';
import NavigationBar from './components/NavigationBar';
import SearchBar from './components/SearchBar';
import AIAssistant from './components/AIAssistant';
import TabManager from './components/TabManager';
import './styles/advanced.css';

// Expose the electronAPI types that the preload script attaches
declare global {
  interface Window {
    electronAPI?: {
      sendChatQuery: (query: string) => Promise<any>;
      fetchPage?: (url: string) => Promise<any>;
      searchMultiModal?: (params: {
        text?: string;
        audio?: ArrayBuffer;
        image?: ArrayBuffer;
      }) => Promise<any>;
      generateContent?: (params: {
        type: 'image' | 'video' | 'voice' | 'game';
        prompt: string;
        config?: any;
      }) => Promise<any>;
    };
  }
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  type?: 'text' | 'image' | 'audio' | 'video';
  metadata?: any;
}

const App: React.FC = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [scene, setScene] = useState<THREE.Scene | null>(null);

  useEffect(() => {
    // Initialize Three.js
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer();
    
    renderer.setSize(400, 300);
    document.getElementById('three-container')?.appendChild(renderer.domElement);
    
    camera.position.z = 5;

    // Add a simple cube
    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const cube = new THREE.Mesh(geometry, material);
    scene.add(cube);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      cube.rotation.x += 0.01;
      cube.rotation.y += 0.01;
      renderer.render(scene, camera);
    };
    animate();

    setScene(scene);

    return () => {
      renderer.dispose();
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user' as const, content: input };
    setMessages(prev => [...prev, userMessage]);
    
    try {
      // Prefer IPC (preload) when available to avoid depending on an HTTP server
      let data: any = null;
      if (window.electronAPI?.sendChatQuery) {
        data = await window.electronAPI.sendChatQuery(input);
      } else {
        const resp = await fetch('http://localhost:3000/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: input }),
        });
        data = await resp.json();
      }

      const assistantText = data?.response ?? data?.error ?? 'No response';
      const assistantMessage = { role: 'assistant' as const, content: assistantText };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      const assistantMessage = { role: 'assistant' as const, content: 'Error: failed to get response' };
      setMessages(prev => [...prev, assistantMessage]);
    }

    setInput('');
  };

  return (
    <div className="app">
      <div className="main-container">
        <div className="chat-container">
          <div className="messages">
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                {msg.content}
              </div>
            ))}
          </div>
          <form onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything..."
            />
            <button type="submit">Send</button>
          </form>
        </div>
        <div id="three-container" className="visualization"></div>
      </div>
    </div>
  );
};

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}