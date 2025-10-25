import React, { useState } from 'react'
import Starfield from './components/Starfield'
import ChatInterface from './components/ChatInterface'
import './App.css'

function App() {
  return (
    <div className="app">
      <Starfield />
      <div className="content">
        <header className="header">
          <h1 className="logo">
            <span className="logo-text">Eupraxia</span>
            <span className="logo-subtitle">Pleroma</span>
          </h1>
          <div className="tagline">Your AI Universe</div>
        </header>
        
        <main className="main">
          <ChatInterface />
        </main>
      </div>
    </div>
  )
}

export default App