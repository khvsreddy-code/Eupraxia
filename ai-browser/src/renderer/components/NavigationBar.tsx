import React from 'react';

const NavigationBar: React.FC = () => {
  return (
    <nav className="navigation-bar glass-panel">
      <div className="nav-controls">
        <button className="nav-button glow-button">
          <span className="icon">←</span>
        </button>
        <button className="nav-button glow-button">
          <span className="icon">→</span>
        </button>
        <button className="nav-button glow-button">
          <span className="icon">↻</span>
        </button>
      </div>
      <div className="address-bar glass-panel">
        <input 
          type="text" 
          className="url-input"
          placeholder="Enter URL or search..."
        />
        <div className="security-indicator">
          <span className="icon">🔒</span>
        </div>
      </div>
      <div className="browser-actions">
        <button className="action-button glow-button">
          <span className="icon">⋮</span>
        </button>
      </div>
    </nav>
  );
};

export default NavigationBar;