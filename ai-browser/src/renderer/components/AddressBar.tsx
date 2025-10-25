import React, { useState } from 'react';
import './AddressBar.css';

interface AddressBarProps {
  currentUrl: string;
  onNavigate: (url: string) => void;
  onAddBookmark: () => void;
}

export const AddressBar: React.FC<AddressBarProps> = ({
  currentUrl,
  onNavigate,
  onAddBookmark,
}) => {
  const [inputValue, setInputValue] = useState(currentUrl);
  const [isFocused, setIsFocused] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    let url = inputValue;
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      // Check if it's a valid domain
      if (url.includes('.') && !url.includes(' ')) {
        url = `https://${url}`;
      } else {
        // Treat as search query
        url = `https://www.google.com/search?q=${encodeURIComponent(url)}`;
      }
    }
    onNavigate(url);
  };

  return (
    <div className="address-bar">
      <div className="navigation-controls">
        <button className="nav-button" onClick={() => window.history.back()}>
          ←
        </button>
        <button className="nav-button" onClick={() => window.history.forward()}>
          →
        </button>
        <button className="nav-button" onClick={() => onNavigate(currentUrl)}>
          ↻
        </button>
      </div>

      <form className={`url-bar ${isFocused ? 'focused' : ''}`} onSubmit={handleSubmit}>
        <div className="security-indicator">
          {currentUrl.startsWith('https://') ? '🔒' : ''}
        </div>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder="Search Google or enter a URL"
        />
      </form>

      <div className="address-actions">
        <button className="action-button" onClick={onAddBookmark}>
          ⭐
        </button>
        <button className="action-button">⋮</button>
      </div>
    </div>
  );
};