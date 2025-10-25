import React, { useState, useRef } from 'react';

const SearchBar: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [searchMode, setSearchMode] = useState<'text' | 'voice' | 'image'>('text');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleVoiceSearch = async () => {
    setIsRecording(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      // Handle voice recording
      // Implementation for voice search
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  const handleImageSearch = () => {
    fileInputRef.current?.click();
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Handle image search
      // Implementation for image search
    }
  };

  return (
    <div className="search-container glass-panel">
      <div className="search-input-container">
        <input
          type="text"
          className="search-input"
          placeholder="Search anything..."
        />
        <div className="search-actions">
          <button
            className={`voice-search-button ${isRecording ? 'recording' : ''}`}
            onClick={handleVoiceSearch}
          >
            <span className="icon">🎤</span>
            {isRecording && <span className="recording-indicator" />}
          </button>
          <button
            className="image-search-button"
            onClick={handleImageSearch}
          >
            <span className="icon">🖼️</span>
          </button>
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            accept="image/*"
            onChange={handleFileUpload}
          />
        </div>
      </div>
      <div className="search-type-indicator glow-text">
        {searchMode === 'voice' && 'Listening...'}
        {searchMode === 'image' && 'Analyzing image...'}
      </div>
    </div>
  );
};

export default SearchBar;