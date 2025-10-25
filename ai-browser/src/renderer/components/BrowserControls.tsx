import React from 'react';
import './BrowserControls.css';

interface BrowserControlsProps {
    onNewTab: () => void;
    onToggleBookmarks: () => void;
    onToggleSidebar: () => void;
    onToggleAI: () => void;
}

export const BrowserControls: React.FC<BrowserControlsProps> = ({ onNewTab, onToggleBookmarks, onToggleSidebar, onToggleAI }) => {
    return (
        <div className="browser-controls">
            <button onClick={onNewTab} title="New tab">＋</button>
            <button onClick={onToggleBookmarks} title="Bookmarks">🔖</button>
            <button onClick={onToggleSidebar} title="Sidebar">📚</button>
            <button onClick={onToggleAI} title="AI Assistant">🤖</button>
        </div>
    );
};
