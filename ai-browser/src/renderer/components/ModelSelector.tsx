import React from 'react';
import './ModelSelector.css';

interface ModelSelectorProps {
    useLocal: boolean;
    onToggle: (useLocal: boolean) => void;
    isConnected?: boolean;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ useLocal, onToggle, isConnected = true }) => {
    return (
        <div className="model-selector">
            <div className="model-toggle">
                <button
                    className={`toggle-btn ${!useLocal ? 'active' : ''}`}
                    onClick={() => onToggle(false)}
                    disabled={!isConnected}
                >
                    OpenAI
                </button>
                <button
                    className={`toggle-btn ${useLocal ? 'active' : ''}`}
                    onClick={() => onToggle(true)}
                    disabled={!isConnected}
                >
                    Local
                </button>
            </div>
            {!isConnected && (
                <div className="connection-error">
                    RAG server not connected
                </div>
            )}
        </div>
    );
};

export default ModelSelector;