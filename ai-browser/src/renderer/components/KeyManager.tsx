import React, { useEffect, useState } from 'react';
import './KeyManager.css';

const API_KEYS = [
    { id: 'openai', name: 'OpenAI API Key', env: 'OPENAI_API_KEY' },
    { id: 'hf', name: 'Hugging Face Token', env: 'HF_API_TOKEN' },
    { id: 'deepseek', name: 'DeepSeek API Key', env: 'DEEPSEEK_API_KEY' },
    { id: 'google', name: 'Google API Key', env: 'GOOGLE_API_KEY' },
];

interface KeyEntry {
    id: string;
    name: string;
    env: string;
    value?: string;
    isSet?: boolean;
}

interface KeyManagerProps {
    onClose: () => void;
}

const KeyManager: React.FC<KeyManagerProps> = ({ onClose }) => {
    const [keys, setKeys] = useState<KeyEntry[]>(API_KEYS);
    const [status, setStatus] = useState<string>('');
    const [error, setError] = useState<string>('');

    useEffect(() => {
        // Check which keys are set
        const checkKeys = async () => {
            try {
                const response = await fetch('http://127.0.0.1:8000/health');
                if (!response.ok) throw new Error('Failed to check API keys');
                const data = await response.json();

                setKeys(prev => prev.map(key => ({
                    ...key,
                    isSet: data[`has_${key.id}_key`] || false
                })));
            } catch (e) {
                setError('Failed to connect to RAG server');
            }
        };
        checkKeys();
    }, []);

    const handleSave = async (key: KeyEntry) => {
        if (!key.value?.trim()) {
            setError(`Please enter a value for ${key.name}`);
            return;
        }

        try {
            // Save to .env via secure endpoint
            const response = await fetch('http://127.0.0.1:8000/keys/set', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    key: key.env,
                    value: key.value
                })
            });

            if (!response.ok) throw new Error('Failed to save key');

            setKeys(prev => prev.map(k =>
                k.id === key.id ? { ...k, value: '', isSet: true } : k
            ));
            setStatus(`${key.name} saved successfully`);
            setTimeout(() => setStatus(''), 3000);
        } catch (e) {
            setError(`Failed to save ${key.name}`);
        }
    };

    const handleClear = async (key: KeyEntry) => {
        try {
            const response = await fetch('http://127.0.0.1:8000/keys/clear', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key: key.env })
            });

            if (!response.ok) throw new Error('Failed to clear key');

            setKeys(prev => prev.map(k =>
                k.id === key.id ? { ...k, value: '', isSet: false } : k
            ));
            setStatus(`${key.name} cleared`);
            setTimeout(() => setStatus(''), 3000);
        } catch (e) {
            setError(`Failed to clear ${key.name}`);
        }
    };

    return (
        <div className="key-manager">
            <div className="key-manager-header">
                <h3>API Key Management</h3>
                <button onClick={onClose}>✕</button>
            </div>

            <div className="key-list">
                {keys.map(key => (
                    <div key={key.id} className="key-entry">
                        <div className="key-info">
                            <span>{key.name}</span>
                            {key.isSet && (
                                <span className="key-status">✓ Set</span>
                            )}
                        </div>

                        <div className="key-actions">
                            <input
                                type="password"
                                placeholder={key.isSet ? '••••••••' : 'Enter API key'}
                                value={key.value || ''}
                                onChange={e => setKeys(prev => prev.map(k =>
                                    k.id === key.id ? { ...k, value: e.target.value } : k
                                ))}
                            />

                            <button
                                onClick={() => key.isSet ? handleClear(key) : handleSave(key)}
                                className={key.isSet ? 'danger' : 'primary'}
                            >
                                {key.isSet ? 'Clear' : 'Save'}
                            </button>
                        </div>
                    </div>
                ))}
            </div>

            {status && <div className="key-status-message success">{status}</div>}
            {error && <div className="key-status-message error">{error}</div>}
        </div>
    );
};

export default KeyManager;