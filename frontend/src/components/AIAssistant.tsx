import React, { useState } from 'react';
import { generateCode } from '../services/apiService';
import './GalaxyTheme.css';

interface ModelOption {
    id: string;
    name: string;
    provider: string;
    description: string;
}

const AVAILABLE_MODELS: ModelOption[] = [
    {
        id: 'deepseek-default',
        name: 'DeepSeek Coder',
        provider: 'deepseek',
        description: 'Advanced code generation with DeepSeek 33B model'
    },
    {
        id: 'deepseek-creative',
        name: 'DeepSeek Creative',
        provider: 'deepseek',
        description: 'Creative code solutions with high flexibility'
    },
    {
        id: 'blackbox-default',
        name: 'Blackbox AI',
        provider: 'blackbox',
        description: 'Advanced code generation and completion'
    },
    {
        id: 'blackbox-fast',
        name: 'Blackbox Fast',
        provider: 'blackbox',
        description: 'Quick code snippets and solutions'
    }
];

export default function AIAssistant() {
    const [prompt, setPrompt] = useState('');
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(false);
    const [selectedModel, setSelectedModel] = useState(AVAILABLE_MODELS[0].id);
    const [error, setError] = useState('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            const model = AVAILABLE_MODELS.find(m => m.id === selectedModel);
            if (!model) throw new Error('Invalid model selected');

            const result = await generateCode(prompt, model.provider);
            
            if (result.error) {
                setError(result.error);
            } else {
                setResponse(result.code || result.text || '');
            }
        } catch (err) {
            setError('Failed to generate code. Please try again.');
            console.error('Error:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="galaxy-background min-h-screen text-white p-6">
            {/* Nebula Effects */}
            <div className="nebula-glow w-96 h-96 top-0 right-0" />
            <div className="nebula-glow w-64 h-64 bottom-0 left-0" />

            <div className="max-w-7xl mx-auto">
                <header className="text-center mb-12">
                    <h1 className="text-4xl font-bold mb-4">
                        <span className="text-accent">Galaxy</span> Code Assistant
                    </h1>
                    <p className="text-lg opacity-80">Your AI copilot for the final frontier</p>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Input Section */}
                    <div className="editor-theme p-6">
                        <div className="mb-6">
                            <label htmlFor="model-select" className="block text-sm font-medium mb-2">
                                Select AI Model
                            </label>
                            <select
                                id="model-select"
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="bg-background text-text px-4 py-2 rounded-lg w-full mb-4"
                            >
                                {AVAILABLE_MODELS.map((model) => (
                                    <option key={model.id} value={model.id}>
                                        {model.name}
                                    </option>
                                ))}
                            </select>
                            
                            <form onSubmit={handleSubmit}>
                                <label htmlFor="prompt-input" className="block text-sm font-medium mb-2">
                                    Describe Your Code
                                </label>
                                <textarea
                                    id="prompt-input"
                                    value={prompt}
                                    onChange={(e) => setPrompt(e.target.value)}
                                    placeholder="Describe the code you want to generate..."
                                    className="w-full h-48 bg-background text-text p-4 rounded-lg mb-4"
                                />
                                <button
                                    type="submit"
                                    disabled={loading}
                                    className="warp-button w-full"
                                >
                                    {loading ? 'Generating...' : 'Generate Code'}
                                </button>
                            </form>
                        </div>
                    </div>

                    {/* Output Section */}
                    <div className="editor-theme p-6">
                        <h2 className="text-xl font-semibold mb-4">Generated Code</h2>
                        {error && (
                            <div className="text-red-500 mb-4 p-3 bg-red-900/20 rounded">
                                {error}
                            </div>
                        )}
                        <pre className="console-text bg-background p-4 rounded-lg overflow-auto max-h-[500px]">
                            {response || 'Your generated code will appear here...'}
                        </pre>
                    </div>
                </div>

                {/* Model Info */}
                <div className="mt-8 text-sm opacity-70">
                    <p>Currently using: {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.description}</p>
                </div>
            </div>
        </div>
    );
}
