interface APIResponse {
    code?: string;
    text?: string;
    error?: string;
}

interface APIConfig {
    apiKey: string;
    endpoint: string;
    maxTokens?: number;
    temperature?: number;
}

const API_CONFIGS: Record<string, APIConfig> = {
    deepseek: {
        // API key intentionally not stored in client. Frontend should call server proxy.
        apiKey: '',
        endpoint: '/api/v1/proxy/generate',
        maxTokens: 2048,
        temperature: 0.7
    },
    blackbox: {
        apiKey: '',
        endpoint: '/api/v1/proxy/generate',
        maxTokens: 2048,
        temperature: 0.7
    }
};

async function generateDeepSeekResponse(prompt: string, config: APIConfig): Promise<APIResponse> {
    try {
        // Call server-side proxy which will attach the secret and forward to DeepSeek
        const response = await fetch(config.endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, provider: 'deepseek', max_tokens: config.maxTokens, temperature: config.temperature })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || data.error || 'Failed to generate code with DeepSeek');
        }

        return { code: data.text || data.code || '' };
    } catch (error) {
        console.error('DeepSeek API Error:', error);
        return {
            error: 'Failed to generate code with DeepSeek. Please try again.'
        };
    }
}

async function generateBlackboxResponse(prompt: string, config: APIConfig): Promise<APIResponse> {
    try {
        // Proxy through backend which holds the BLACKBOX_API_KEY
        const response = await fetch(config.endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, provider: 'blackbox', max_tokens: config.maxTokens, temperature: config.temperature })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || data.error || 'Failed to generate code with Blackbox');
        }

        return { code: data.text || data.code || '' };
    } catch (error) {
        console.error('Blackbox API Error:', error);
        return {
            error: 'Failed to generate code with Blackbox. Please try again.'
        };
    }
}

export async function generateCode(prompt: string, provider: string): Promise<APIResponse> {
    const config = API_CONFIGS[provider];
    if (!config) {
        return {
            error: `Unsupported API provider: ${provider}`
        };
    }

    // Add context to the prompt for better code generation
    const enhancedPrompt = `Generate high-quality, production-ready code for the following request:\n\n${prompt}\n\nPlease ensure the code follows best practices and includes proper error handling.`;

    switch (provider) {
        case 'deepseek':
            return generateDeepSeekResponse(enhancedPrompt, config);
        case 'blackbox':
            return generateBlackboxResponse(enhancedPrompt, config);
        default:
            return {
                error: `No implementation found for provider: ${provider}`
            };
    }
}
