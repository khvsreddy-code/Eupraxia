/**
 * RAG service for interacting with the FastAPI backend
 */

const RAG_SERVER = 'http://127.0.0.1:8000';

export interface GenerateRequest {
    prompt: string;
    top_k?: number;
    use_local?: boolean;
    temperature?: number;
    max_tokens?: number;
}

export interface GenerateResponse {
    source: 'local' | 'openai' | 'retrieval_only';
    text: string;
    context?: string[];
}

export interface SearchRequest {
    query: string;
    top_k?: number;
}

export interface IngestRequest {
    id?: string;
    text: string;
    metadata?: Record<string, any>;
}

export class RagService {
    static async generate(request: GenerateRequest): Promise<GenerateResponse> {
        const response = await fetch(`${RAG_SERVER}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`RAG server error: ${error}`);
        }

        return response.json();
    }

    static async search(request: SearchRequest) {
        const response = await fetch(`${RAG_SERVER}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`RAG server error: ${error}`);
        }

        return response.json();
    }

    static async ingest(request: IngestRequest) {
        const response = await fetch(`${RAG_SERVER}/ingest`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`RAG server error: ${error}`);
        }

        return response.json();
    }

    static async checkHealth() {
        try {
            const response = await fetch(`${RAG_SERVER}/health`);
            if (!response.ok) return false;
            const data = await response.json();
            return data.status === 'ok';
        } catch (e) {
            console.error('RAG server health check failed:', e);
            return false;
        }
    }
}