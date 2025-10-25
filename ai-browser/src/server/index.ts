import express from 'express';
import { OpenAI } from 'openai';
import net from 'net';

// Try to load .env if present (optional)
try {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  require('dotenv').config();
} catch (e) {
  // dotenv not installed or failed to load - ignore
}

const app = express();
const port = parseInt(process.env.PORT || '3000', 10);

app.use(express.json());

// Initialize OpenAI client only if API key is provided. Do not throw if missing.
const OPENAI_API_KEY = (process.env.OPENAI_API_KEY || '').trim();
let openai: OpenAI | null = null;
let openaiEnabled = false;
if (OPENAI_API_KEY) {
  try {
    openai = new OpenAI({ apiKey: OPENAI_API_KEY });
    openaiEnabled = true;
    console.log('OpenAI client initialized.');
  } catch (err) {
    console.warn('Failed to initialize OpenAI client:', err);
    openai = null;
    openaiEnabled = false;
  }
} else {
  console.warn('OPENAI_API_KEY not set. Chat endpoint will use a local fallback.');
}

// Chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { message } = req.body;
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'message is required' });
    }

    if (!openaiEnabled || !openai) {
      // Local fallback: simple echo with minimal enhancements
      const fallback = `Fallback response (OpenAI API not configured). Echo: ${message}`;
      return res.json({ response: fallback });
    }

    // Call OpenAI Chat Completions
    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
      messages: [{ role: 'user', content: message }],
    });

    const responseText = completion.choices?.[0]?.message?.content || 'No response';
    res.json({ response: responseText });
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ error: 'Failed to get response' });
  }
});

// Programmatic chat handler that can be used by IPC handlers (main process)
export async function handleChatMessage(message: string): Promise<{ response: string } | { error: string }> {
  try {
    if (!message || typeof message !== 'string') {
      return { error: 'message is required' };
    }

    if (!openaiEnabled || !openai) {
      const fallback = `Fallback response (OpenAI API not configured). Echo: ${message}`;
      return { response: fallback };
    }

    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
      messages: [{ role: 'user', content: message }],
    });

    const responseText = completion.choices?.[0]?.message?.content || 'No response';
    return { response: responseText };
  } catch (err: any) {
    console.error('handleChatMessage error:', err);
    return { error: 'Failed to get response' };
  }
}

// Page fetching endpoint
app.get('/api/fetch-page', async (req, res) => {
  try {
    const { url } = req.query;
    if (!url || typeof url !== 'string') {
      return res.status(400).json({ error: 'URL is required' });
    }

    // Use global fetch (Node 18+) or node-fetch if necessary
    const fetchFn: typeof fetch = (globalThis as any).fetch;
    if (!fetchFn) {
      return res.status(500).json({ error: 'Fetch is not available in this environment' });
    }

    const response = await fetchFn(url);
    const text = await response.text();
    res.json({ content: text });
  } catch (error) {
    console.error('Fetch page error:', error);
    res.status(500).json({ error: 'Failed to fetch page' });
  }
});

async function isPortFree(p: number): Promise<boolean> {
  return new Promise((resolve) => {
    const tester = net.createServer()
      .once('error', () => {
        resolve(false);
      })
      .once('listening', () => {
        tester.close();
        resolve(true);
      })
      .listen(p);
  });
}

export const startServer = async () => {
  try {
    const free = await isPortFree(port);
    if (!free) {
      console.warn(`Port ${port} is already in use. Skipping HTTP server startup.`);
      return;
    }

    const server = app.listen(port, () => {
      console.log(`Server running at http://localhost:${port}`);
    });

    server.on('error', (err: any) => {
      console.error(`Server failed to start on port ${port}:`, err);
    });
  } catch (err) {
    console.error('Failed to start server:', err);
  }
};