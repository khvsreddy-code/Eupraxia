const config = {
    server: {
        port: 3001,
        host: 'localhost'
    },
    ai: {
        models: {
            base: process.env.OPENAI_MODEL || 'gpt-4-1106-preview',
            voice: 'local-voice-model',
            image: 'local-image-model',
            video: 'local-video-model'
        }
    },
    browser: {
        width: 1280,
        height: 800,
        minWidth: 800,
        minHeight: 600,
        frame: false,
        transparent: true,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: true,
            enableRemoteModule: false
        }
    },
    theme: {
        dark: true,
        accentColor: '#7c4dff',
        background: '#0a0a1a'
    }
}