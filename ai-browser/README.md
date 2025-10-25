# AI Browser

A Chromium-based AI browser with advanced graphics capabilities, powered by GPT-4 and WebGPU.

## Features

- Chromium-based browser interface
- Real-time AI chat with GPT-4
- Advanced 3D visualizations using WebGPU/Three.js
- Web content analysis and summarization
- Hardware-accelerated graphics
- Privacy-focused design

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Windows 10/11 with WebGPU-capable GPU

### Installation

1. Install dependencies:
```bash
# Run the setup script
setup.bat
```

2. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
```

3. Start the development server:
```bash
npm run dev
```

## Development

- `npm run dev` - Start the development server with hot reload
- `npm run build` - Build the application
- `npm run make` - Package the application for distribution

## Architecture

- Frontend: Electron, React, TypeScript
- Graphics: Three.js, WebGPU
- Backend: Express, OpenAI API
- Packaging: Electron Forge

## License

MIT