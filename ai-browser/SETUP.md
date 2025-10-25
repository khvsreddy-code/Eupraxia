# Backup Setup Instructions

1. Node.js Installation
- Download: https://nodejs.org/ (LTS version)
- Install with "Add to PATH" enabled
- Restart computer after installation

2. Project Setup
```bash
# Create project directory
cd "c:\Users\harin\Desktop\vscode repos\eupraxiapl vsc"
mkdir ai-browser
cd ai-browser

# Initialize project
npm init -y

# Install core dependencies
npm install electron react react-dom three @types/three express openai
npm install -D typescript @types/react @types/react-dom @types/express @types/node
npm install -D @electron-forge/cli @electron-forge/maker-squirrel @electron-forge/maker-zip
npm install -D @electron-forge/plugin-webpack webpack webpack-cli ts-loader
npm install -D @mozilla/readability jsdom

# Initialize TypeScript
npx tsc --init
```

3. Environment Setup
Create `.env` file:
```env
OPENAI_API_KEY=your_key_here
```

4. Project Structure
```
ai-browser/
├── src/
│   ├── main/
│   │   └── index.ts         # Electron main process
│   ├── renderer/
│   │   ├── App.tsx         # React UI
│   │   ├── index.html      # HTML template
│   │   └── styles.css      # Styles
│   └── server/
│       └── index.ts        # Express backend
├── package.json
├── tsconfig.json
├── forge.config.js
└── .env
```

5. Features Implemented:
- WebGPU/Three.js graphics system
- Chat context management
- Content extraction
- Advanced visualization
- Shader management