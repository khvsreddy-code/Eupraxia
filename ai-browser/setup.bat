@echo off
echo Installing dependencies...
call npm install

echo Installing development dependencies...
call npm install -D electron typescript @types/react @types/react-dom @types/express @types/node
call npm install -D @electron-forge/cli @electron-forge/maker-squirrel @electron-forge/maker-zip @electron-forge/maker-deb @electron-forge/maker-rpm
call npm install -D @electron-forge/plugin-webpack webpack webpack-cli ts-loader css-loader style-loader
call npm install -D @typescript-eslint/eslint-plugin @typescript-eslint/parser

echo Installing production dependencies...
call npm install express react react-dom three @types/three openai electron-squirrel-startup

echo Initializing Electron Forge...
call npx electron-forge import

echo Setup complete! Run 'npm run dev' to start the development server.