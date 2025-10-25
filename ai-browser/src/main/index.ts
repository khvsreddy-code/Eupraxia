import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
import { startServer, handleChatMessage } from '../server';

declare const MAIN_WINDOW_WEBPACK_ENTRY: string;
declare const MAIN_WINDOW_PRELOAD_WEBPACK_ENTRY: string;

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
  app.quit();
}

let mainWindow: BrowserWindow | null = null;

const createWindow = (): void => {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    frame: false,
    titleBarStyle: 'hidden',
    trafficLightPosition: { x: 20, y: 20 },
    backgroundColor: '#252525',
    webPreferences: {
      // Use a preload script and enable context isolation for secure IPC
      preload: MAIN_WINDOW_PRELOAD_WEBPACK_ENTRY,
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // and load the index.html of the app.
  mainWindow.loadURL(MAIN_WINDOW_WEBPACK_ENTRY);

  // Open the DevTools in development.
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }
};

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
app.on('ready', () => {
  // Start the Express server
  // startServer is async and will safely skip if the port is in use
  void startServer();
  createWindow();
});

// Quit when all windows are closed.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Handle IPC messages
ipcMain.handle('chat-query', async (_event, query: string) => {
  try {
    const result = await handleChatMessage(query);
    if ('response' in result) return { response: result.response };
    return { error: result.error };
  } catch (err) {
    console.error('IPC chat-query error:', err);
    return { error: 'Internal error' };
  }
});

// IPC handler for page fetching (used by renderer when server is unavailable)
ipcMain.handle('fetch-page', async (_event, url: string) => {
  try {
    if (!url || typeof url !== 'string') return { error: 'url required' };

    // Prefer global fetch if available, otherwise fall back to node-fetch
    let fetchFn: any = (globalThis as any).fetch;
    if (!fetchFn) {
      // dynamic import to avoid adding a hard dependency at bundle time
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      fetchFn = require('node-fetch');
    }

    const resp = await fetchFn(url);
    const text = await resp.text();
    return { content: text };
  } catch (err) {
    console.error('IPC fetch-page error:', err);
    return { error: 'Failed to fetch page' };
  }
});