import { contextBridge, ipcRenderer } from 'electron';

// Expose IPC communication to renderer
contextBridge.exposeInMainWorld('electronAPI', {
  sendChatQuery: (query: string) => ipcRenderer.invoke('chat-query', query),
  fetchPage: (url: string) => ipcRenderer.invoke('fetch-page', url),
});