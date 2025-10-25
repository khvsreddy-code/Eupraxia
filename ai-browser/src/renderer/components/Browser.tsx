import React, { useState, useEffect } from 'react';
import { ChromeTabs } from './ChromeTabs';
import { Bookmarks } from './Bookmarks';
import { AddressBar } from './AddressBar';
import { TabContent } from './TabContent';
import { AIAssistant } from './AIAssistant';
import { Sidebar } from './Sidebar';
import { BrowserControls } from './BrowserControls';
import defaultLogo from '../assets/eupraxia-logo';
import '../styles/advanced.css';

interface Tab {
  id: string;
  title: string;
  url: string;
  favicon?: string;
  isLoading: boolean;
}

interface Bookmark {
  id: string;
  title: string;
  url: string;
  favicon?: string;
  folder?: string;
}

const Browser: React.FC = () => {
  const [tabs, setTabs] = useState<Tab[]>([{
    id: '1',
    title: 'New Tab',
    url: 'about:blank',
    isLoading: false
  }]);
  const [activeTabId, setActiveTabId] = useState('1');
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [isBookmarksVisible, setBookmarksVisible] = useState(false);
  const [isSidebarVisible, setSidebarVisible] = useState(false);
  const [isSidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isAIAssistantVisible, setAIAssistantVisible] = useState(true);
  const [aiInitialMessage, setAIInitialMessage] = useState<string | undefined>(undefined);

  const handleNewTab = () => {
    const newTab: Tab = {
      id: Date.now().toString(),
      title: 'New Tab',
      url: 'about:blank',
      isLoading: false
    };
    setTabs([...tabs, newTab]);
    setActiveTabId(newTab.id);
  };

  const handleCloseTab = (tabId: string) => {
    const newTabs = tabs.filter(tab => tab.id !== tabId);
    if (newTabs.length === 0) {
      handleNewTab();
    } else if (activeTabId === tabId) {
      setActiveTabId(newTabs[newTabs.length - 1].id);
    }
    setTabs(newTabs);
  };

  const openSpecialTab = (title: string, url: string) => {
    const newTab: Tab = {
      id: Date.now().toString(),
      title,
      url,
      isLoading: false,
    };
    setTabs(prev => [...prev, newTab]);
    setActiveTabId(newTab.id);
  };

  // Sidebar action handlers
  const handleSearchClick = () => {
    setAIInitialMessage('');
    setAIAssistantVisible(true);
  };

  const handleCreateClick = () => {
    openSpecialTab('Create', 'about:create');
  };

  const handleProjectsClick = () => {
    setBookmarksVisible(true);
  };

  const handlePersonasClick = () => {
    setAIInitialMessage('Switch persona');
    setAIAssistantVisible(true);
  };

  const handleImagesClick = () => openSpecialTab('Images', 'about:images');
  const handleAudioClick = () => openSpecialTab('Audio', 'about:audio');
  const handleCodeClick = () => openSpecialTab('Code', 'about:code');
  const handleSettingsClick = () => openSpecialTab('Settings', 'about:settings');

  const handleNavigate = (url: string) => {
    const updatedTabs = tabs.map(tab =>
      tab.id === activeTabId
        ? { ...tab, url, isLoading: true }
        : tab
    );
    setTabs(updatedTabs);
    // Implement actual navigation logic here
  };

  const handleAddBookmark = () => {
    const currentTab = tabs.find(tab => tab.id === activeTabId);
    if (currentTab) {
      const newBookmark: Bookmark = {
        id: Date.now().toString(),
        title: currentTab.title,
        url: currentTab.url,
        favicon: currentTab.favicon
      };
      setBookmarks([...bookmarks, newBookmark]);
    }
  };

  return (
    <div className="browser-container">
      <div className="browser-chrome">
        <BrowserControls
          onNewTab={handleNewTab}
          onToggleBookmarks={() => setBookmarksVisible(!isBookmarksVisible)}
          onToggleSidebar={() => setSidebarVisible(!isSidebarVisible)}
          onToggleAI={() => setAIAssistantVisible(!isAIAssistantVisible)}
        />
        <ChromeTabs
          tabs={tabs}
          activeTabId={activeTabId}
          onTabSelect={setActiveTabId}
          onTabClose={handleCloseTab}
          onNewTab={handleNewTab}
        />
        <AddressBar
          currentUrl={tabs.find(tab => tab.id === activeTabId)?.url || ''}
          onNavigate={handleNavigate}
          onAddBookmark={handleAddBookmark}
        />
      </div>

      <div className="browser-content">
        {isSidebarVisible && (
          <Sidebar
            isCollapsed={isSidebarCollapsed}
            onCollapse={() => setSidebarCollapsed(s => !s)}
            logoSrc={defaultLogo}
            bookmarks={bookmarks}
            onBookmarkClick={(url: string) => handleNavigate(url)}
            onClose={() => setSidebarVisible(false)}
            onSearch={handleSearchClick}
            onCreate={handleCreateClick}
            onProjects={handleProjectsClick}
            onPersonas={handlePersonasClick}
            onOpenImages={handleImagesClick}
            onOpenAudio={handleAudioClick}
            onOpenCode={handleCodeClick}
            onSettings={handleSettingsClick}
          />
        )}

        <div className="main-content">
          {tabs.map(tab => (
            <TabContent
              key={tab.id}
              tab={tab}
              isActive={tab.id === activeTabId}
              onTitleChange={(title) => {
                const updatedTabs = tabs.map(t =>
                  t.id === tab.id ? { ...t, title } : t
                );
                setTabs(updatedTabs);
              }}
              onFaviconChange={(favicon) => {
                const updatedTabs = tabs.map(t =>
                  t.id === tab.id ? { ...t, favicon } : t
                );
                setTabs(updatedTabs);
              }}
            />
          ))}
        </div>

        {isAIAssistantVisible && (
          <AIAssistant
            initialMessage={aiInitialMessage}
            onClose={() => setAIAssistantVisible(false)}
            currentUrl={tabs.find(tab => tab.id === activeTabId)?.url || ''}
          />
        )}
      </div>

      {isBookmarksVisible && (
        <Bookmarks
          bookmarks={bookmarks}
          onBookmarkClick={(url) => handleNavigate(url)}
          onClose={() => setBookmarksVisible(false)}
          onDelete={(id) => {
            setBookmarks(bookmarks.filter(b => b.id !== id));
          }}
        />
      )}
    </div>
  );
};

export default Browser;