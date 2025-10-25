import React from 'react';
import './ChromeTabs.css';

interface Tab {
  id: string;
  title: string;
  url: string;
  favicon?: string;
  isLoading: boolean;
}

interface ChromeTabsProps {
  tabs: Tab[];
  activeTabId: string;
  onTabSelect: (tabId: string) => void;
  onTabClose: (tabId: string) => void;
  onNewTab: () => void;
}

export const ChromeTabs: React.FC<ChromeTabsProps> = ({
  tabs,
  activeTabId,
  onTabSelect,
  onTabClose,
  onNewTab,
}) => {
  return (
    <div className="chrome-tabs">
      <div className="tabs-container">
        {tabs.map((tab) => (
          <div
            key={tab.id}
            className={`tab ${tab.id === activeTabId ? 'active' : ''}`}
            onClick={() => onTabSelect(tab.id)}
          >
            <div className="tab-background">
              <div className="tab-background-inner" />
            </div>
            <div className="tab-content">
              {tab.favicon && (
                <img
                  src={tab.favicon}
                  alt=""
                  className="tab-favicon"
                />
              )}
              {tab.isLoading ? (
                <div className="tab-loading-spinner" />
              ) : null}
              <span className="tab-title">{tab.title}</span>
              <button
                className="tab-close"
                onClick={(e) => {
                  e.stopPropagation();
                  onTabClose(tab.id);
                }}
              >
                ×
              </button>
            </div>
          </div>
        ))}
      </div>
      <button className="new-tab-button" onClick={onNewTab}>
        +
      </button>
    </div>
  );
};