import React from 'react';
import './TabContent.css';

interface Tab {
    id: string;
    title: string;
    url: string;
    favicon?: string;
    isLoading: boolean;
}

interface TabContentProps {
    tab: Tab;
    isActive: boolean;
    onTitleChange?: (title: string) => void;
    onFaviconChange?: (favicon?: string) => void;
}

export const TabContent: React.FC<TabContentProps> = ({ tab, isActive }) => {
    return (
        <div className={`tab-panel ${isActive ? 'active' : 'hidden'}`}>
            {/* Simple placeholder: in your app this could be an <iframe> */}
            <div className="tab-panel-content">
                <div className="tab-title">{tab.title}</div>
                <div className="tab-url">{tab.url}</div>
            </div>
        </div>
    );
};
