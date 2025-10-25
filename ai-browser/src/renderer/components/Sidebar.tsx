import React from 'react';
import './Sidebar.css';
import defaultLogo from '../assets/eupraxia-logo';

interface Bookmark {
  id: string;
  title: string;
  url: string;
  favicon?: string;
  folder?: string;
}

interface SidebarProps {
  isCollapsed?: boolean;
  onCollapse?: () => void;
  bookmarks?: Bookmark[];
  onBookmarkClick?: (url: string) => void;
  onClose?: () => void;
  logoSrc?: string; // prefer a PNG path if provided
  onSearch?: () => void;
  onCreate?: () => void;
  onProjects?: () => void;
  onPersonas?: () => void;
  onOpenImages?: () => void;
  onOpenAudio?: () => void;
  onOpenCode?: () => void;
  onSettings?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isCollapsed = false,
  onCollapse,
  bookmarks = [],
  onBookmarkClick,
  onClose,
  logoSrc,
  onSearch,
  onCreate,
  onProjects,
  onPersonas,
  onOpenImages,
  onOpenAudio,
  onOpenCode,
  onSettings,
}) => {
  const logoToUse = logoSrc || defaultLogo;

  return (
    <nav className={`sidebar ${isCollapsed ? 'collapsed' : ''}`} aria-label="Primary sidebar">
      <div className="sidebar-header">
        <div className="logo-container">
          <img src={logoToUse} alt="Eupraxia" className="logo" />
        </div>
        <div className="header-controls">
          {onClose && (
            <button className="collapse-button" onClick={onClose} title="Close sidebar" aria-label="Close sidebar">
              ✕
            </button>
          )}
          <button className="collapse-button" onClick={onCollapse} title="Toggle collapse" aria-label="Toggle sidebar">
            {isCollapsed ? '→' : '←'}
          </button>
        </div>
      </div>

      <div className="sidebar-content">
        <div className="sidebar-section main-nav">
          <button
            className="sidebar-button"
            onClick={onSearch}
            title="Search"
            aria-label="Search"
            data-tooltip="Search"
          >
            <span className="icon">🔍</span>
            <span className="label">Search</span>
          </button>

          <button
            className="sidebar-button"
            onClick={onCreate}
            title="Create"
            aria-label="Create"
            data-tooltip="Create"
          >
            <span className="icon">✨</span>
            <span className="label">Create</span>
          </button>

          <button
            className="sidebar-button"
            onClick={onProjects}
            title="Projects"
            aria-label="Projects"
            data-tooltip="Projects"
          >
            <span className="icon">🎮</span>
            <span className="label">Projects</span>
          </button>

          <button
            className="sidebar-button"
            onClick={onPersonas}
            title="Personas"
            aria-label="Personas"
            data-tooltip="Personas"
          >
            <span className="icon">👤</span>
            <span className="label">Personas</span>
          </button>
        </div>

        {bookmarks.length > 0 && (
          <div className="sidebar-section bookmarks">
            <div className="section-title">Bookmarks</div>
            {bookmarks.map(b => (
              <button
                key={b.id}
                className="sidebar-button bookmark-button"
                onClick={() => onBookmarkClick && onBookmarkClick(b.url)}
                title={b.title}
                aria-label={`Open bookmark ${b.title}`}
              >
                <span className="icon">🔖</span>
                <span className="label">{b.title}</span>
              </button>
            ))}
          </div>
        )}

        <div className="sidebar-section tools">
          <button
            className="sidebar-button"
            onClick={onOpenImages}
            title="Images"
            aria-label="Images"
            data-tooltip="Images"
          >
            <span className="icon">🖼️</span>
            <span className="label">Images</span>
          </button>

          <button
            className="sidebar-button"
            onClick={onOpenAudio}
            title="Audio"
            aria-label="Audio"
            data-tooltip="Audio"
          >
            <span className="icon">🎵</span>
            <span className="label">Audio</span>
          </button>

          <button
            className="sidebar-button"
            onClick={onOpenCode}
            title="Code"
            aria-label="Code"
            data-tooltip="Code"
          >
            <span className="icon">📝</span>
            <span className="label">Code</span>
          </button>
        </div>

        <div className="sidebar-section settings">
          <button
            className="sidebar-button"
            onClick={onSettings}
            title="Settings"
            aria-label="Settings"
            data-tooltip="Settings"
          >
            <span className="icon">⚙️</span>
            <span className="label">Settings</span>
          </button>
        </div>
      </div>
    </nav>
  );
};