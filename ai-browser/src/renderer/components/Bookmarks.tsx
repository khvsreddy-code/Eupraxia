import React from 'react';
import './Sidebar.css';

interface Bookmark {
    id: string;
    title: string;
    url: string;
    favicon?: string;
}

interface BookmarksProps {
    bookmarks: Bookmark[];
    onBookmarkClick?: (url: string) => void;
    onClose?: () => void;
    onDelete?: (id: string) => void;
}

export const Bookmarks: React.FC<BookmarksProps> = ({ bookmarks = [], onBookmarkClick, onClose, onDelete }) => {
    return (
        <div className="bookmarks-drawer">
            <div className="bookmarks-header">
                <strong>Bookmarks</strong>
                <button className="collapse-button" onClick={onClose}>✕</button>
            </div>
            <div className="bookmarks-list">
                {bookmarks.length === 0 ? (
                    <div className="empty-message">No bookmarks</div>
                ) : (
                    bookmarks.map(b => (
                        <div key={b.id} className="bookmark-item">
                            <button className="sidebar-button bookmark-button" onClick={() => onBookmarkClick && onBookmarkClick(b.url)}>
                                <span className="icon">🔖</span>
                                <span className="label">{b.title}</span>
                            </button>
                            {onDelete && (
                                <button className="collapse-button" onClick={() => onDelete(b.id)} title="Delete">🗑️</button>
                            )}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};
