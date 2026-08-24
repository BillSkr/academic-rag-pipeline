import React from 'react';
import { BookOpen } from 'lucide-react';

const Citation = ({ text }) => {
  // Handle both string citations and object citations from the backend
  let displayText = '';
  if (typeof text === 'object' && text !== null) {
    const meta = text.metadata || {};
    const title = meta.title || 'Unknown';
    const year = meta.year || 'N/A';
    displayText = `${title} (${year})`;
  } else {
    displayText = String(text || '');
  }

  return (
    <span className="citation-badge" title={displayText}>
      <BookOpen size={12} style={{ marginRight: '6px' }} />
      {displayText.substring(0, 50)}{displayText.length > 50 ? '...' : ''}
    </span>
  );
};

export default Citation;
