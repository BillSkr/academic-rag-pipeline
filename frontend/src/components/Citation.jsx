import React from 'react';
import { BookOpen } from 'lucide-react';

const Citation = ({ text }) => {
  // Handle both string citations and object citations from the backend
  let displayText = '';
  let url = null;
  
  if (typeof text === 'object' && text !== null) {
    const meta = text.metadata || {};
    const title = meta.title || 'Unknown';
    const year = meta.year || 'N/A';
    displayText = `${title} (${year})`;
    
    // If doc_id is numeric, it's a PubMed ID
    if (meta.doc_id && /^\d+$/.test(meta.doc_id)) {
      url = `https://pubmed.ncbi.nlm.nih.gov/${meta.doc_id}/`;
    }
  } else {
    displayText = String(text || '');
  }

  const badge = (
    <span className="citation-badge" title={displayText}>
      <BookOpen size={12} style={{ marginRight: '6px' }} />
      {displayText.substring(0, 50)}{displayText.length > 50 ? '...' : ''}
    </span>
  );

  if (url) {
    return (
      <a href={url} target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>
        {badge}
      </a>
    );
  }

  return badge;
};

export default Citation;
