import React from 'react';
import { BookOpen } from 'lucide-react';

const Citation = ({ text }) => {
  return (
    <span className="citation-badge" title={text}>
      <BookOpen size={12} style={{ marginRight: '6px' }} />
      {text.substring(0, 40)}{text.length > 40 ? '...' : ''}
    </span>
  );
};

export default Citation;
