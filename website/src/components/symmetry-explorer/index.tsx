import React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

export default function SymmetryExplorerWrapper() {
  return (
    <BrowserOnly fallback={<div>Loading Symmetry Explorer...</div>}>
      {() => {
        const SymmetryExplorer = require('./SymmetryExplorer.jsx').default;
        return <SymmetryExplorer />;
      }}
    </BrowserOnly>
  );
}
