import React from 'react';
import { Button } from '@/components/ui/button';

export default function ExplorerModal({ title, titleId, open, onClose, children, width = 'min(960px, 92vw)' }) {
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="max-h-[85vh] overflow-y-auto rounded-xl bg-white shadow-2xl"
        style={{ width }}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-4 border-b border-border/70 px-5 py-4">
          <div id={titleId} className="text-sm font-medium text-foreground">
            {title}
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            aria-label={`Close ${typeof title === 'string' ? title : 'modal'}`}
            onClick={onClose}
          >
            Close
          </Button>
        </div>
        <div className="px-5 pb-5 pt-5">{children}</div>
      </div>
    </div>
  );
}
