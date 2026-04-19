import { useState } from 'react';
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '../lib/utils.js';

/**
 * Wraps any heading content in a click-to-copy permalink. A tiny `#` glyph
 * appears on hover; clicking copies `window.origin + pathname + #anchorId`
 * to the clipboard and updates `location.hash` so the native browser
 * "Copy link" and Cmd/Ctrl-click behaviours also work on right-click.
 *
 * Pass `labelText` to customise the aria-label and tooltip (e.g. "Section 4",
 * "Mental framework"); it only affects a11y and the mouse-over tooltip, not
 * the visible content.
 *
 * Designed to degrade cleanly: if `anchorId` is absent, just renders children.
 */
function AnchorLink({ anchorId, labelText, hashGlyphClassName, children }) {
  const [copied, setCopied] = useState(false);

  if (!anchorId) return <>{children}</>;

  const onClick = async (event) => {
    // Let Cmd/Ctrl/Shift-clicks and middle-clicks fall through to the browser's
    // default "open in new tab / window" handling; only hijack the bare click.
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
    event.preventDefault();
    const url = `${window.location.origin}${window.location.pathname}#${anchorId}`;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      // Clipboard can be blocked (non-https contexts, sandbox, etc.); just
      // update the hash so the URL bar carries the link the user can copy.
    }
    window.history.replaceState(null, '', `#${anchorId}`);
  };

  return (
    <a
      href={`#${anchorId}`}
      onClick={onClick}
      className="group inline-flex items-baseline gap-2 no-underline hover:no-underline focus-visible:outline-2 focus-visible:outline-offset-2"
      title={copied ? 'Link copied!' : `Copy link${labelText ? ` to ${labelText}` : ''}`}
      aria-label={`Copy link${labelText ? ` to ${labelText}` : ''}`}
    >
      {children}
      <span
        aria-hidden="true"
        className={cn(
          'inline-block font-mono leading-none transition-opacity',
          hashGlyphClassName ?? 'text-[13px] text-muted-foreground',
          copied ? 'opacity-100 text-emerald-600' : 'opacity-0 group-hover:opacity-60 group-focus-visible:opacity-100',
        )}
      >
        {copied ? '✓ copied' : '#'}
      </span>
    </a>
  );
}

/**
 * Distinctive section label for the five top-level sections of the explorer,
 * and (with an optional `label`) for appendix subsections that want the same
 * typography. Single source of truth so main-page and modal eyebrows cannot
 * drift stylistically.
 *
 * Visual grammar (Distill / academic-paper):
 *   - 'SECTION'  — small uppercase, letterspaced, muted gray (the caption role)
 *   - 'N'        — large serif italic, coral/primary, the eye-catching anchor
 *   - `label`    — optional small-caps descriptor rendered after the number,
 *                  separated by a muted bullet. Used by the appendix modal to
 *                  hang section labels ("Preliminaries", "Motivating example",
 *                  …) off the same eyebrow without introducing a second style.
 *
 * When an `anchorId` is supplied, the whole eyebrow becomes a click-to-copy
 * permalink via the shared AnchorLink helper (same UX used on subsection
 * headings so the affordance feels consistent page-wide).
 */
function SectionEyebrow({ n, anchorId, label = null }) {
  // `.sec-kicker` spec from design-system/preview/components.html:
  // 11px / 600 / 0.16em tracking / uppercase / coral.
  const captionClass =
    'text-[11px] font-semibold uppercase tracking-[0.16em] text-coral';
  const content = (
    <span className="inline-flex items-baseline gap-2">
      <span className={captionClass}>Section</span>
      <span
        className="font-serif italic leading-none text-primary"
        style={{ fontSize: '22px', fontWeight: 700 }}
      >
        {n}
      </span>
      {label && (
        <span className={`${captionClass} inline-flex items-baseline gap-2`}>
          <span aria-hidden>·</span>
          <span>{label}</span>
        </span>
      )}
    </span>
  );

  return (
    <AnchorLink anchorId={anchorId} labelText={`Section ${n}`}>
      {content}
    </AnchorLink>
  );
}

function ExplorerSectionCard({
  title,
  description,
  eyebrow,
  action,
  children,
  className,
  contentClassName,
  ...props
}) {
  const hasContent = Array.isArray(children)
    ? children.some((child) => child != null && child !== false && child !== '')
    : children != null && children !== false && children !== '';

  return (
    <Card className={cn('border-gray-200 shadow-none', className)} {...props}>
      {(title || description || eyebrow || action) && (
        <CardHeader className="border-b border-gray-100">
          <div className="flex items-start justify-between gap-2.5">
            <div className="space-y-2">
              {eyebrow ? (
                typeof eyebrow === 'string'
                  // Back-compat: string eyebrows (used by the Algorithm-at-
                  // a-Glance preamble and a few appendix cards) get the same
                  // coral kicker treatment as SectionEyebrow so the page
                  // eyebrow rhythm stays consistent.
                  ? <CardDescription className="text-[11px] font-semibold uppercase tracking-[0.16em] text-coral">{eyebrow}</CardDescription>
                  : <div>{eyebrow}</div>
              ) : null}
              {title ? <CardTitle className="text-lg leading-tight">{title}</CardTitle> : null}
              {description ? <CardDescription className="text-sm leading-6">{description}</CardDescription> : null}
            </div>
            {action ? <CardAction>{action}</CardAction> : null}
          </div>
        </CardHeader>
      )}
      {hasContent ? (
        <CardContent className={cn('pt-5', contentClassName)}>
          {children}
        </CardContent>
      ) : null}
    </Card>
  );
}

export { ExplorerSectionCard, SectionEyebrow, AnchorLink };
export default ExplorerSectionCard;
