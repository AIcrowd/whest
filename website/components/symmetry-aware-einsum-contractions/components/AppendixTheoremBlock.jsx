import { useState } from 'react';
import InlineMathText from './InlineMathText.jsx';

// V3.1 §46 Appendix Theorem Block.
//
// Two presentation modes co-exist on the same component so the appendix can be
// migrated case-by-case without breaking existing call sites:
//
//   1. Legacy (children-based) — pre-V3.1 callers pass `kind` / `lead` /
//      `children`; the block renders a single italicised paragraph the way
//      the original AppendixTheoremBlock did.
//   2. V3.1 4-field template — callers pass `condition`, `claim`, `reason`,
//      `mainPageShortcut`. Each is rendered on its own line with a
//      bold standardized label ("Condition.", "Claim.", "Reason.",
//      "Main-page shortcut."). The order is fixed so reviewers can scan
//      cases consistently across the appendix.
//
// In V3.1 mode the block also accepts:
//   - `anchorId`  — when set, the block wraps itself in a section element
//                   carrying that id, and renders a small "#" anchor button
//                   that copies "<origin><pathname>#<anchorId>" to the
//                   clipboard (same UX as ExplorerSectionCard's AnchorLink).
//   - `claim`     — also gets a small "Copy" button next to it that copies
//                   the raw formula text (the unrendered $...$ source) so
//                   readers can paste it into a paper or notebook without
//                   re-typesetting. The button has an aria-label so screen
//                   readers don't see a bare glyph.
//
// Backward compat: callers that still pass only `kind` + `children` (and
// optionally `lead`) keep getting the original italicised paragraph layout.
// The new fields are all optional; mixing them into a legacy call gracefully
// degrades to V3.1 mode for the new fields and renders the legacy paragraph
// alongside.
export default function AppendixTheoremBlock({
  kind,
  children,
  lead = null,
  condition = null,
  claim = null,
  reason = null,
  mainPageShortcut = null,
  anchorId = null,
}) {
  // Detect V3.1 mode purely from the new fields. Any one of them switches the
  // block into the labeled-template layout; missing fields are skipped.
  const hasV31Fields =
    condition != null ||
    claim != null ||
    reason != null ||
    mainPageShortcut != null;

  const headingPrefix = kind ? `${kind}.` : null;

  // Legacy paragraph (kind / lead / children) — preserved for callers that
  // have not yet migrated to the 4-field template.
  const legacyParagraph = children != null
    ? (
      <p className="font-serif text-[17px] leading-[1.85] text-gray-800">
        {headingPrefix ? (
          <span className="font-semibold text-gray-900">
            {headingPrefix}
          </span>
        ) : null}
        {lead ? (
          <span className="ml-1 font-semibold text-gray-900">
            <InlineMathText>{lead}</InlineMathText>
          </span>
        ) : null}
        {' '}
        <span className="italic">
          <InlineMathText>{children}</InlineMathText>
        </span>
      </p>
    )
    : null;

  // V3.1 labeled rows — only the fields the caller passes are rendered.
  // Each row has the same baseline typography as the legacy paragraph so
  // mixed layouts (kind heading + V3.1 rows on the same case) stay coherent.
  const v31Rows = hasV31Fields ? (
    <div className="space-y-1.5">
      {condition != null ? (
        <p className="font-serif text-[17px] leading-[1.85] text-gray-800">
          <span className="font-semibold text-gray-900">Condition.</span>{' '}
          <InlineMathText>{condition}</InlineMathText>
        </p>
      ) : null}
      {claim != null ? (
        <p className="font-serif text-[17px] leading-[1.85] text-gray-800">
          <span className="font-semibold text-gray-900">Claim.</span>{' '}
          <InlineMathText>{claim}</InlineMathText>
          {typeof claim === 'string' ? (
            <CopyFormulaButton text={claim} />
          ) : null}
        </p>
      ) : null}
      {reason != null ? (
        <p className="font-serif text-[17px] leading-[1.85] text-gray-800">
          <span className="font-semibold text-gray-900">Reason.</span>{' '}
          <InlineMathText>{reason}</InlineMathText>
        </p>
      ) : null}
      {mainPageShortcut != null ? (
        <p className="font-serif text-[17px] leading-[1.85] text-gray-800">
          <span className="font-semibold text-gray-900">Main-page shortcut.</span>{' '}
          <InlineMathText>{mainPageShortcut}</InlineMathText>
        </p>
      ) : null}
    </div>
  ) : null;

  // V3.1 mode without legacy children: render an optional bold "kind." line
  // followed by the labeled rows. This is the new canonical layout.
  // Legacy mode without V3.1 fields: just the original paragraph.
  // Mixed: render the legacy paragraph first (so old callers keep their
  // visuals), then the V3.1 rows below.
  const heading = hasV31Fields && !children && headingPrefix ? (
    <p className="font-serif text-[17px] font-semibold leading-[1.85] text-gray-900">
      {headingPrefix}
      {lead ? (
        <span className="ml-1">
          <InlineMathText>{lead}</InlineMathText>
        </span>
      ) : null}
    </p>
  ) : null;

  const body = (
    <>
      {heading}
      {legacyParagraph}
      {v31Rows}
    </>
  );

  // When an anchorId is provided, wrap in a <section> with that id and a
  // hover-revealed "#" link button. The link copies the full URL to the
  // clipboard so readers can share the case directly. Without anchorId the
  // block stays an unwrapped fragment, preserving the legacy behaviour
  // exactly (no DOM diff for migrated callers that don't opt in).
  if (anchorId) {
    return (
      <section id={anchorId} className="group/theorem relative">
        <AnchorLinkButton anchorId={anchorId} />
        {body}
      </section>
    );
  }
  return body;
}

// Small "#" link that copies window.origin + pathname + #anchorId. Mirrors the
// AnchorLink helper in ExplorerSectionCard so the anchor UX is consistent
// across appendix surfaces. Keyboard-focusable + aria-labelled so screen
// readers and keyboard users get the same affordance as mouse hover.
function AnchorLinkButton({ anchorId }) {
  const [copied, setCopied] = useState(false);

  const onClick = async (event) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
    event.preventDefault();
    if (typeof window !== 'undefined') {
      const url = `${window.location.origin}${window.location.pathname}#${anchorId}`;
      try {
        if (typeof navigator !== 'undefined' && navigator.clipboard) {
          await navigator.clipboard.writeText(url);
          setCopied(true);
          setTimeout(() => setCopied(false), 1400);
        }
      } catch {
        // Non-https / sandboxed contexts can block clipboard; fall back to
        // updating the hash so the URL bar carries the shareable link.
      }
      window.history.replaceState(null, '', `#${anchorId}`);
    }
  };

  return (
    <a
      href={`#${anchorId}`}
      onClick={onClick}
      aria-label={copied ? 'Link copied' : 'Copy link to this theorem block'}
      title={copied ? 'Link copied!' : 'Copy link to this theorem block'}
      className="absolute -left-5 top-0 inline-flex items-center font-mono text-[13px] leading-none text-muted-foreground opacity-0 transition-opacity hover:opacity-100 focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-offset-2 group-hover/theorem:opacity-60"
    >
      <span aria-hidden="true">{copied ? '✓' : '#'}</span>
    </a>
  );
}

// Small "Copy" button rendered inline next to the formula. Copies the raw
// claim text (the unrendered $...$ source) so the reader can paste it
// straight into a paper or notebook. Keyboard-focusable + aria-labelled.
function CopyFormulaButton({ text }) {
  const [copied, setCopied] = useState(false);

  const onClick = async (event) => {
    event.preventDefault();
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      try {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1400);
      } catch {
        // Clipboard write may be blocked; the button silently no-ops in that
        // case rather than throwing — the formula is still selectable.
      }
    }
  };

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={copied ? 'Formula copied' : 'Copy formula to clipboard'}
      title={copied ? 'Formula copied!' : 'Copy formula'}
      className="ml-2 inline-flex items-center rounded border border-gray-200 px-1.5 py-0.5 align-baseline font-sans text-[10px] font-semibold uppercase tracking-[0.08em] text-gray-500 transition-colors hover:border-gray-300 hover:text-gray-700 focus-visible:outline-2 focus-visible:outline-offset-2"
    >
      {copied ? '✓ copied' : 'copy'}
    </button>
  );
}
