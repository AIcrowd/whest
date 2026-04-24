import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

// Flopscope Design System: `Flopscope.` with a permanent coral period.
// The dot is the brand glyph and carries the identity on its own — the
// brush-ink logo at /logo.png reads too small at nav scale, so the nav
// anchor is the wordmark only.
function Wordmark() {
  return (
    <span className="flopscope-wordmark text-[22px]" aria-label="Flopscope.">
      Flopscope<span className="flopscope-wordmark__dot">.</span>
    </span>
  );
}

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: <Wordmark />,
    },
    githubUrl: 'https://github.com/AIcrowd/flopscope',
  };
}
