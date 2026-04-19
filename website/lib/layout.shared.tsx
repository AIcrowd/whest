import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import Image from 'next/image';
import { withBasePath } from '@/lib/base-path';

// Whest Design System: `Whest.` with a permanent coral period.
// The dot is not punctuation — it is the brand glyph, echoing the coral dot
// in the brush-ink primary mark at /logo.png.
function Wordmark() {
  return (
    <span className="flex items-center gap-2">
      <Image src={withBasePath('/logo.png')} alt="" width={24} height={24} aria-hidden />
      <span className="whest-wordmark text-[17px]" aria-label="Whest.">
        Whest<span className="whest-wordmark__dot">.</span>
      </span>
    </span>
  );
}

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: <Wordmark />,
    },
    githubUrl: 'https://github.com/AIcrowd/whest',
  };
}
