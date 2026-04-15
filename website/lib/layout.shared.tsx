import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import Image from 'next/image';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <>
          <Image src="/whest/logo.png" alt="whest" width={28} height={28} />
          <span className="font-semibold">whest</span>
        </>
      ),
    },
    githubUrl: 'https://github.com/AIcrowd/whest',
  };
}
