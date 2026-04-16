import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import Image from 'next/image';
import { withBasePath } from '@/lib/base-path';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <>
          <Image src={withBasePath('/logo.png')} alt="whest" width={36} height={36} />
          <span className="font-semibold">whest</span>
        </>
      ),
    },
    githubUrl: 'https://github.com/AIcrowd/whest',
  };
}
