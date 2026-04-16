import { Inter, Geist } from 'next/font/google';
import { Provider } from '@/components/provider';
import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import 'katex/dist/katex.css';
import './global.css';
import { cn } from '@/lib/utils';
import { withBasePath } from '@/lib/base-path';

export const metadata: Metadata = {
  icons: {
    icon: withBasePath('/logo.png'),
    apple: withBasePath('/logo.png'),
  },
};

const geist = Geist({ subsets: ['latin'], variable: '--font-sans' });

const inter = Inter({
  subsets: ['latin'],
});

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={cn(inter.className, 'font-sans', geist.variable)} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
