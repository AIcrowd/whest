import { whestDark, whestLight } from '@/lib/shiki-themes';

const whestApiDark = {
  ...whestDark,
  name: 'whest-api-ink',
  colors: {
    ...whestDark.colors,
    'editor.background': '#232628',
    'editor.selectionBackground': '#34393B',
    'editorLineNumber.foreground': '#7B8284',
  },
};

export const apiCodeThemes = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  light: whestLight as any,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  dark: whestApiDark as any,
} as const;
