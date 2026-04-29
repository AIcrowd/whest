import { flopscopeDark, flopscopeLight } from '@/lib/shiki-themes';

const flopscopeApiDark = {
  ...flopscopeDark,
  name: 'flopscope-api-ink',
  colors: {
    ...flopscopeDark.colors,
    'editor.background': '#232628',
    'editor.selectionBackground': '#34393B',
    'editorLineNumber.foreground': '#7B8284',
  },
};

export const apiCodeThemes = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  light: flopscopeLight as any,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  dark: flopscopeApiDark as any,
} as const;
