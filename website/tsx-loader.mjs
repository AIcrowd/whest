/**
 * Node ESM loader for .tsx/.ts files — used by the test suite.
 * Transforms TypeScript + JSX using the bundled esbuild.
 * Also resolves @/ path aliases (tsconfig paths: "@/*" → "./*").
 */
import { transform } from 'esbuild';
import { readFileSync } from 'fs';
import { fileURLToPath, pathToFileURL } from 'url';
import { resolve as pathResolve } from 'path';

const WEBSITE_ROOT = new URL('.', import.meta.url).pathname.replace(/\/$/, '');

export async function resolve(specifier, context, nextResolve) {
  // Resolve @/ path alias → website root
  if (specifier.startsWith('@/')) {
    const resolved = pathResolve(WEBSITE_ROOT, specifier.slice(2));
    for (const ext of ['', '.ts', '.tsx', '/index.ts', '/index.tsx', '/index.mjs', '/index.js']) {
      try {
        return await nextResolve(pathToFileURL(resolved + ext).href, context);
      } catch {
        // try next extension
      }
    }
  }

  // Help resolve bare imports when the parent is a .tsx/.ts file
  if (
    context.parentURL &&
    (context.parentURL.endsWith('.tsx') || context.parentURL.endsWith('.ts')) &&
    !specifier.startsWith('.') &&
    !specifier.startsWith('/')
  ) {
    return nextResolve(specifier, {
      ...context,
      parentURL: pathToFileURL(WEBSITE_ROOT + '/package.json').href,
    });
  }

  return nextResolve(specifier, context);
}

export async function load(url, context, nextLoad) {
  if (
    (url.endsWith('.tsx') || url.endsWith('.ts')) &&
    !url.includes('node_modules')
  ) {
    const filePath = fileURLToPath(url);
    const source = readFileSync(filePath, 'utf8');
    const result = await transform(source, {
      loader: url.endsWith('.tsx') ? 'tsx' : 'ts',
      format: 'esm',
      jsx: 'automatic',
      jsxImportSource: 'react',
    });
    return {
      format: 'module',
      source: result.code,
      shortCircuit: true,
    };
  }
  return nextLoad(url, context);
}
