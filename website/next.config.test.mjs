import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));
const configUrl = pathToFileURL(path.join(websiteRoot, 'next.config.mjs')).href;

async function loadConfig(nodeEnv) {
  const previousNodeEnv = process.env.NODE_ENV;
  const previousFumadocsMdx = process.env._FUMADOCS_MDX;
  process.env.NODE_ENV = nodeEnv;
  process.env._FUMADOCS_MDX = '1';

  try {
    const module = await import(
      `${configUrl}?node-env=${nodeEnv}&ts=${Date.now()}`
    );
    return module.default;
  } finally {
    if (previousNodeEnv === undefined) {
      delete process.env.NODE_ENV;
    } else {
      process.env.NODE_ENV = previousNodeEnv;
    }

    if (previousFumadocsMdx === undefined) {
      delete process.env._FUMADOCS_MDX;
    } else {
      process.env._FUMADOCS_MDX = previousFumadocsMdx;
    }
  }
}

test('development config serves docs from the root path', async () => {
  const config = await loadConfig('development');

  assert.equal(config.basePath, undefined);
  assert.equal(config.turbopack?.root, websiteRoot);
});

test('production config keeps the GitHub Pages base path', async () => {
  const config = await loadConfig('production');

  assert.equal(config.basePath, '/whest');
});
