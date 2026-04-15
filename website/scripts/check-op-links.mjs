import {readFile, readdir} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

import refsJson from '../.generated/op-refs.json' with {type: 'json'};

const refs = refsJson;

function stripCodeTicks(value) {
  return value.replace(/`/g, '');
}

export function resolveOpRef(name, manifest = refs) {
  const ref = manifest[name];
  if (!ref) {
    return undefined;
  }

  return {
    ...ref,
    label: stripCodeTicks(ref.label),
  };
}

function stripFrontmatter(source) {
  return source.replace(/^---\n[\s\S]*?\n---\n*/, '');
}

function stripFencedCode(source) {
  return source.replace(/```[\s\S]*?```/g, '');
}

export function scanMarkdownForBareOps(source, manifest = refs) {
  const stripped = stripFencedCode(stripFrontmatter(source));
  const seen = new Set();

  for (const match of stripped.matchAll(/\bwe\.[A-Za-z_][A-Za-z0-9_.]*/g)) {
    const opRef = match[0];
    const key = opRef.replace(/^we\./, '');
    if (manifest[key]) {
      seen.add(opRef);
    }
  }

  return [...seen];
}

async function* walkDocs(rootDir) {
  for (const entry of await readdir(rootDir, {withFileTypes: true})) {
    const fullPath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      yield* walkDocs(fullPath);
    } else if (entry.isFile() && fullPath.endsWith('.mdx')) {
      yield fullPath;
    }
  }
}

async function main() {
  const docsRoot = path.join(process.cwd(), 'content', 'docs');

  for await (const file of walkDocs(docsRoot)) {
    const source = await readFile(file, 'utf8');
    const problems = scanMarkdownForBareOps(source);
    if (problems.length > 0) {
      console.error(`${path.relative(process.cwd(), file)}: bare op refs -> ${problems.join(', ')}`);
      process.exitCode = 1;
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
