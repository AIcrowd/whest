import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const workflowPath = path.join(process.cwd(), '.github', 'workflows', 'docs-dispatch.yml');

test('flopscope docs dispatch workflow triggers only from main on docs contract changes', () => {
  const source = fs.readFileSync(workflowPath, 'utf8');

  assert.match(source, /push:\s*\n\s*branches:\s*\[main\]/);
  assert.match(source, /website\/content\/docs\/\*\*/);
  assert.match(source, /website\/docs-kit\/\*\*/);
  assert.match(source, /docs\/unified-docs-process\.md/);
});

test('flopscope docs dispatch workflow targets flopscope-docs via repository_dispatch', () => {
  const source = fs.readFileSync(workflowPath, 'utf8');

  assert.match(source, /repos\/AIcrowd\/flopscope-docs\/dispatches/);
  assert.match(source, /event_type": "source-updated"/);
  assert.match(source, /AICROWD_DOCS_DISPATCH_TOKEN/);
});
