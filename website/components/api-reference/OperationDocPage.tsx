import fs from 'node:fs/promises';
import path from 'node:path';

import OperationDocExample from './OperationDocExample';
import OperationDocHeader from './OperationDocHeader';
import OperationDocBody from './OperationDocBody';
import OperationDocOverlay from './OperationDocOverlay';
import OperationDocSection from './OperationDocSection';
import OperationDocSignature from './OperationDocSignature';
import type {OperationDocRecord} from './op-doc-types';

async function loadOpDoc(name: string): Promise<OperationDocRecord> {
  const slug = name.replaceAll('.', '-');
  const candidatePaths = [
    path.join(process.cwd(), '.generated', 'ops', `${slug}.json`),
    path.join(process.cwd(), 'website', '.generated', 'ops', `${slug}.json`),
  ];
  let filePath: string | null = null;

  for (const candidate of candidatePaths) {
    try {
      await fs.access(candidate);
      filePath = candidate;
      break;
    } catch {
      // Try the next path candidate.
    }
  }

  if (!filePath) {
    throw new Error(`Could not find generated doc for op: ${name}`);
  }

  const raw = await fs.readFile(filePath, 'utf8');
  return JSON.parse(raw) as OperationDocRecord;
}

export default async function OperationDocPage({name}: {name: string}) {
  const op = await loadOpDoc(name);

  return (
    <>
      <OperationDocHeader op={op} />
      <OperationDocOverlay op={op} />
      {op.body_sections && op.body_sections.length > 0 ? (
        <OperationDocBody
          sections={op.body_sections}
          headerSummary={op.summary}
          signature={op.signature}
          whestSourceUrl={op.whest_source_url}
          upstreamSourceUrl={op.upstream_source_url}
        />
      ) : (
        <>
          <OperationDocSignature
            signature={op.signature}
            whestSourceUrl={op.whest_source_url}
            upstreamSourceUrl={op.upstream_source_url}
          />
          <OperationDocSection title="Parameters" fields={op.parameters} />
          <OperationDocSection title="Returns" fields={op.returns} />
          <OperationDocSection title="See also" links={op.see_also} />
          <OperationDocSection title="Notes" paragraphs={op.notes_sections} />
        <OperationDocExample example={op.example} />
      </>
      )}
    </>
  );
}
