import opDocsJson from '../../.generated/op-docs.json';
import OperationDocExample from './OperationDocExample';
import OperationDocHeader from './OperationDocHeader';
import OperationDocNav from './OperationDocNav';
import OperationDocOverlay from './OperationDocOverlay';
import OperationDocSection from './OperationDocSection';
import type {OperationDocRecord} from './op-doc-types';

const opDocs = opDocsJson as Record<string, OperationDocRecord>;

export default function OperationDocPage({name}: {name: string}) {
  const op = opDocs[name];

  if (!op) {
    throw new Error(`Unknown operation doc: ${name}`);
  }

  return (
    <>
      <OperationDocHeader op={op} />
      <OperationDocOverlay op={op} />
      <OperationDocSection title="Parameters" fields={op.parameters} />
      <OperationDocSection title="Returns" fields={op.returns} />
      <OperationDocSection title="See also" links={op.see_also} />
      <OperationDocSection title="Notes" paragraphs={op.notes_sections} />
      <OperationDocExample example={op.example} />
      <OperationDocNav previous={op.previous} next={op.next} />
    </>
  );
}
