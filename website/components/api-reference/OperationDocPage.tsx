import OperationDocExample from './OperationDocExample';
import OperationDocHeader from './OperationDocHeader';
import OperationDocBody from './OperationDocBody';
import OperationDocOverlay from './OperationDocOverlay';
import OperationDocSection from './OperationDocSection';
import OperationDocSignature from './OperationDocSignature';
import type {OperationDocRecord} from './op-doc-types';

export default async function OperationDocPage({op}: {op: OperationDocRecord}) {
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
