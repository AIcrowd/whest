import OperationDocExample from './OperationDocExample';
import OperationDocHeader from './OperationDocHeader';
import OperationDocBody from './OperationDocBody';
import OperationDocOverlay from './OperationDocOverlay';
import OperationDocSection from './OperationDocSection';
import OperationDocSignature from './OperationDocSignature';
import type {ApiDocRecord} from './op-doc-types';

export default async function OperationDocPage({op}: {op: ApiDocRecord}) {
  // Layout follows numpy.org's reference page: H1 → signature →
  // brief summary → meta → extended description → Parameters → ...
  // The signature lives at the top so readers see the call shape before
  // the prose; the flopscope-specific overlay (area/type, cost,
  // flopscope-context) sits below the summary as supplemental metadata.
  return (
    <>
      <OperationDocSignature
        signature={op.signature}
        flopscopeSourceUrl={op.flopscope_source_url}
        upstreamSourceUrl={op.upstream_source_url}
        upstreamSourceLabel={op.upstream_source_label}
      />
      <OperationDocHeader
        summary={op.summary}
        provenanceLabel={op.provenance_label}
        provenanceUrl={op.provenance_url}
        provenanceRef={op.provenance_ref ?? op.numpy_ref}
      />
      <OperationDocOverlay op={op} />
      {op.body_sections && op.body_sections.length > 0 ? (
        <OperationDocBody
          sections={op.body_sections}
          headerSummary={op.summary}
        />
      ) : (
        <>
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
