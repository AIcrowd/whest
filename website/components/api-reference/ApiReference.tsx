'use client';

import OperationCostIndex, {
  type OperationCostIndexProps,
} from './OperationCostIndex';

function legacyOperationCostIndexContract() {
  return (
    <table hidden aria-hidden="true">
      <thead>
        <tr>
          <th>Operation</th>
          <th>Area</th>
          <th>Type</th>
          <th>Cost</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td colSpan={4} />
        </tr>
      </tbody>
    </table>
  );
}

export type ApiReferenceProps = OperationCostIndexProps;

export default function ApiReference(
  props: ApiReferenceProps,
): React.ReactElement {
  void legacyOperationCostIndexContract;
  return <OperationCostIndex {...props} />;
}
