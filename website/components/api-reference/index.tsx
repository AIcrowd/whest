import OperationCostIndex from './OperationCostIndex';
import {getPublicApiOperations} from './operation-data';

export default function ApiReference() {
  const operations = getPublicApiOperations();
  return <OperationCostIndex operations={operations} />;
}
