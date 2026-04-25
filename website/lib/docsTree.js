export const STANDALONE_SYMMETRY_AWARE_EINSUM_URL = '/symmetry-aware-einsum-contractions';

const LEGACY_DOCS_URL = '/docs/understanding/symmetry-explorer';
const TOOL_NAME = 'Symmetry Aware Einsum Contractions';
const UNDERSTANDING_FOLDER = 'Understanding Flopscope';

function createLaunchItem() {
  return {
    type: 'page',
    name: TOOL_NAME,
    url: STANDALONE_SYMMETRY_AWARE_EINSUM_URL,
    external: true,
  };
}

export function injectSymmetryAwareEinsumContractionsLink(tree) {
  return {
    ...tree,
    children: tree.children.map((node) => {
      if (node.type !== 'folder' || node.name !== UNDERSTANDING_FOLDER) return node;

      const nextChildren = [];
      let inserted = false;

      for (const child of node.children) {
        if (child.type === 'page' && child.url === LEGACY_DOCS_URL) {
          nextChildren.push(createLaunchItem());
          inserted = true;
          continue;
        }

        nextChildren.push(child);
      }

      if (!inserted) {
        nextChildren.push(createLaunchItem());
      }

      return { ...node, children: nextChildren };
    }),
  };
}
