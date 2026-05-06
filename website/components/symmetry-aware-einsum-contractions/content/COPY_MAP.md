# Copy Map

Use this folder for string-first narrative edits. The content modules below are the approved copy surface for the symmetry explorer.

## Registry to consumer map

- `main/preamble.js` → `AlgorithmAtAGlance.jsx`
- `main/einsumGlance.js` → `explorerNarrative.js`
- `main/productSymmetry.js` → `explorerNarrative.js`
- `main/projection.js` → `explorerNarrative.js`
- `main/rowsCols.js` → `explorerNarrative.js`
- `main/componentFactorization.js` → `explorerNarrative.js`
- `main/certification.js` → `explorerNarrative.js`
- `main/countingShortcuts.js` → `explorerNarrative.js`
- `main/typedPartition.js` → `explorerNarrative.js`
- `main/assembleCost.js` → `explorerNarrative.js`
- `main/appendixTransition.js` → `explorerNarrative.js`
- `appendix/section1.ts` → `ExpressionLevelModal.jsx`
- `appendix/section2.ts` → `ExpressionLevelModal.jsx`
- `appendix/section3.ts` → `ExpressionLevelModal.jsx`
- `appendix/section4.ts` → `ExpressionLevelModal.jsx`
- `appendix/section5.ts` → `ExpressionLevelModal.jsx`
- `appendix/section6.ts` → `ExpressionLevelModal.jsx`
- `index.ts` → shared barrel and registry only
- `schema.ts` → copy schema only
- `renderProseBlocks.jsx` → shared prose renderer only

## Stays in JSX

Keep these in component code instead of moving them into the content registry:

- layout, spacing, card shells, and responsive structure
- `Latex`, links, inline code, strong/em emphasis, and other non-string tokens
- example-driven formulas, counts, badges, tables, and tooltips
