import mainPreamble from './main/preamble.js';
import mainEinsumGlance from './main/einsumGlance.js';
import mainProductSymmetry from './main/productSymmetry.js';
import mainProjection from './main/projection.js';
import mainRowsCols from './main/rowsCols.js';
import mainComponentFactorization from './main/componentFactorization.js';
import mainCertification from './main/certification.js';
import mainCountingShortcuts from './main/countingShortcuts.js';
import mainTypedPartition from './main/typedPartition.js';
import mainAssembleCost from './main/assembleCost.js';
import mainAppendixTransition from './main/appendixTransition.js';
import appendixSection1 from './appendix/section1';
import appendixSection2 from './appendix/section2';
import appendixSection3 from './appendix/section3';
import appendixSection4 from './appendix/section4';
import appendixSection5 from './appendix/section5';
import appendixSection6 from './appendix/section6';

export * from './schema';
export { default as renderProseBlocks } from './renderProseBlocks.jsx';
export { default as mainPreamble } from './main/preamble.js';
export { default as mainEinsumGlance } from './main/einsumGlance.js';
export { default as mainProductSymmetry } from './main/productSymmetry.js';
export { default as mainProjection } from './main/projection.js';
export { default as mainRowsCols } from './main/rowsCols.js';
export { default as mainComponentFactorization } from './main/componentFactorization.js';
export { default as mainCertification } from './main/certification.js';
export { default as mainCountingShortcuts } from './main/countingShortcuts.js';
export { default as mainTypedPartition } from './main/typedPartition.js';
export { default as mainAssembleCost } from './main/assembleCost.js';
export { default as mainAppendixTransition } from './main/appendixTransition.js';
export { default as appendixSection1 } from './appendix/section1';
export { default as appendixSection2 } from './appendix/section2';
export { default as appendixSection3 } from './appendix/section3';
export { default as appendixSection4 } from './appendix/section4';
export { default as appendixSection5 } from './appendix/section5';
export { default as appendixSection6 } from './appendix/section6';

export const contentRegistry = {
  main: {
    preamble: mainPreamble,
    einsumGlance: mainEinsumGlance,
    productSymmetry: mainProductSymmetry,
    projection: mainProjection,
    rowsCols: mainRowsCols,
    componentFactorization: mainComponentFactorization,
    certification: mainCertification,
    countingShortcuts: mainCountingShortcuts,
    typedPartition: mainTypedPartition,
    assembleCost: mainAssembleCost,
    appendixTransition: mainAppendixTransition,
  },
  appendix: {
    section1: appendixSection1,
    section2: appendixSection2,
    section3: appendixSection3,
    section4: appendixSection4,
    section5: appendixSection5,
    section6: appendixSection6,
  },
} as const;
