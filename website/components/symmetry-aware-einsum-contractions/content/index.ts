import mainPreamble from './main/preamble.js';
import mainSection1 from './main/section1.js';
import mainSection2 from './main/section2.js';
import mainSection3 from './main/section3.js';
import mainSection4 from './main/section4.js';
import mainSection5 from './main/section5.js';
import appendixSection1 from './appendix/section1';
import appendixSection2 from './appendix/section2';
import appendixSection3 from './appendix/section3';
import appendixSection4 from './appendix/section4';
import appendixSection5 from './appendix/section5';
import appendixSection6 from './appendix/section6';

export * from './schema';
export { default as renderProseBlocks } from './renderProseBlocks.jsx';
export { default as mainPreamble } from './main/preamble.js';
export { default as mainSection1 } from './main/section1.js';
export { default as mainSection2 } from './main/section2.js';
export { default as mainSection3 } from './main/section3.js';
export { default as mainSection4 } from './main/section4.js';
export { default as mainSection5 } from './main/section5.js';
export { default as appendixSection1 } from './appendix/section1';
export { default as appendixSection2 } from './appendix/section2';
export { default as appendixSection3 } from './appendix/section3';
export { default as appendixSection4 } from './appendix/section4';
export { default as appendixSection5 } from './appendix/section5';
export { default as appendixSection6 } from './appendix/section6';

export const contentRegistry = {
  main: {
    preamble: mainPreamble,
    section1: mainSection1,
    section2: mainSection2,
    section3: mainSection3,
    section4: mainSection4,
    section5: mainSection5,
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
