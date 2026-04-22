import mainPreamble from './main/preamble.ts';
import mainSection1 from './main/section1.ts';
import mainSection2 from './main/section2.ts';
import mainSection3 from './main/section3.ts';
import mainSection4 from './main/section4.ts';
import mainSection5 from './main/section5.ts';
import appendixSection1 from './appendix/section1.ts';
import appendixSection2 from './appendix/section2.ts';
import appendixSection3 from './appendix/section3.ts';
import appendixSection4 from './appendix/section4.ts';
import appendixSection5 from './appendix/section5.ts';
import appendixSection6 from './appendix/section6.ts';

export * from './schema.ts';
export { default as renderProseBlocks } from './renderProseBlocks.jsx';
export { default as mainPreamble } from './main/preamble.ts';
export { default as mainSection1 } from './main/section1.ts';
export { default as mainSection2 } from './main/section2.ts';
export { default as mainSection3 } from './main/section3.ts';
export { default as mainSection4 } from './main/section4.ts';
export { default as mainSection5 } from './main/section5.ts';
export { default as appendixSection1 } from './appendix/section1.ts';
export { default as appendixSection2 } from './appendix/section2.ts';
export { default as appendixSection3 } from './appendix/section3.ts';
export { default as appendixSection4 } from './appendix/section4.ts';
export { default as appendixSection5 } from './appendix/section5.ts';
export { default as appendixSection6 } from './appendix/section6.ts';

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
