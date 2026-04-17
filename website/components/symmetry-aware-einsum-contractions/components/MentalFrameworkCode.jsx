import { Fragment } from 'react';
import { tokenizePseudocodeLine } from '../engine/teachingModel.js';

/**
 * Editorial-light rendering of the symmetry-aware contraction pseudocode.
 *
 * Card layout (top to bottom):
 *   1. Title caption (Mental framework · contraction.py)
 *   2. Slogan — the whole algorithm as one sentence:
 *      "Compute each distinct product ONCE. Spread it to every output cell
 *      it contributes to."
 *   3. Pseudocode block — an uninterrupted Python fragment. Two inline
 *      "annotation rows" sit at the exact indent of the lines they describe,
 *      carrying a coloured bar marker:
 *         ┃ Step 1 · multiply once    (coral, indent 4, before base_val)
 *         ┃ Step 2 · accumulate many  (amber, indent 8, before R[out] +=)
 *
 * Key goal: the reader sees the "multiply once, accumulate many" shape
 * directly in the code, with step labels anchored at the exact spot they
 * explain — no visual discontinuity between annotation and the code it
 * annotates. The accumulation line shows `coeff(rep, out) * base_val` so
 * the weighting (how many orbit members land on a bin) is visible.
 */

const TOKEN_CLASS = {
  keyword: 'text-primary font-semibold',
  function: 'text-slate-700 font-semibold',
  number: 'text-amber-700',
  state: 'text-stone-900 font-semibold',
  comment: 'text-stone-500 italic',
  plain: 'text-stone-800',
};

const STEPS = {
  mult: {
    kicker: 'Step 1',
    label: 'multiply once',
    color: 'text-primary',
  },
  acc: {
    kicker: 'Step 2',
    label: 'accumulate many',
    color: 'text-amber-700',
  },
};

// Each row has `kind: 'code' | 'annotation'`. Annotation rows sit inline at
// the same indent as the code line they describe and carry a coloured ┃ bar.
const LINES = [
  { id: 'comment-rep',  number: 1,  kind: 'code', code: '# RepSet     — unique input tuples to multiply.' },
  { id: 'comment-outs', number: 2,  kind: 'code', code: '# Outs(rep)  — unique output bins this product lands in.' },
  { id: 'comment-coef', number: 3,  kind: 'code', code: '# coeff      — how many orbit copies land on that bin.' },
  { id: 'blank-1',      number: 4,  kind: 'code', code: '' },
  { id: 'rep-loop',     number: 5,  kind: 'code', code: 'for rep in RepSet:' },
  { id: 'step1-ann',    number: 6,  kind: 'annotation', step: 'mult', indent: 4 },
  { id: 'base-val',     number: 7,  kind: 'code', code: '    base_val = product_at(rep)' },
  { id: 'blank-2',      number: 8,  kind: 'code', code: '' },
  { id: 'out-loop',     number: 9,  kind: 'code', code: '    for out in Outs(rep):' },
  { id: 'step2-ann',    number: 10, kind: 'annotation', step: 'acc', indent: 8 },
  { id: 'reduce',       number: 11, kind: 'code', code: '        R[out] += coeff(rep, out) * base_val' },
];

function AnnotationRow({ line }) {
  const step = STEPS[line.step];
  return (
    <Fragment>
      <span className="select-none pr-3 text-right text-xs text-stone-400">
        {line.number}
      </span>
      <code className="whitespace-pre border-l border-stone-200/80 pl-4">
        {/* indent whitespace preserved via whitespace-pre */}
        <span>{' '.repeat(line.indent)}</span>
        <span className={`${step.color} font-bold`} aria-hidden="true">┃ </span>
        <span className={`${step.color} font-semibold not-italic`}>
          {step.kicker} · {step.label}
        </span>
      </code>
    </Fragment>
  );
}

function CodeRow({ line }) {
  const tokens = tokenizePseudocodeLine(line.code);
  return (
    <Fragment>
      <span className="select-none pr-3 text-right text-xs text-stone-400">
        {line.number}
      </span>
      <code className="whitespace-pre border-l border-stone-200/80 pl-4 text-stone-800">
        {tokens.length === 0 ? (
          <span>&nbsp;</span>
        ) : (
          tokens.map((token, idx) => (
            <span key={idx} className={TOKEN_CLASS[token.kind] ?? TOKEN_CLASS.plain}>
              {token.text}
            </span>
          ))
        )}
      </code>
    </Fragment>
  );
}

export default function MentalFrameworkCode() {
  return (
    <figure className="relative overflow-hidden rounded-2xl border border-stone-200 bg-stone-50 shadow-sm">
      <figcaption className="flex items-baseline justify-between gap-3 border-b border-stone-200/70 px-5 py-3 md:px-6">
        <span className="font-heading text-xs font-semibold uppercase tracking-[0.16em] text-stone-500">
          Mental framework
        </span>
        <span className="font-mono text-xs text-stone-400">contraction.py</span>
      </figcaption>

      {/* Slogan — the whole algorithm in two sentences. */}
      <div className="border-b border-stone-200/60 px-5 py-4 md:px-6">
        <p className="text-[14px] leading-6 text-stone-800">
          <strong className="font-semibold text-stone-900">
            Compute each distinct product ONCE.
          </strong>{' '}
          Spread it to every output cell it contributes to.
        </p>
      </div>

      <div
        className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 overflow-x-auto px-5 pb-6 pt-4 font-mono text-[13px] leading-7 md:px-6"
        role="region"
        aria-label="Symmetry-aware contraction pseudocode"
      >
        {LINES.map((line) =>
          line.kind === 'annotation' ? (
            <AnnotationRow key={line.id} line={line} />
          ) : (
            <CodeRow key={line.id} line={line} />
          ),
        )}
      </div>

      {/* Counting convention — introduces μ and α for the first time on the
          page. Sits directly below the code grid; figure takes its natural
          height and the grid's items-start lets the left column flow
          independently. */}
      <div className="border-t border-stone-200/70 bg-stone-100/60 px-5 py-4 md:px-6">
        <div className="font-heading text-[10px] font-semibold uppercase tracking-[0.18em] text-stone-500">
          Counting convention
        </div>
        <p className="mt-1.5 text-[12.5px] leading-6 text-stone-700">
          Every{' '}
          <code className="rounded bg-stone-200/60 px-1 font-mono text-[12px] text-stone-800">
            base_val
          </code>{' '}
          above is short-hand for a multiplicative chain: two-operand einsums
          cost one multiply per line, three-operand einsums cost two,
          four-operand einsums cost three. Summing across all lines gives what
          we call the{' '}
          <strong className="font-semibold text-primary">Multiplication Cost (μ)</strong>.
          Every{' '}
          <code className="rounded bg-stone-200/60 px-1 font-mono text-[12px] text-stone-800">
            R[out] += coeff · base_val
          </code>{' '}
          is one fused multiply-add; the sum of those is the{' '}
          <strong className="font-semibold text-amber-700">Accumulation Cost (α)</strong>.
        </p>
      </div>
    </figure>
  );
}
