import { Fragment } from 'react';
import { tokenizePseudocodeLine } from '../engine/teachingModel.js';
import { notationColor } from '../lib/notationSystem.js';

/**
 * Editorial-light rendering of the symmetry-aware contraction pseudocode.
 *
 * Card layout (top to bottom):
 *   1. Slogan — the whole algorithm in four words:
 *      "Multiply Once → Accumulate Many"
 *   2. Pseudocode block — an uninterrupted Python fragment. Two inline
 *      "annotation rows" sit at the exact indent of the lines they describe,
 *      carrying a coloured bar marker:
 *         ┃ Step 1 · multiply once    (coral, indent 4, before base_val)
 *         ┃ Step 2 · accumulate many  (coral, indent 8, before R[out] +=)
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
    color: 'text-[#ef5a4c]',
  },
  acc: {
    kicker: 'Step 2',
    label: 'accumulate many',
    color: 'text-[#ef5a4c]',
  },
};

function normalizeExampleForPseudocode(example) {
  if (!example) return null;
  if (Array.isArray(example.subscripts) && Array.isArray(example.operandNames)) {
    return {
      subscripts: example.subscripts,
      output: example.output ?? '',
      operandNames: example.operandNames,
    };
  }
  if (example.expression?.subscripts && example.expression?.operandNames) {
    return {
      subscripts: example.expression.subscripts.split(',').map((part) => part.trim()),
      output: example.expression.output ?? '',
      operandNames: example.expression.operandNames.split(',').map((part) => part.trim()),
    };
  }
  return null;
}

function formatSubscript(subscript) {
  return subscript.split('').join(',');
}

function buildBaseValueComment(example) {
  const normalized = normalizeExampleForPseudocode(example);
  if (!normalized) return '';
  const factors = normalized.subscripts.map((subscript, idx) => {
    const name = normalized.operandNames[idx] ?? `X${idx}`;
    return `${name}[${formatSubscript(subscript)}]`;
  });
  return factors.length > 0 ? `  # = ${factors.join(' * ')}` : '';
}

function buildReduceComment(example) {
  const normalized = normalizeExampleForPseudocode(example);
  if (!normalized?.output) return '';
  const allLabels = normalized.subscripts.join('').split('');
  const outputLabels = normalized.output.split('');
  const contractedLabels = [...new Set(allLabels.filter((label) => !outputLabels.includes(label)))];
  const contractedSuffix = contractedLabels.length > 0
    ? ` across all contracted indices ${contractedLabels.join(',')}`
    : '';
  return `  # R[${formatSubscript(normalized.output)}]${contractedSuffix}`;
}

function wrapCommentLines(commentText, maxChars = 84) {
  const normalized = commentText.trim().replace(/^#\s*/, '');
  if (!normalized) return [];

  const words = normalized.split(/\s+/);
  const lines = [];
  let current = '';

  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length <= maxChars || current.length === 0) {
      current = next;
      continue;
    }
    lines.push(current);
    current = word;
  }

  if (current) lines.push(current);
  return lines.map((line) => `# ${line}`);
}

// Each row has `kind: 'code' | 'annotation'`. Annotation rows sit inline at
// the same indent as the code line they describe and carry a coloured ┃ bar.
const BASE_LINES = [
  { id: 'comment-rep',  number: 1,  kind: 'code', code: '# RepSet     — unique input tuples to multiply.' },
  { id: 'comment-outs', number: 2,  kind: 'code', code: '# Outs(rep)  — unique output bins this product lands in.' },
  { id: 'comment-coef', number: 3,  kind: 'code', code: '# coeff      — how many orbit copies land on that bin.' },
  { id: 'blank-1',      number: 4,  kind: 'code', code: '' },
  { id: 'rep-loop',     number: 5,  kind: 'code', code: 'for rep in RepSet:' },
  { id: 'step1-ann',    number: 6,  kind: 'annotation', step: 'mult', indent: 4 },
  { id: 'base-val',     number: 7,  kind: 'code', code: '    base_val = product_at(rep)' },
  { id: 'out-loop',     number: 8,  kind: 'code', code: '    for out in Outs(rep):' },
  { id: 'step2-ann',    number: 9,  kind: 'annotation', step: 'acc', indent: 8 },
  { id: 'reduce',       number: 10, kind: 'code', code: '        R[out] += coeff(rep, out) * base_val' },
];

function buildLines(example) {
  const baseValueComment = buildBaseValueComment(example);
  const reduceComment = buildReduceComment(example);
  return BASE_LINES.map((line) => {
    if (line.id === 'base-val') {
      return { ...line, code: `    base_val = product_at(rep)${baseValueComment}` };
    }
    if (line.id === 'reduce') {
      return { ...line, code: `        R[out] += coeff(rep, out) * base_val${reduceComment}` };
    }
    return line;
  });
}

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
  const inlineCommentMatch = line.code.match(/(\s+#.*)$/);
  const codePrefix = inlineCommentMatch ? line.code.slice(0, -inlineCommentMatch[1].length) : line.code;
  const commentSuffix = inlineCommentMatch ? inlineCommentMatch[1] : '';
  const leadingWhitespace = codePrefix.match(/^\s*/)?.[0] ?? '';
  const tokens = tokenizePseudocodeLine(codePrefix);
  const wrappedCommentLines = commentSuffix ? wrapCommentLines(commentSuffix) : [];
  return (
    <Fragment>
      <span className="select-none pr-3 text-right text-xs text-stone-400">
        {line.number}
      </span>
      <code className="whitespace-pre border-l border-stone-200/80 pl-4 text-stone-800">
        {tokens.length === 0 ? (
          <span>&nbsp;</span>
        ) : (
          <>
            {tokens.map((token, idx) => (
              <span key={idx} className={TOKEN_CLASS[token.kind] ?? TOKEN_CLASS.plain}>
                {token.text}
              </span>
            ))}
            {wrappedCommentLines.length > 0 ? (
              <span
                className="mt-0.5 block max-w-[84ch] text-stone-500"
                style={{ paddingLeft: `${leadingWhitespace.length}ch` }}
              >
                {wrappedCommentLines.map((commentLine, idx) => (
                  <span key={idx} className="block italic">
                    {commentLine}
                  </span>
                ))}
              </span>
            ) : null}
          </>
        )}
      </code>
    </Fragment>
  );
}

export default function MentalFrameworkCode({ example }) {
  const lines = buildLines(example);
  return (
    <figure className="relative flex h-full flex-col overflow-hidden rounded-2xl border border-stone-200 bg-white shadow-sm">
      {/* Slogan — the whole algorithm in two sentences. */}
      <div className="border-b border-stone-200/60 px-5 py-4 md:px-6">
        <p className="text-[15px] leading-6 text-stone-800">
          <span className="font-semibold text-[#ef5a4c]">Multiply Once</span>
          <span className="px-2 text-stone-900">-&gt;</span>
          <span className="font-semibold text-[#ef5a4c]">Accumulate Many</span>
        </p>
      </div>

      <div
        className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 overflow-x-auto px-5 pb-6 pt-4 font-mono text-[13px] leading-7 md:px-6"
        role="region"
        aria-label="Symmetry-aware contraction pseudocode"
      >
        {lines.map((line) =>
          line.kind === 'annotation' ? (
            <AnnotationRow key={line.id} line={line} />
          ) : (
            <CodeRow key={line.id} line={line} />
          ),
        )}
      </div>

      {/* Counting convention — introduces μ and α for the first time on the
          page. Sits directly below the code grid; mt-auto anchors it to the
          bottom of the figure so that when the parent column stretches to
          match the left side's height, the band stays glued to the bottom. */}
      <div className="mt-auto border-t border-stone-200/70 bg-gray-50 px-5 py-4 md:px-6">
        <div className="font-sans text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">
          Counting convention
        </div>
        <p className="mt-1.5 text-[12.5px] leading-6 text-stone-700">
          The number of representative products is <strong className="font-semibold">M</strong>. Every{' '}
          <code className="rounded bg-stone-200/60 px-1 font-mono text-[12px] text-stone-800">
            base_val
          </code>{' '}
          above is shorthand for a multiplicative chain, so a k-operand product contributes k-1 binary multiplies. We write{' '}
          <strong className="font-semibold" style={{ color: notationColor('mu_total') }}>
            μ = (k-1)M
          </strong>{' '}
          for those multiplication-chain events. Every{' '}
          <code className="rounded bg-stone-200/60 px-1 font-mono text-[12px] text-stone-800">
            R[out] += coeff · base_val
          </code>{' '}
          is one direct output-bin update event, implemented as a fused multiply-add when the coefficient is not one; summing those events gives{' '}
          <strong className="font-semibold" style={{ color: notationColor('alpha_total') }}>
            α
          </strong>. This page reports μ + α and does not model memory traffic, BLAS kernels, or contraction-path rewrites.
        </p>
      </div>
    </figure>
  );
}
