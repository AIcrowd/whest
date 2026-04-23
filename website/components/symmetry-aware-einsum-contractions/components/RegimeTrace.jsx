import React from 'react';
import { REGIME_SPEC } from '../engine/regimeSpec.js';
import { SHAPE_SPEC } from '../engine/shapeSpec.js';

const LABEL_OVERRIDES = {
  allSummed: 'Direct Scalar Events',
};

function labelFor(step) {
  return (
    LABEL_OVERRIDES[step.regimeId] ||
    REGIME_SPEC[step.regimeId]?.label ||
    SHAPE_SPEC[step.regimeId]?.label ||
    step.regimeId
  );
}

function Row({ step, depth = 0, index = 0 }) {
  const fired = step.decision === 'fired';
  const paddingLeft = 8 + depth * 16;
  return (
    <>
      <div
        className={`animate-trace-in flex items-start gap-2 border-l-2 py-1.5 text-xs ${fired ? 'border-green-500 bg-green-50/60' : 'border-gray-200 bg-transparent'}`}
        style={{ paddingLeft, animationDelay: `${Math.min(depth * 30 + index * 30 + 30, 240)}ms` }}
      >
        <span
          className={`inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full font-bold ${fired ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-600'}`}
          aria-hidden="true"
        >
          {fired ? '✓' : '·'}
        </span>
        <div className="flex flex-col">
          <span className="font-medium text-gray-900">{labelFor(step)}</span>
          <span className="text-gray-500">{step.reason}</span>
          <span className="text-[10px] uppercase tracking-wider text-gray-400">{step.decision}</span>
        </div>
      </div>
      {(step.subSteps || []).map((sub, i) => (
        <Row key={`${step.regimeId}-sub-${i}`} step={sub} depth={depth + 1} />
      ))}
    </>
  );
}

export default function RegimeTrace({ trace }) {
  if (!trace || trace.length === 0) return null;
  return (
    <div className="rounded-md border border-gray-200 bg-white">
      {trace.map((step, i) => (
        <Row key={`${step.regimeId}-${i}`} step={step} index={i} />
      ))}
    </div>
  );
}
