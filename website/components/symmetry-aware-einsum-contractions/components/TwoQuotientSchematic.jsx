/**
 * TwoQuotientSchematic — §4 Rows and Columns
 *
 * Visualizes the two quotients that define rows (O) and columns (Q):
 *
 *   X ──quotient by G_pt──▶  X/G_pt      (rows O)
 *   Y ──quotient by H──────▶  Y/H         (columns Q)
 *   X ──π_V──────────────────▶  Y
 *
 * Three example presets (Cross S2 / Bilinear trace / Triple outer).
 * Three hover interactions (G_pt → X/G_pt highlight, H → Y/H highlight,
 * π_V → fade summed labels in X box).
 *
 * All colours come from design-system tokens. No raw hex.
 * Accessibility: keyboard-navigable toggle buttons, ARIA labels on interactive
 * labels, prefers-reduced-motion respected.
 */

import { useState, useEffect } from 'react';

/* ─────────────────────────────────────────────────────────────────────────────
   Design-system token map (mirrors colors_and_type.css tier 1 + 3A)
   ───────────────────────────────────────────────────────────────────────────── */
const TOKEN = {
  coral:       'var(--coral)',        // #F0524D
  coralLight:  'var(--coral-light)',  // #FEF2F1
  gray900:     'var(--gray-900)',     // #292C2D
  gray600:     'var(--gray-600)',     // #5D5F60
  gray400:     'var(--gray-400)',     // #AAACAD
  gray200:     'var(--gray-200)',     // #D9DCDC
  gray100:     'var(--gray-100)',     // #F1F3F5
  gray50:      'var(--gray-50)',      // #F8F9F9
  white:       'var(--white)',        // #FFFFFF
  einV:        'var(--ein-v)',        // #4A7CFF  (free-label / visible)
  einW:        'var(--ein-w)',        // #64748B  (summed-label)
};

/* ─────────────────────────────────────────────────────────────────────────────
   Preset data — mock visual content for each of the three examples
   ───────────────────────────────────────────────────────────────────────────── */
const PRESETS = {
  crossS2: {
    id: 'crossS2',
    label: 'Cross S₂',
    description: 'H is trivial — every output assignment is its own representative.',
    /* Tuples shown in the X box (full assignment space).
       Last token is the "summed" label (j); others are visible. */
    xTuples: [
      { elems: ['0','0','0'], summedIdx: 1 },
      { elems: ['0','1','0'], summedIdx: 1 },
      { elems: ['1','0','0'], summedIdx: 1 },
      { elems: ['1','1','0'], summedIdx: 1 },
    ],
    /* Orbit representatives shown in X/G_pt */
    xQuotientTuples: [
      '(0,0,0)',
      '(0,1,0)',
      '(1,1,0)',
    ],
    /* Tuples in Y (output assignment space — only visible labels) */
    yTuples: ['(0,0)', '(0,1)', '(1,0)', '(1,1)'],
    /* Y/H — H trivial so Y/H = Y */
    yQuotientTuples: ['(0,0)', '(0,1)', '(1,0)', '(1,1)'],
    hTrivial: true,
    /* Projection arrows from X/G_pt → Y/H (one-to-many possible) */
    projections: [
      [0, [0, 1]],   // rep 0 → reps 0,1
      [1, [2, 3]],   // rep 1 → reps 2,3
      [2, [0]],      // rep 2 → rep 0
    ],
  },
  bilinearTrace: {
    id: 'bilinearTrace',
    label: 'Bilinear trace',
    description: 'H is nontrivial — output assignments sharing stored representatives collapse.',
    xTuples: [
      { elems: ['0','0','0','0'], summedIdx: 2 },
      { elems: ['0','1','0','1'], summedIdx: 2 },
      { elems: ['1','0','1','0'], summedIdx: 2 },
      { elems: ['1','1','1','1'], summedIdx: 2 },
    ],
    xQuotientTuples: [
      '(0,0,·,·)',
      '(0,1,·,·)',
      '(1,1,·,·)',
    ],
    yTuples: ['(0,0)', '(0,1)', '(1,0)', '(1,1)'],
    /* H nontrivial — (0,1) and (1,0) share a rep */
    yQuotientTuples: ['(0,0)', '(0,1)/(1,0)', '(1,1)'],
    hTrivial: false,
    projections: [
      [0, [0]],
      [1, [1]],
      [2, [2]],
    ],
  },
  tripleOuter: {
    id: 'tripleOuter',
    label: 'Triple outer',
    description: 'Rows and columns coincide — X/G_pt ≅ Y/H.',
    xTuples: [
      { elems: ['0','0','0'], summedIdx: 2 },
      { elems: ['0','1','0'], summedIdx: 2 },
      { elems: ['1','0','0'], summedIdx: 2 },
      { elems: ['1','1','0'], summedIdx: 2 },
    ],
    xQuotientTuples: [
      '(0,0)',
      '(0,1)',
      '(1,1)',
    ],
    yTuples: ['(0,0)', '(0,1)', '(1,0)', '(1,1)'],
    yQuotientTuples: ['(0,0)', '(0,1)', '(1,1)'],
    hTrivial: false,
    projections: [
      [0, [0]],
      [1, [1]],
      [2, [2]],
    ],
  },
};

const PRESET_ORDER = ['crossS2', 'bilinearTrace', 'tripleOuter'];

/* ─────────────────────────────────────────────────────────────────────────────
   SVG layout constants
   ───────────────────────────────────────────────────────────────────────────── */
const SVG_W = 620;
const SVG_H = 360;

// Box dimensions
const BOX_W = 148;
const BOX_H = 120;
const BOX_RX = 10;

// Horizontal positions (box left-edge x)
const LEFT_X = 24;
const RIGHT_X = SVG_W - BOX_W - 24;

// Vertical positions (box top-edge y)
const TOP_Y = 44;
const BOT_Y = 212;

// Arrow column centre (between left and right boxes)
const ARROW_CX = (LEFT_X + BOX_W + RIGHT_X) / 2;

// Vertical arrow x (left side of the left boxes)
const VERT_ARROW_X = LEFT_X - 28;

/* ─────────────────────────────────────────────────────────────────────────────
   Helper: detect reduced-motion preference
   ───────────────────────────────────────────────────────────────────────────── */
function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const handler = (e) => setReduced(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);
  return reduced;
}

/* ─────────────────────────────────────────────────────────────────────────────
   Sub-components
   ───────────────────────────────────────────────────────────────────────────── */

/** A rounded rect box with a header label and list of tuple chips. */
function Box({ x, y, w, h, label, tuples, highlight, dimSummed, reducedMotion }) {
  const borderColor = highlight ? TOKEN.coral : TOKEN.gray200;
  const borderWidth = highlight ? 2.5 : 1.5;
  const chipH = 18;
  const chipGap = 5;
  const contentY = y + 30;

  return (
    <g role="figure" aria-label={`Space ${label}`}>
      {/* Box border */}
      <rect
        x={x} y={y} width={w} height={h} rx={BOX_RX}
        fill={TOKEN.white}
        stroke={borderColor}
        strokeWidth={borderWidth}
        style={{
          transition: reducedMotion ? 'none' : 'stroke 0.18s ease, stroke-width 0.18s ease',
        }}
      />
      {/* Header label */}
      <text
        x={x + w / 2} y={y + 18}
        textAnchor="middle"
        fontSize={13}
        fontWeight={700}
        fontFamily="'IBM Plex Mono', monospace"
        fill={highlight ? TOKEN.coral : TOKEN.gray900}
        style={{ transition: reducedMotion ? 'none' : 'fill 0.18s ease' }}
      >
        {label}
      </text>

      {/* Tuple chips */}
      {tuples.slice(0, 5).map((tuple, i) => {
        const isSummed = typeof tuple === 'object' && dimSummed;
        const chipY = contentY + i * (chipH + chipGap);
        if (chipY + chipH > y + h - 4) return null;

        if (typeof tuple === 'object') {
          // Structured tuple with summed index — for X box
          const { elems, summedIdx } = tuple;
          const chipW = w - 16;
          const baseOpacity = (isSummed && reducedMotion) ? 0.35 : 1;
          return (
            <g key={i}>
              <rect
                x={x + 8} y={chipY}
                width={chipW} height={chipH} rx={4}
                fill={TOKEN.gray100} stroke={TOKEN.gray200} strokeWidth={0.5}
              />
              <text
                x={x + 8 + chipW / 2} y={chipY + chipH / 2 + 1}
                textAnchor="middle" dominantBaseline="middle"
                fontSize={10} fontFamily="'IBM Plex Mono', monospace"
                fill={TOKEN.gray600}
              >
                {'('}
                {elems.map((el, ei) => {
                  const isSummedEl = ei === summedIdx;
                  return (
                    <tspan
                      key={ei}
                      fill={isSummedEl ? TOKEN.einW : TOKEN.einV}
                      opacity={isSummedEl && dimSummed
                        ? (reducedMotion ? 0.25 : 0.22)
                        : 1}
                      style={{
                        transition: (reducedMotion || !isSummedEl)
                          ? 'none'
                          : 'opacity 0.25s ease',
                      }}
                      fontWeight={isSummedEl ? 400 : 500}
                    >
                      {ei > 0 ? ',' : ''}{el}
                    </tspan>
                  );
                })}
                {')'}
              </text>
            </g>
          );
        }

        // Plain string tuple
        const chipW = w - 16;
        return (
          <g key={i}>
            <rect
              x={x + 8} y={chipY}
              width={chipW} height={chipH} rx={4}
              fill={TOKEN.gray100} stroke={TOKEN.gray200} strokeWidth={0.5}
            />
            <text
              x={x + 8 + chipW / 2} y={chipY + chipH / 2 + 1}
              textAnchor="middle" dominantBaseline="middle"
              fontSize={10} fontFamily="'IBM Plex Mono', monospace"
              fill={TOKEN.gray600}
            >
              {tuple}
            </text>
          </g>
        );
      })}
    </g>
  );
}

/** Horizontal arrow with label above/below */
function HArrow({ fromX, fromY, toX, label, labelHover, onHoverStart, onHoverEnd, highlight, reducedMotion }) {
  const midX = (fromX + toX) / 2;
  const arrowY = fromY;
  const [focused, setFocused] = useState(false);
  const isActive = highlight;

  return (
    <g>
      {/* Shaft */}
      <line
        x1={fromX} y1={arrowY}
        x2={toX - 10} y2={arrowY}
        stroke={isActive ? TOKEN.coral : TOKEN.gray400}
        strokeWidth={isActive ? 2 : 1.5}
        style={{ transition: reducedMotion ? 'none' : 'stroke 0.18s ease' }}
      />
      {/* Arrowhead */}
      <polygon
        points={`${toX},${arrowY} ${toX - 9},${arrowY - 5} ${toX - 9},${arrowY + 5}`}
        fill={isActive ? TOKEN.coral : TOKEN.gray400}
        style={{ transition: reducedMotion ? 'none' : 'fill 0.18s ease' }}
      />
      {/* Interactive label */}
      <text
        x={midX} y={arrowY - 8}
        textAnchor="middle"
        fontSize={11}
        fontWeight={600}
        fontFamily="'Inter', sans-serif"
        fill={isActive ? TOKEN.coral : TOKEN.gray600}
        style={{
          cursor: 'pointer',
          transition: reducedMotion ? 'none' : 'fill 0.18s ease',
          outline: focused ? `2px solid ${TOKEN.einV}` : 'none',
        }}
        tabIndex={0}
        role="button"
        aria-label={`Hover to highlight: ${labelHover}`}
        aria-pressed={isActive}
        onMouseEnter={onHoverStart}
        onMouseLeave={onHoverEnd}
        onFocus={() => { setFocused(true); onHoverStart(); }}
        onBlur={() => { setFocused(false); onHoverEnd(); }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onHoverStart();
          }
        }}
      >
        quotient by {label}
      </text>
    </g>
  );
}

/** Vertical downward arrow on the left side */
function VArrow({ x, fromY, toY, label, onHoverStart, onHoverEnd, highlight, reducedMotion }) {
  const midY = (fromY + toY) / 2;
  const [focused, setFocused] = useState(false);
  const isActive = highlight;

  return (
    <g>
      {/* Shaft — dashed per spec ("dashed arrow is intentional") */}
      <line
        x1={x} y1={fromY}
        x2={x} y2={toY - 10}
        stroke={isActive ? TOKEN.coral : TOKEN.gray400}
        strokeWidth={isActive ? 2 : 1.5}
        strokeDasharray="5,4"
        style={{ transition: reducedMotion ? 'none' : 'stroke 0.18s ease' }}
      />
      {/* Arrowhead */}
      <polygon
        points={`${x},${toY} ${x - 5},${toY - 9} ${x + 5},${toY - 9}`}
        fill={isActive ? TOKEN.coral : TOKEN.gray400}
        style={{ transition: reducedMotion ? 'none' : 'fill 0.18s ease' }}
      />
      {/* Interactive label */}
      <text
        x={x - 14} y={midY}
        textAnchor="middle"
        fontSize={12}
        fontWeight={700}
        fontFamily="'IBM Plex Mono', monospace"
        fill={isActive ? TOKEN.coral : TOKEN.gray600}
        style={{
          cursor: 'pointer',
          transition: reducedMotion ? 'none' : 'fill 0.18s ease',
          outline: focused ? `2px solid ${TOKEN.einV}` : 'none',
        }}
        tabIndex={0}
        role="button"
        aria-label="Hover to see projection: pi_V from X to Y"
        aria-pressed={isActive}
        onMouseEnter={onHoverStart}
        onMouseLeave={onHoverEnd}
        onFocus={() => { setFocused(true); onHoverStart(); }}
        onBlur={() => { setFocused(false); onHoverEnd(); }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onHoverStart();
          }
        }}
      >
        {label}
      </text>
    </g>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Main component
   ───────────────────────────────────────────────────────────────────────────── */
export default function TwoQuotientSchematic() {
  const [presetId, setPresetId] = useState('crossS2');
  const [hoverTarget, setHoverTarget] = useState(null); // 'gpt' | 'h' | 'pi'
  const reducedMotion = usePrefersReducedMotion();

  const preset = PRESETS[presetId];

  /* Hover helpers */
  const startHover = (target) => setHoverTarget(target);
  const endHover = () => setHoverTarget(null);

  /* Derived highlight flags */
  const xQuotientHighlight = hoverTarget === 'gpt';
  const yQuotientHighlight = hoverTarget === 'h';
  const dimSummed = hoverTarget === 'pi';

  /* Arrow endpoints (centres of box edges) */
  const topArrowFromX = LEFT_X + BOX_W + 4;
  const topArrowToX   = RIGHT_X - 4;
  const topArrowY     = TOP_Y + BOX_H / 2;

  const botArrowFromX = LEFT_X + BOX_W + 4;
  const botArrowToX   = RIGHT_X - 4;
  const botArrowY     = BOT_Y + BOX_H / 2;

  const vertArrowFromY = TOP_Y + BOX_H + 4;
  const vertArrowToY   = BOT_Y - 4;

  /* Vertical arrow label — π_V rendered as text (plain) */
  const piLabel = 'πᵥ'; // π_V via unicode subscript V fallback

  return (
    <div
      className="two-quotient-schematic"
      style={{ fontFamily: "'Inter', sans-serif" }}
      aria-label="Two-quotient schematic: X to X-quotient-G_pt and Y to Y-quotient-H, connected by projection pi_V"
    >
      {/* Example toggle buttons */}
      <div
        role="group"
        aria-label="Select example preset"
        style={{
          display: 'flex',
          gap: '8px',
          marginBottom: '16px',
          flexWrap: 'wrap',
        }}
      >
        {PRESET_ORDER.map((pid) => {
          const active = pid === presetId;
          return (
            <button
              key={pid}
              type="button"
              aria-pressed={active}
              onClick={() => setPresetId(pid)}
              style={{
                padding: '5px 14px',
                fontSize: '13px',
                fontWeight: active ? 600 : 400,
                fontFamily: "'Inter', sans-serif",
                borderRadius: '6px',
                border: `1.5px solid ${active ? TOKEN.coral : TOKEN.gray200}`,
                background: active ? TOKEN.coralLight : TOKEN.white,
                color: active ? TOKEN.coral : TOKEN.gray600,
                cursor: 'pointer',
                transition: reducedMotion
                  ? 'none'
                  : 'background 0.15s ease, color 0.15s ease, border-color 0.15s ease',
              }}
            >
              {PRESETS[pid].label}
            </button>
          );
        })}
      </div>

      {/* Description text */}
      <p
        style={{
          fontSize: '13px',
          color: TOKEN.gray600,
          marginBottom: '12px',
          fontFamily: "'Inter', sans-serif",
          minHeight: '18px',
        }}
        aria-live="polite"
      >
        {preset.description}
      </p>

      {/* Interaction hint */}
      <p style={{ fontSize: '11px', color: TOKEN.gray400, marginBottom: '8px' }}>
        Hover or focus the arrow labels to highlight the corresponding quotient space.
      </p>

      {/* SVG schematic */}
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        style={{ width: '100%', maxWidth: SVG_W, height: 'auto', display: 'block' }}
        role="img"
        aria-label="Two-quotient schematic diagram"
      >
        {/* ── Row labels (column headers) ── */}
        <text x={LEFT_X + BOX_W / 2} y={32}
          textAnchor="middle" fontSize={10} fontWeight={600}
          fontFamily="'Inter', sans-serif" fill={TOKEN.gray400}
          letterSpacing="0.08em">
          FULL SPACE
        </text>
        <text x={RIGHT_X + BOX_W / 2} y={32}
          textAnchor="middle" fontSize={10} fontWeight={600}
          fontFamily="'Inter', sans-serif" fill={TOKEN.gray400}
          letterSpacing="0.08em">
          QUOTIENT
        </text>

        {/* ── Top row: X → X/G_pt ── */}
        <Box
          x={LEFT_X} y={TOP_Y}
          w={BOX_W} h={BOX_H}
          label="X"
          tuples={preset.xTuples}
          highlight={false}
          dimSummed={dimSummed}
          reducedMotion={reducedMotion}
        />
        <Box
          x={RIGHT_X} y={TOP_Y}
          w={BOX_W} h={BOX_H}
          label="X/G_pt"
          tuples={preset.xQuotientTuples}
          highlight={xQuotientHighlight}
          dimSummed={false}
          reducedMotion={reducedMotion}
        />
        <HArrow
          fromX={topArrowFromX}
          fromY={topArrowY}
          toX={topArrowToX}
          label="G_pt"
          labelHover="G_pt row quotient — highlights X/G_pt box"
          onHoverStart={() => startHover('gpt')}
          onHoverEnd={endHover}
          highlight={xQuotientHighlight}
          reducedMotion={reducedMotion}
        />

        {/* ── Vertical arrow: X → Y (π_V) ── */}
        <VArrow
          x={VERT_ARROW_X}
          fromY={vertArrowFromY}
          toY={vertArrowToY}
          label="π_V"
          onHoverStart={() => startHover('pi')}
          onHoverEnd={endHover}
          highlight={dimSummed}
          reducedMotion={reducedMotion}
        />

        {/* ── Bottom row: Y → Y/H ── */}
        <Box
          x={LEFT_X} y={BOT_Y}
          w={BOX_W} h={BOX_H}
          label="Y"
          tuples={preset.yTuples}
          highlight={false}
          dimSummed={false}
          reducedMotion={reducedMotion}
        />
        <Box
          x={RIGHT_X} y={BOT_Y}
          w={BOX_W} h={BOX_H}
          label={preset.hTrivial ? 'Y/H = Y' : 'Y/H'}
          tuples={preset.yQuotientTuples}
          highlight={yQuotientHighlight}
          dimSummed={false}
          reducedMotion={reducedMotion}
        />
        <HArrow
          fromX={botArrowFromX}
          fromY={botArrowY}
          toX={botArrowToX}
          label="H"
          labelHover="H column quotient — highlights Y/H box"
          onHoverStart={() => startHover('h')}
          onHoverEnd={endHover}
          highlight={yQuotientHighlight}
          reducedMotion={reducedMotion}
        />

        {/* ── Row role labels (right margin) ── */}
        <text
          x={RIGHT_X + BOX_W + 10} y={TOP_Y + BOX_H / 2 + 5}
          fontSize={11} fontWeight={500}
          fontFamily="'Inter', sans-serif"
          fill={TOKEN.einV}
        >
          rows O
        </text>
        <text
          x={RIGHT_X + BOX_W + 10} y={BOT_Y + BOX_H / 2 + 5}
          fontSize={11} fontWeight={500}
          fontFamily="'Inter', sans-serif"
          fill={TOKEN.einV}
        >
          cols Q
        </text>

        {/* ── Hover legend (bottom) ── */}
        <text x={SVG_W / 2} y={SVG_H - 10}
          textAnchor="middle" fontSize={10}
          fontFamily="'Inter', sans-serif"
          fill={TOKEN.gray400}
        >
          {hoverTarget === 'pi'
            ? 'Summed labels (dim colour) drop under projection π_V'
            : hoverTarget === 'gpt'
              ? 'X/Gₚₜ — product orbits (rows O)'
              : hoverTarget === 'h'
                ? 'Y/H — stored output representatives (columns Q)'
                : 'Hover a label to explore interactions'}
        </text>
      </svg>
    </div>
  );
}
