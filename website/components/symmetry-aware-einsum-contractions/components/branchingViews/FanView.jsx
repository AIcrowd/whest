import { explorerThemeColor, explorerThemeTint, getActiveExplorerThemeId } from '../../lib/explorerTheme.js';
import { notationColor } from '../../lib/notationSystem.js';

export default function FanView({ orbit, reachedReps }) {
  const themeId = getActiveExplorerThemeId();
  const orbitColor = notationColor('m_component');
  const repColor = notationColor('h_output');
  const edgeColor = explorerThemeColor(themeId, 'muted');

  const reps = reachedReps ?? [];
  const cy = 100;
  const repSpacing = reps.length > 1 ? 140 / (reps.length - 1) : 0;

  return (
    <svg viewBox="0 0 340 200" className="w-full" role="img" aria-label="orbit fan diagram">
      <circle cx="60" cy={cy} r="22" fill={orbitColor} />
      <text x="60" y={cy + 5} textAnchor="middle" fill={explorerThemeColor(themeId, 'surface')} fontSize="13" fontWeight="600">O</text>
      <text x="60" y={cy + 45} textAnchor="middle" fontSize="11" fill={explorerThemeColor(themeId, 'muted')}>
        orbit of {orbit?.size ?? '—'}
      </text>
      {reps.map((rep, idx) => {
        const ry = reps.length === 1 ? cy : 30 + idx * repSpacing;
        return (
          <g key={idx}>
            <path
              d={`M 82 ${cy} Q 180 ${(cy + ry) / 2} 250 ${ry}`}
              stroke={edgeColor}
              strokeWidth="2.4"
              fill="none"
            />
            <circle cx="260" cy={ry} r="11" fill={repColor} />
            <text x="285" y={ry + 4} fontSize="11" fill={explorerThemeColor(themeId, 'body')}>
              {`Q${idx + 1} (×${rep.weight ?? 1})`}
            </text>
          </g>
        );
      })}
      <text x="60" y="20" textAnchor="middle" fontSize="11" fontWeight="600" fill={explorerThemeColor(themeId, 'body')}>product orbit</text>
      <text x="265" y="20" textAnchor="middle" fontSize="11" fontWeight="600" fill={explorerThemeColor(themeId, 'body')}>Y/H</text>
    </svg>
  );
}
