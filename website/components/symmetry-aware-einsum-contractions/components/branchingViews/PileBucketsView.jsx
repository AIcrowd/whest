import { explorerThemeColor, explorerThemeTint, getActiveExplorerThemeId } from '../../lib/explorerTheme.js';
import { notationColor } from '../../lib/notationSystem.js';

export default function PileBucketsView({ orbit, reachedReps }) {
  const themeId = getActiveExplorerThemeId();
  const memberColor = notationColor('m_component');
  const bucketBorder = notationColor('h_output');
  const bucketFill = explorerThemeTint(themeId, 'freeSide', 0.08);
  const memberCount = orbit?.members?.length ?? 0;

  const reps = reachedReps ?? [];
  const bucketHeight = 38;
  const gap = 8;

  return (
    <svg viewBox="0 0 340 200" className="w-full" role="img" aria-label="orbit members bucketed by H-class">
      <text x="60" y="20" textAnchor="middle" fontSize="11" fontWeight="600" fill={explorerThemeColor(themeId, 'body')}>orbit O</text>
      {Array.from({ length: memberCount }, (_, idx) => {
        const cx = 30 + (idx % 3) * 22;
        const cy = 40 + Math.floor(idx / 3) * 22;
        return <circle key={idx} cx={cx} cy={cy} r={6} fill={memberColor} />;
      })}
      <text x="170" y="100" textAnchor="middle" fontSize="13" fill={explorerThemeColor(themeId, 'body')}>π_V → bucket by H</text>
      <text x="265" y="20" textAnchor="middle" fontSize="11" fontWeight="600" fill={explorerThemeColor(themeId, 'body')}>Y/H buckets</text>
      {reps.map((rep, idx) => {
        const y = 30 + idx * (bucketHeight + gap);
        return (
          <g key={idx}>
            <rect x="220" y={y} width="100" height={bucketHeight} rx="6" fill={bucketFill} stroke={bucketBorder} />
            <text x="270" y={y + 14} textAnchor="middle" fontSize="10" fontWeight="600" fill={bucketBorder}>{`Q${idx + 1}`}</text>
            {Array.from({ length: rep.weight ?? 1 }, (_, j) => (
              <circle key={j} cx={235 + j * 18} cy={y + 28} r="5" fill={memberColor} />
            ))}
          </g>
        );
      })}
    </svg>
  );
}
