import { explorerThemeColor, getActiveExplorerThemeId } from '../../lib/explorerTheme.js';
import { notationColor } from '../../lib/notationSystem.js';

export default function ArcsView({ orbit, reachedReps }) {
  const themeId = getActiveExplorerThemeId();
  const memberColor = notationColor('m_component');
  const repColor = notationColor('h_output');
  const edgeColor = explorerThemeColor(themeId, 'muted');

  const members = orbit?.members ?? [];
  const reps = reachedReps ?? [];

  return (
    <svg viewBox="0 0 340 200" className="w-full" role="img" aria-label="orbit members projecting to Y/H">
      <text x="60" y="22" textAnchor="middle" fontSize="11" fontWeight="600" fill={explorerThemeColor(themeId, 'body')}>X (orbit members)</text>
      <text x="280" y="22" textAnchor="middle" fontSize="11" fontWeight="600" fill={explorerThemeColor(themeId, 'body')}>Y/H</text>
      <rect x="20" y="40" width="80" height="120" fill="none" stroke={explorerThemeColor(themeId, 'border')} />
      <rect x="240" y="40" width="80" height="120" fill="none" stroke={explorerThemeColor(themeId, 'border')} />
      {members.map((member, idx) => {
        const mx = 30 + (idx % 2) * 40;
        const my = 50 + Math.floor(idx / 2) * 30;
        const repIdx = member.repIndex ?? 0;
        const ry = reps.length === 1 ? 100 : 50 + repIdx * (100 / Math.max(reps.length - 1, 1));
        return (
          <g key={idx}>
            <path d={`M ${mx + 8} ${my} Q 170 ${(my + ry) / 2} 250 ${ry}`} stroke={edgeColor} strokeWidth="1.4" fill="none" />
            <circle cx={mx} cy={my} r="5" fill={memberColor} />
          </g>
        );
      })}
      {reps.map((rep, idx) => {
        const ry = reps.length === 1 ? 100 : 50 + idx * (100 / Math.max(reps.length - 1, 1));
        return <circle key={`rep-${idx}`} cx="260" cy={ry} r="6" fill={repColor} />;
      })}
    </svg>
  );
}
