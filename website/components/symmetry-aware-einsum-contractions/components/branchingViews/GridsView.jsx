import { explorerThemeColor, explorerThemeTint, getActiveExplorerThemeId, getExplorerThemeOperandPalette } from '../../lib/explorerTheme.js';

export default function GridsView({ orbit, allOrbits, reachedReps, hClasses }) {
  const themeId = getActiveExplorerThemeId();
  const palette = getExplorerThemeOperandPalette(themeId) ?? [];
  const xCells = orbit?.xGridCells ?? [];
  const yCells = orbit?.yGridCells ?? [];
  const xRows = orbit?.xGridRows ?? 3;
  const xCols = orbit?.xGridCols ?? 3;
  const yRows = orbit?.yGridRows ?? 2;
  const yCols = orbit?.yGridCols ?? 2;

  function colourForOrbit(idx) {
    return palette[idx % palette.length] ?? explorerThemeColor(themeId, 'muted');
  }
  function colourForHClass(idx) {
    return palette[idx % palette.length] ?? explorerThemeColor(themeId, 'muted');
  }

  const cellSize = 28;

  return (
    <svg viewBox="0 0 340 200" className="w-full" role="img" aria-label="assignment grid X coloured by orbit, Y/H grid coloured by H-class">
      <text x="80" y="20" textAnchor="middle" fontSize="11" fontWeight="600" fill={explorerThemeColor(themeId, 'body')}>X</text>
      {xCells.map((cell, idx) => (
        <rect
          key={`x-${idx}`}
          x={20 + cell.col * cellSize}
          y={30 + cell.row * cellSize}
          width={cellSize}
          height={cellSize}
          fill={colourForOrbit(cell.orbitIdx ?? 0)}
          stroke={explorerThemeColor(themeId, 'border')}
        />
      ))}
      <text x="170" y="100" textAnchor="middle" fontSize="13" fill={explorerThemeColor(themeId, 'body')}>→ π_V</text>
      <text x="265" y="20" textAnchor="middle" fontSize="11" fontWeight="600" fill={explorerThemeColor(themeId, 'body')}>Y/H</text>
      {yCells.map((cell, idx) => (
        <rect
          key={`y-${idx}`}
          x={215 + cell.col * cellSize}
          y={30 + cell.row * cellSize}
          width={cellSize}
          height={cellSize}
          fill={colourForHClass(cell.hClassIdx ?? 0)}
          stroke={explorerThemeColor(themeId, 'border')}
        />
      ))}
    </svg>
  );
}
