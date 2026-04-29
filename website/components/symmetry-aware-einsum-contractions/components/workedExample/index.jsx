import { explorerThemeColor, getActiveExplorerThemeId } from '../../lib/explorerTheme.js';

export const APPENDIX_PROSE_CLASS = 'font-serif text-[17px] leading-[1.75] text-gray-900';
export const APPENDIX_MONO_LEDGER_CLASS = 'font-mono text-[13px] leading-relaxed text-gray-900';

export function vStyle() {
  return {
    color: explorerThemeColor(getActiveExplorerThemeId(), 'hero'),
    fontWeight: 600,
  };
}

export function wStyle() {
  return {
    color: explorerThemeColor(getActiveExplorerThemeId(), 'summedSide'),
    fontWeight: 600,
  };
}

export function WorkedExampleIndex({ value, role = 'plain' }) {
  const style =
    role === 'v' ? vStyle() :
    role === 'w' ? wStyle() :
    undefined;
  return <span style={style}>{value}</span>;
}

export function WorkedExampleCoords({ coords = [], roles = [] }) {
  return coords.map((coord, idx) => (
    <span key={`${coord}-${idx}`}>
      {idx > 0 ? ',' : null}
      <WorkedExampleIndex value={coord} role={roles[idx] ?? 'plain'} />
    </span>
  ));
}

export function WorkedExampleTensorRef({ name, coords = [], roles = [] }) {
  if (!coords.length) return <>{name}</>;
  return (
    <>
      {name}[<WorkedExampleCoords coords={coords} roles={roles} />]
    </>
  );
}

export function WorkedExampleTensorProduct({ factors = [], scalarValues = null, total = null }) {
  return (
    <>
      {factors.map((factor, idx) => (
        <span key={`${factor.name}-${idx}`}>
          {idx > 0 ? ' · ' : null}
          <WorkedExampleTensorRef
            name={factor.name}
            coords={factor.coords}
            roles={factor.roles}
          />
        </span>
      ))}
      {Array.isArray(scalarValues) && scalarValues.length ? (
        <>
          {' = '}
          {scalarValues.map((value, idx) => (
            <span key={`${value}-${idx}`}>
              {idx > 0 ? ' · ' : null}
              {value}
            </span>
          ))}
          {total !== null ? (
            <>
              {' = '}
              <strong>{total}</strong>
            </>
          ) : null}
        </>
      ) : null}
    </>
  );
}

export function WorkedExampleDisplayEquation({
  outputCoords = [],
  outputRoles = [],
  sumCoords = [],
  sumRoles = [],
  factors = [],
}) {
  return (
    <div className={`pl-0 sm:pl-4 ${APPENDIX_MONO_LEDGER_CLASS}`}>
      <div>
        <WorkedExampleTensorRef name="R" coords={outputCoords} roles={outputRoles} />
        {' = '}
        {sumCoords.length ? (
          <>
            ∑
            <sub className="text-[0.72em]">
              <WorkedExampleCoords coords={sumCoords} roles={sumRoles} />
            </sub>
            {' '}
          </>
        ) : null}
        <WorkedExampleTensorProduct factors={factors} />
      </div>
    </div>
  );
}

export function WorkedExampleEquation({ assignment, numeric }) {
  return (
    <div className="space-y-1">
      <div className="text-gray-900">{assignment}</div>
      <div className="pl-[5.5ch] text-gray-700">{numeric}</div>
    </div>
  );
}

export function WorkedExampleEquationLedger({ children }) {
  return (
    <div className={APPENDIX_MONO_LEDGER_CLASS}>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

export function WorkedExampleNote({ tone = 'neutral', children }) {
  const contentClass =
    tone === 'success'
      ? APPENDIX_PROSE_CLASS.replace('text-gray-900', 'text-gray-800')
      : APPENDIX_PROSE_CLASS.replace('text-gray-900', 'text-gray-700');
  return (
    <div className={contentClass}>
      {children}
    </div>
  );
}
