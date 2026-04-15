import {
  buildMentalModelLines,
  getFocusedLines,
  tokenizePseudocodeLine,
} from '../engine/teachingModel.js';

export default function PseudocodeRail({ activeStepId = null, selectedOrbitRow = null }) {
  const focusedLines = new Set(activeStepId ? getFocusedLines(activeStepId) : []);
  const lines = buildMentalModelLines(selectedOrbitRow);

  return (
    <div className="pseudocode-rail" aria-label="Algorithm pseudocode">
      <div className="pseudocode-rail-inner">
        <div className="pseudocode-editor">
          <div className="pseudocode-editor-header">
            <div className="pseudocode-editor-chrome" aria-hidden="true">
              <span className="pseudocode-editor-dot pseudocode-editor-dot-red" />
              <span className="pseudocode-editor-dot pseudocode-editor-dot-amber" />
              <span className="pseudocode-editor-dot pseudocode-editor-dot-green" />
            </div>
            <div className="pseudocode-editor-title">
              <span className="pseudocode-kicker">Mental Framework</span>
              <h2>cost_model.py</h2>
            </div>
            <span className="pseudocode-editor-lang">python</span>
          </div>

          <p className="pseudocode-editor-caption">
            Read this as the mental model for the later cost formulas: first count
            one symmetry-unique representative, then count every distinct output-bin update it causes.
          </p>

          <div className="pseudocode-editor-body">
            <ol className="pseudocode-code-list">
              {lines.map((line, idx) => {
                const isFocused = focusedLines.size > 0 && focusedLines.has(line.number);
                const prevFocused = idx > 0 && focusedLines.has(lines[idx - 1].number);
                const nextFocused =
                  idx < lines.length - 1 &&
                  focusedLines.has(lines[idx + 1].number);
                const segments = tokenizePseudocodeLine(line.code);

                return (
                  <li
                    key={line.id}
                    className={[
                      'pseudocode-code-line',
                      isFocused ? 'pseudocode-code-line-focused' : '',
                      isFocused && !prevFocused ? 'pseudocode-code-line-block-start' : '',
                      isFocused && !nextFocused ? 'pseudocode-code-line-block-end' : '',
                    ]
                      .filter(Boolean)
                      .join(' ')}
                    aria-current={isFocused ? 'step' : undefined}
                  >
                    <span className="pseudocode-code-gutter" aria-hidden="true">
                      {line.number}
                    </span>
                    <code className="pseudocode-code-text">
                      {segments.map((segment, segmentIdx) => (
                        <span
                          key={`${line.id}-${segmentIdx}`}
                          className={`pseudocode-token pseudocode-token-${segment.kind}`}
                        >
                          {segment.text}
                        </span>
                      ))}
                    </code>
                  </li>
                );
              })}
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
