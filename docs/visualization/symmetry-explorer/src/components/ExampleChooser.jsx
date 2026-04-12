export default function ExampleChooser({ examples, selected, onSelect, dimensionN, onDimensionChange }) {
  return (
    <div className="example-chooser">
      <div className="example-grid">
        {examples.map((ex, i) => (
          <button
            key={ex.id}
            className={`example-card ${selected === i ? 'active' : ''}`}
            style={{ '--accent': ex.color }}
            onClick={() => onSelect(i)}
          >
            <div className="example-name">{ex.name}</div>
            <code className="example-formula">{ex.formula}</code>
            <div className="example-group">{ex.expectedGroup}</div>
            <div className="example-desc">{ex.description}</div>
          </button>
        ))}
      </div>
      <div className="dimension-slider">
        <label>
          Dimension <strong>n = {dimensionN}</strong>
          <input
            type="range"
            min={2}
            max={20}
            value={dimensionN}
            onChange={e => onDimensionChange(Number(e.target.value))}
          />
        </label>
        <span className="dim-hint">Affects Burnside counts & cost (steps 6-7)</span>
      </div>
    </div>
  );
}
