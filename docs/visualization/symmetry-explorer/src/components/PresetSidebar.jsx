import { SYMMETRY_ICONS } from '../engine/colorPalette.js';

const CUSTOM_IDX = -1;

export default function PresetSidebar({ examples, selected, onSelect, onCustom }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">Presets</div>
      <button
        className={`sidebar-item sidebar-item-custom ${selected === CUSTOM_IDX ? 'active' : ''}`}
        onClick={onCustom}
      >
        <div className="sidebar-item-accent" style={{ backgroundColor: '#7C3AED' }} />
        <div className="sidebar-item-content">
          <div className="sidebar-item-name">✎ Custom</div>
          <code className="sidebar-item-formula">Define your own</code>
        </div>
      </button>
      <nav className="sidebar-list">
        {examples.map((ex, i) => (
          <button
            key={ex.id}
            className={`sidebar-item ${selected === i ? 'active' : ''}`}
            onClick={() => onSelect(i)}
          >
            <div
              className="sidebar-item-accent"
              style={{ backgroundColor: ex.color }}
            />
            <div className="sidebar-item-content">
              <div className="sidebar-item-name">{ex.name}</div>
              <code className="sidebar-item-formula">{ex.formula}</code>
              <span className="sidebar-item-group">{ex.expectedGroup}</span>
            </div>
          </button>
        ))}
      </nav>
    </aside>
  );
}
