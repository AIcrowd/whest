import RoleBadge from './RoleBadge.jsx';

function actionTags(actionSummary = {}) {
  const tags = [];
  if (actionSummary.hasCross) tags.push({ key: 'cross', className: 'group-tag-cross', text: 'cross V/W present' });
  if (actionSummary.hasVOnly) tags.push({ key: 'v', className: 'group-tag-v', text: 'V-only action present' });
  if (actionSummary.hasWOnly) tags.push({ key: 'w', className: 'group-tag-w', text: 'W-only action present' });
  if (actionSummary.hasCorrelated) tags.push({ key: 'correlated', className: 'group-tag-correlated', text: 'correlated V/W action present' });
  return tags;
}

export default function GroupView({ group }) {
  const tags = actionTags(group.actionSummary);

  return (
    <div className="group-view">
      <div className="group-side-section">
        <h4 className="group-side-title">Full Group On Active Labels</h4>

        <div className="group-summary-row">
          <div className="group-badge">
            <span className="group-name">{group.fullGroupName}</span>
            <span className="group-order">order {group.fullOrder}</span>
          </div>
          <div className="group-meta">
            <span>{group.fullDegree} labels ({group.allLabels.join(', ')})</span>
            <span>{group.fullGenerators.length} generator{group.fullGenerators.length === 1 ? '' : 's'}</span>
          </div>
        </div>

        <p className="group-side-note">
          This is the main symmetry object for the example. The V/W split still matters as a role
          annotation on labels, but the cost model is driven by the full group on all active labels.
        </p>

        {tags.length > 0 && (
          <div className="group-tags">
            {tags.map((tag) => (
              <span key={tag.key} className={`group-tag ${tag.className}`}>{tag.text}</span>
            ))}
          </div>
        )}

        <div className="group-split-summary">
          <div className="group-split-card">
            <RoleBadge role="v" as="badge" interactive>V</RoleBadge>
            <strong>{group.vGroupName || 'trivial'}</strong>
            <span>{group.vLabels.length} free label{group.vLabels.length === 1 ? '' : 's'}</span>
          </div>
          <div className="group-split-card">
            <RoleBadge role="w" as="badge" interactive>W</RoleBadge>
            <strong>{group.wLabels.length > 0 ? (group.wGroupName || 'trivial') : 'none'}</strong>
            <span>{group.wLabels.length} summed label{group.wLabels.length === 1 ? '' : 's'}</span>
          </div>
        </div>

        <div className="group-elements">
          <div className="perm-list">
            {(group.fullElements || []).map((element, idx) => {
              const isId = element.isIdentity;
              const isGenerator = (group.fullGenerators || []).some((generator) => generator.key() === element.key());
              const cycles = element.cyclicForm();
              return (
                <div
                  key={`full-group-element-${idx}`}
                  className={`perm-card ${isId ? 'identity' : ''} ${isGenerator ? 'generator' : ''}`}
                >
                  <code className="perm-notation">{element.cycleNotation(group.allLabels)}</code>
                  <span className="perm-structure">
                    {isId
                      ? 'identity'
                      : isGenerator
                        ? 'generator'
                        : cycles.map((cycle) => `${cycle.length}-cycle`).join(' + ')}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
