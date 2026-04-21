import React from 'react';
import Latex from './Latex.jsx';
import GlossaryProse from './GlossaryProse.jsx';

/**
 * Render a glossary as a paper-style definition list.
 *
 * `entries` is `Array<{ term: string, definition: string }>` where `term`
 * is a bare LaTeX expression (no surrounding `$`) and `definition` is
 * prose that may contain inline `$...$` math (rendered by GlossaryProse).
 */
export default function GlossaryList({ entries, themeOverride = null }) {
  if (!entries || !Array.isArray(entries) || entries.length === 0) return null;
  return (
    <dl className="space-y-1.5">
      {entries.map(({ term, definition }, i) => (
        <div key={`${term}-${i}`} className="flex items-baseline gap-2">
          <dt className="shrink-0 whitespace-nowrap text-gray-900">
            <Latex math={term} themeOverride={themeOverride} />
          </dt>
          <dd className="min-w-0 flex-1 text-gray-700">
            <GlossaryProse text={definition} themeOverride={themeOverride} />
          </dd>
        </div>
      ))}
    </dl>
  );
}
