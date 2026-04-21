import {
  EXPLORER_THEME_PRESETS,
  getExplorerThemePreset,
} from '../lib/explorerTheme.js';

export default function ExplorerThemeDock({ explorerThemeId, onChange }) {
  const activeTheme = getExplorerThemePreset(explorerThemeId);

  return (
    <details className="fixed bottom-6 right-6 z-50">
      <summary
        className="list-none cursor-pointer rounded-full border-[color:color-mix(in_oklab,var(--explorer-editorial-accent)_28%,var(--explorer-border))] bg-white px-4 py-2 text-[12px] font-semibold uppercase tracking-[0.16em] text-foreground shadow-[0_10px_30px_rgba(41,44,45,0.12)] transition-colors hover:border-[var(--coral)] hover:text-[var(--coral)]"
      >
        Explorer Theme
      </summary>
      <div className="mt-3 w-[min(22rem,calc(100vw-2rem))] rounded-2xl border-[color:color-mix(in_oklab,var(--explorer-editorial-accent)_24%,var(--explorer-border))] bg-white p-4 shadow-[0_18px_48px_rgba(41,44,45,0.16)]">
        <div className="mb-3">
          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--explorer-editorial-accent)]">
            Explorer Theme
          </div>
          <p className="mt-1 font-serif text-[15px] leading-[1.55] text-gray-700">
            {activeTheme.summary}
          </p>
        </div>
        <label className="block text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--explorer-editorial-accent)]">
          <span className="sr-only">Explorer theme picker</span>
          <select
            aria-label="Explorer theme picker"
            value={explorerThemeId}
            onChange={(event) => onChange(event.target.value)}
            className="mt-1 w-full rounded-xl border border-border bg-white px-3 py-2 text-[14px] font-medium text-foreground outline-none transition-colors focus:border-[var(--coral)]"
          >
            {EXPLORER_THEME_PRESETS.map((theme) => (
              <option key={theme.id} value={theme.id}>
                {theme.label}
              </option>
            ))}
          </select>
        </label>
      </div>
    </details>
  );
}
