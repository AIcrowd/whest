# Symmetry Aware Einsum Contractions — setup notes for agents

Route: `/symmetry-aware-einsum-contractions/`
Page:  `website/app/symmetry-aware-einsum-contractions/page.tsx`
Entry: `website/components/symmetry-aware-einsum-contractions/index.tsx`

## Run it in Claude Preview

```
preview_start(name="symmetry-aware-einsum-contractions")
```

The launch config lives in `.claude/launch.json`, which is **gitignored per-machine**. If the entry is missing, add this block to the `configurations` array:

```json
{
  "name": "symmetry-aware-einsum-contractions",
  "runtimeExecutable": "bash",
  "runtimeArgs": [
    "-c",
    "set -e; if [ -s \"${NVM_DIR:-$HOME/.nvm}/nvm.sh\" ]; then . \"${NVM_DIR:-$HOME/.nvm}/nvm.sh\"; nvm use --silent default >/dev/null || true; fi; NODE_MAJOR=$(node -p 'process.versions.node.split(\".\")[0]' 2>/dev/null || echo 0); if [ \"$NODE_MAJOR\" -lt 20 ]; then echo \"[launch] Node $NODE_MAJOR too old, need >=20. Run: nvm install 20 && nvm alias default 20\" >&2; exit 1; fi; cd website; if [ ! -d node_modules ] || ! ls node_modules/@tailwindcss/oxide-* >/dev/null 2>&1; then echo \"[launch] installing website deps under $(node -v)\"; rm -rf node_modules package-lock.json .next; npm install; fi; exec npm run dev -- --port 3047"
  ],
  "port": 3047
}
```

This boots Next.js on port **3047**, activates your nvm default so `node -v` matches what the build uses, hard-fails early if it's older than 20, and auto-heals the arm64-binary bug below by reinstalling.

## Known gotchas — read before reinstalling

### 1. Next.js 16 requires Node ≥ 20
Check that your `nvm alias default` points at a Node ≥ 20 — older nvm defaults (e.g. Node 16) will silently break the build and the `make ci` / pre-push hook. One-time fix:

```bash
nvm install 20    # or newer
nvm alias default 20
```

Then run the dev server from `website/`:

```bash
cd website
npm install
npm run dev -- --port 3047
```

### 2. `@tailwindcss/oxide-*` missing after install
This is the [npm optional-deps bug](https://github.com/npm/cli/issues/4828). If `npm install` ever ran under the wrong Node (e.g. Node 16), npm caches a broken `package-lock.json` that omits the platform-specific native binary. The dev server then crashes with:

```
Error: Cannot find module '@tailwindcss/oxide-darwin-arm64'
```

(s/darwin-arm64/your platform/ on Linux/Windows.) Fix: nuke and reinstall with a modern Node active.

```bash
cd website
rm -rf node_modules package-lock.json .next
npm install
```

Verify: `ls node_modules/@tailwindcss/` should include an `oxide-<your-platform>` directory.

The launch config already guards against this — it checks for the oxide native binary and reinstalls if missing — so **prefer using `preview_start` over running `npm install` by hand**.

### 3. First request is slow
Turbopack compiles the route on demand. The first `GET /symmetry-aware-einsum-contractions/` can take 10–30s; subsequent requests are instant.

## Tests

Integration tests for this component live as `website/symmetry-explorer.*.test.mjs` and `website/symmetry-aware-einsum-contractions.route.test.mjs`. Run them from `website/` with `npm test` (also needs Homebrew Node).
