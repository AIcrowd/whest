/**
 * Parse a Python-style function signature into structured tokens for display.
 *
 * Input examples:
 *   fnp.einsum(subscripts: 'str', *operands: '_np.ndarray', out=None, **kwargs) -> '_np.ndarray'
 *   numpy.einsum(subscripts, *operands, out=None, dtype=None, order='K')
 *
 * Output structure:
 *   {
 *     namespace: 'fnp.',
 *     functionName: 'einsum',
 *     parameters: [
 *       { prefix: '', name: 'subscripts', type: "'str'", default: undefined },
 *       { prefix: '*', name: 'operands', type: "'_np.ndarray'", default: undefined },
 *       { prefix: '', name: 'out', type: undefined, default: 'None' },
 *       { prefix: '**', name: 'kwargs', type: undefined, default: undefined },
 *     ],
 *     returnType: "'_np.ndarray'",
 *   }
 *
 * Falls back gracefully (parameters: undefined) for signatures that don't
 * match the expected shape; the caller renders the raw string in that case.
 */
export interface ParsedParameter {
  prefix: '' | '*' | '**';
  name: string;
  type?: string;
  default?: string;
}

export interface ParsedSignature {
  namespace: string;
  functionName: string;
  parameters?: ParsedParameter[];
  returnType?: string;
  remainder?: string; // raw text after the function name when parsing failed
}

const OPENERS: Record<string, string> = {'(': ')', '[': ']', '{': '}'};

/** Split a string at top-level occurrences of `sep` (depth-0 only). */
function splitTopLevel(input: string, sep: string): string[] {
  const out: string[] = [];
  const stack: string[] = [];
  let buffer = '';
  let quote: string | null = null;
  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    if (quote) {
      buffer += ch;
      if (ch === '\\' && i + 1 < input.length) {
        buffer += input[++i];
        continue;
      }
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === "'" || ch === '"') {
      quote = ch;
      buffer += ch;
      continue;
    }
    if (ch in OPENERS) {
      stack.push(OPENERS[ch]);
      buffer += ch;
      continue;
    }
    if (stack.length > 0 && ch === stack[stack.length - 1]) {
      stack.pop();
      buffer += ch;
      continue;
    }
    if (stack.length === 0 && ch === sep) {
      out.push(buffer);
      buffer = '';
      continue;
    }
    buffer += ch;
  }
  if (buffer.length > 0) out.push(buffer);
  return out;
}

/** Find the `=` that separates a parameter's type from its default value at depth 0. */
function findDefaultSplit(input: string): number {
  const stack: string[] = [];
  let quote: string | null = null;
  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    if (quote) {
      if (ch === '\\' && i + 1 < input.length) { i++; continue; }
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === "'" || ch === '"') { quote = ch; continue; }
    if (ch in OPENERS) { stack.push(OPENERS[ch]); continue; }
    if (stack.length > 0 && ch === stack[stack.length - 1]) { stack.pop(); continue; }
    if (stack.length === 0 && ch === '=') return i;
  }
  return -1;
}

function parseParameter(raw: string): ParsedParameter | null {
  let s = raw.trim();
  if (!s) return null;
  let prefix: '' | '*' | '**' = '';
  if (s.startsWith('**')) { prefix = '**'; s = s.slice(2).trim(); }
  else if (s.startsWith('*')) { prefix = '*'; s = s.slice(1).trim(); }
  if (!s) return null;
  // Ignore positional-only / keyword-only markers (`/`, `*`).
  if (s === '/' || s === '*') return null;

  // Split off default at top-level `=`.
  const eq = findDefaultSplit(s);
  let nameAndType = s;
  let defaultValue: string | undefined;
  if (eq >= 0) {
    nameAndType = s.slice(0, eq).trim();
    defaultValue = s.slice(eq + 1).trim();
  }

  // Split off type annotation at top-level `:`.
  let name = nameAndType;
  let typeAnno: string | undefined;
  const colonIdx = nameAndType.indexOf(':');
  if (colonIdx >= 0) {
    // Simple `:` is fine; function-level annotations don't nest a `:` at depth 0.
    name = nameAndType.slice(0, colonIdx).trim();
    typeAnno = nameAndType.slice(colonIdx + 1).trim();
  }
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
    // Bail if the "name" doesn't look like an identifier.
    return null;
  }
  return { prefix, name, type: typeAnno, default: defaultValue };
}

export function parseSignature(signature: string): ParsedSignature {
  const fail = (remainder: string): ParsedSignature => {
    const m = signature.match(/^([^([]+)/);
    const head = m ? m[1] : signature;
    const lastDot = head.lastIndexOf('.');
    return {
      namespace: lastDot >= 0 ? head.slice(0, lastDot + 1) : '',
      functionName: lastDot >= 0 ? head.slice(lastDot + 1) : head,
      remainder,
    };
  };

  // Match `name_chain(...args...) [-> return_type]`.
  const m = signature.match(/^([^([]+)\(([\s\S]*?)\)\s*(?:->\s*([\s\S]+?))?\s*$/);
  if (!m) return fail(signature.replace(/^[^([]+/, ''));

  const head = m[1];
  const args = m[2];
  const returnType = m[3]?.trim();
  const lastDot = head.lastIndexOf('.');
  const namespace = lastDot >= 0 ? head.slice(0, lastDot + 1) : '';
  const functionName = lastDot >= 0 ? head.slice(lastDot + 1) : head;

  const rawParams = splitTopLevel(args, ',');
  const parameters: ParsedParameter[] = [];
  for (const raw of rawParams) {
    const parsed = parseParameter(raw);
    if (!parsed) {
      if (raw.trim().length === 0) continue;
      // One bad token aborts structured rendering; fall back to plaintext.
      return fail(`(${args})${returnType ? ` -> ${returnType}` : ''}`);
    }
    parameters.push(parsed);
  }

  return { namespace, functionName, parameters, returnType };
}
