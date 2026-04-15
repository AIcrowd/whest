import { useCallback, useMemo, useState } from 'react';

function highlightPython(code) {
  const tokens = [];
  const lines = code.split('\n');

  for (const line of lines) {
    let rest = line;

    const commentIdx = rest.indexOf('#');
    let comment = '';
    if (commentIdx >= 0) {
      comment = rest.slice(commentIdx);
      rest = rest.slice(0, commentIdx);
    }

    let i = 0;
    while (i < rest.length) {
      const ch = rest[i];
      if (ch === "'" || ch === '"') {
        const close = rest.indexOf(ch, i + 1);
        if (close >= 0) {
          if (i > 0) tokens.push({ type: 'code', text: rest.slice(0, i) });
          tokens.push({ type: 'str', text: rest.slice(i, close + 1) });
          rest = rest.slice(close + 1);
          i = 0;
          continue;
        }
      }
      i += 1;
    }

    if (rest) tokens.push({ type: 'code', text: rest });
    if (comment) tokens.push({ type: 'comment', text: comment });
    tokens.push({ type: 'newline' });
  }

  const KEYWORDS = new Set([
    'import', 'from', 'as', 'for', 'in', 'if', 'else', 'def', 'return',
    'class', 'sum', 'range', 'list', 'True', 'False', 'None',
  ]);

  function esc(value) {
    return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function highlightCode(text) {
    const re = /(\b\d+\.?\d*\b)|(\b[a-zA-Z_]\w*\b)/g;
    let result = '';
    let last = 0;
    let match;
    while ((match = re.exec(text)) !== null) {
      result += esc(text.slice(last, match.index));
      const num = match[1];
      const word = match[2];
      if (num) {
        result += `<span class="hl-num">${esc(num)}</span>`;
      } else if (KEYWORDS.has(word)) {
        result += `<span class="hl-kw">${esc(word)}</span>`;
      } else {
        const after = text.slice(re.lastIndex).match(/^\s*\(/);
        if (after) {
          result += `<span class="hl-fn">${esc(word)}</span>`;
        } else {
          result += esc(word);
        }
      }
      last = re.lastIndex;
    }
    result += esc(text.slice(last));
    return result;
  }

  const parts = [];
  for (const tok of tokens) {
    if (tok.type === 'newline') parts.push('\n');
    else if (tok.type === 'str') parts.push(`<span class="hl-str">${esc(tok.text)}</span>`);
    else if (tok.type === 'comment') parts.push(`<span class="hl-cmt">${esc(tok.text)}</span>`);
    else parts.push(highlightCode(tok.text));
  }

  const result = parts.join('');
  return result.endsWith('\n') ? result.slice(0, -1) : result;
}

function PythonHighlight({ code }) {
  const html = useMemo(() => highlightPython(code), [code]);
  return (
    <pre className="overflow-x-auto whitespace-pre-wrap rounded-lg bg-gray-900 p-4 font-mono text-xs leading-relaxed text-gray-300">
      <code dangerouslySetInnerHTML={{ __html: html }} />
    </pre>
  );
}

export default function PythonCodeBlock({
  code,
  title = 'Reference Code',
  description = 'This is a generated Python sketch of the contraction you are about to analyze.',
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [code]);

  return (
    <div className="rounded-xl border border-gray-200 bg-gray-50 p-4">
      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-500">{title}</div>
      <p className="mt-1 text-xs text-gray-600">{description}</p>
      <div className="relative mt-3">
        <button
          className="absolute right-2 top-2 z-10 rounded p-1.5 text-gray-400 transition-colors hover:bg-gray-700 hover:text-white"
          onClick={handleCopy}
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
        <PythonHighlight code={code} />
      </div>
    </div>
  );
}
