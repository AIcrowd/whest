import { useCallback, useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Check, Copy } from 'lucide-react';
import ExplorerSectionCard from './ExplorerSectionCard.jsx';

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
    <pre className="min-h-0 h-full overflow-auto whitespace-pre-wrap rounded-xl bg-slate-950 p-5 font-mono text-sm leading-7 text-slate-300">
      <code
        className="[&_.hl-cmt]:text-slate-500 [&_.hl-fn]:font-semibold [&_.hl-fn]:text-sky-300 [&_.hl-kw]:font-semibold [&_.hl-kw]:text-rose-300 [&_.hl-num]:text-amber-300 [&_.hl-str]:text-emerald-300"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </pre>
  );
}

export default function PythonCodeBlock({
  code,
  title,
  description,
  className,
  contentClassName,
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [code]);

  const codeBody = (
    <div className="relative min-h-0 flex-1">
      <PythonHighlight code={code} />
      <Button
        type="button"
        size="icon-sm"
        variant="outline"
        className="absolute right-3 top-3 z-10 size-7 border-white/25 bg-slate-900/85 text-slate-200 hover:bg-slate-900"
        onClick={handleCopy}
        aria-label={copied ? 'Copied' : 'Copy code'}
        title={copied ? 'Copied' : 'Copy code'}
      >
        {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
      </Button>
    </div>
  );

  if (!title && !description) {
    return (
      <div className={['flex min-h-0 flex-col', className, contentClassName].filter(Boolean).join(' ')}>
        {codeBody}
      </div>
    );
  }

  return (
    <ExplorerSectionCard
      eyebrow={title}
      description={description}
      className={['border-border/70 bg-muted/20', className].filter(Boolean).join(' ')}
      contentClassName={['pt-5', 'min-h-0', 'flex', 'flex-col', contentClassName].filter(Boolean).join(' ')}
      action={null}
    >
      {codeBody}
    </ExplorerSectionCard>
  );
}
