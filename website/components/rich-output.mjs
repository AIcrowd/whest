export const RICH_OUTPUT_TITLE = '__rich_output__';
const BASE_STYLE = Object.freeze({
  color: null,
  bold: false,
  dim: false,
});
const ANSI_COLOR_CODES = new Map([
  [30, 'black'],
  [31, 'red'],
  [32, 'green'],
  [33, 'yellow'],
  [34, 'blue'],
  [35, 'magenta'],
  [36, 'cyan'],
  [37, 'white'],
]);

function toArray(value) {
  if (value === undefined || value === null || value === false) return [];
  return Array.isArray(value) ? value : [value];
}

function isElementLike(value) {
  return value !== null && typeof value === 'object' && 'props' in value;
}

function isLineElement(value) {
  if (!isElementLike(value)) return false;
  const className = value.props?.className;
  return typeof className === 'string' && className.split(/\s+/).includes('line');
}

export function isRichOutputBlock(props = {}) {
  return props.title === RICH_OUTPUT_TITLE;
}

export function decodeAnsiEscapes(text) {
  if (typeof text !== 'string' || !text.includes('\\')) return text;

  return text.replace(
    /\\u\{([0-9a-fA-F]+)\}|\\u([0-9a-fA-F]{4})|\\x([0-9a-fA-F]{2})|\\n|\\r|\\t|\\\\/g,
    (match, unicodeCodePoint, unicodeCodeUnit, hexCodeUnit) => {
      if (unicodeCodePoint) {
        return String.fromCodePoint(Number.parseInt(unicodeCodePoint, 16));
      }

      if (unicodeCodeUnit) {
        return String.fromCharCode(Number.parseInt(unicodeCodeUnit, 16));
      }

      if (hexCodeUnit) {
        return String.fromCharCode(Number.parseInt(hexCodeUnit, 16));
      }

      switch (match) {
        case '\\n':
          return '\n';
        case '\\r':
          return '\r';
        case '\\t':
          return '\t';
        case '\\\\':
          return '\\';
        default:
          return match;
      }
    },
  );
}

function applyAnsiCode(style, rawCode) {
  const code = Number.parseInt(rawCode, 10);

  if (Number.isNaN(code) || code === 0) {
    return { ...BASE_STYLE };
  }

  if (code === 1) {
    return { ...style, bold: true };
  }

  if (code === 2) {
    return { ...style, dim: true };
  }

  if (code === 22) {
    return { ...style, bold: false, dim: false };
  }

  if (code === 39) {
    return { ...style, color: null };
  }

  return ANSI_COLOR_CODES.has(code)
    ? { ...style, color: ANSI_COLOR_CODES.get(code) }
    : style;
}

function appendStyledText(lines, style, chunk) {
  const parts = chunk.split('\n');

  for (let index = 0; index < parts.length; index += 1) {
    const part = parts[index];

    if (part) {
      const currentLine = lines[lines.length - 1];
      const lastSegment = currentLine.at(-1);

      if (
        lastSegment &&
        lastSegment.color === style.color &&
        lastSegment.bold === style.bold &&
        lastSegment.dim === style.dim
      ) {
        lastSegment.text += part;
      } else {
        currentLine.push({ text: part, ...style });
      }
    }

    if (index < parts.length - 1) {
      lines.push([]);
    }
  }
}

export function parseAnsiRichText(text) {
  const lines = [[]];
  const ansiRegex = /\u001b\[([0-9;]*)m/g;
  let style = { ...BASE_STYLE };
  let lastIndex = 0;

  for (const match of text.matchAll(ansiRegex)) {
    appendStyledText(lines, style, text.slice(lastIndex, match.index));

    const codes = match[1] ? match[1].split(';') : ['0'];
    for (const code of codes) {
      style = applyAnsiCode(style, code);
    }

    lastIndex = match.index + match[0].length;
  }

  appendStyledText(lines, style, text.slice(lastIndex));
  return lines;
}

export function extractRichOutputText(node) {
  if (node === undefined || node === null || node === false) return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);

  if (Array.isArray(node)) {
    if (node.length > 0 && node.every(isLineElement)) {
      return node.map((child) => extractRichOutputText(child)).join('\n');
    }

    return node.map((child) => extractRichOutputText(child)).join('');
  }

  if (!isElementLike(node)) return '';

  if (isLineElement(node)) {
    return extractRichOutputText(node.props?.children);
  }

  const children = toArray(node.props?.children);
  if (children.length > 0 && children.every(isLineElement)) {
    return children.map((child) => extractRichOutputText(child)).join('\n');
  }

  return extractRichOutputText(children);
}
