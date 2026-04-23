import Latex from './Latex.jsx';
import {
  notationColor,
  notationColoredLatex,
  notationLatex,
  notationText,
} from '../lib/notationSystem.js';

export default function NotationSymbol({
  id,
  mode = 'text',
  colorize = true,
  className = '',
}) {
  if (mode === 'math') {
    return (
      <span className={className}>
        <Latex math={colorize ? notationColoredLatex(id) : notationLatex(id)} />
      </span>
    );
  }

  return (
    <span className={className} style={{ color: notationColor(id), fontWeight: 600 }}>
      {notationText(id)}
    </span>
  );
}
