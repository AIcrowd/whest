export function cn(...inputs) {
  return inputs.flatMap((value) => {
    if (Array.isArray(value)) return value;
    return [value];
  }).filter(Boolean).join(' ');
}
