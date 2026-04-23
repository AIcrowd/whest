function toBase64(str) {
  if (typeof btoa === 'function') return btoa(str);
  return Buffer.from(str, 'binary').toString('base64');
}

function fromBase64(str) {
  if (typeof atob === 'function') return atob(str);
  return Buffer.from(str, 'base64').toString('binary');
}

export function encodePlaygroundState(state) {
  const json = JSON.stringify(state);
  return toBase64(encodeURIComponent(json));
}

export function decodePlaygroundState(str) {
  if (!str || typeof str !== 'string') return null;
  try {
    const encoded = fromBase64(str);
    const json = decodeURIComponent(encoded);
    return JSON.parse(json);
  } catch (e) {
    return null;
  }
}
