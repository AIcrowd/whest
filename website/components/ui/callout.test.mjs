import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';
import React from 'react';
import { Callout } from './callout.tsx';

test('Callout renders children', () => {
  const html = renderToStaticMarkup(
    React.createElement(Callout, null, 'hello world')
  );
  assert.match(html, /hello world/);
});

test('Callout default variant sets data-variant="default"', () => {
  const html = renderToStaticMarkup(
    React.createElement(Callout, { label: 'NOTE' }, 'body text')
  );
  assert.match(html, /data-variant="default"/);
  assert.match(html, /NOTE/);
});

test('Callout accent variant sets data-variant="accent"', () => {
  const html = renderToStaticMarkup(
    React.createElement(Callout, { variant: 'accent', label: 'TIP' }, 'x')
  );
  assert.match(html, /data-variant="accent"/);
  assert.match(html, /TIP/);
});
