import { test, expect } from '@playwright/test';

test('log console', async ({ page }) => {
  const logs = [];
  page.on('console', (msg) => {
    logs.push(`${msg.type()} ${msg.text()}`);
  });
  page.on('pageerror', (err) => {
    logs.push(`PAGEERR ${err.stack}`);
  });
  page.on('requestfailed', (req) => {
    const failure = req.failure();
    logs.push(`REQFAIL ${req.url()} ${failure?.errorText}`);
  });

  await page.goto('http://127.0.0.1:3001/symmetry-aware-einsum-contractions/', { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(7000);
  console.log('----BROWSER LOGS----');
  for (const line of logs) console.log(line);

  await expect(page.locator('body')).toBeVisible();
});
