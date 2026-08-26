// Runtime check: load built pages in a real browser and assert that the things Documenter does
// client-side actually happened.
//
// This exists because a purely static check cannot catch the failure that motivated it. In August
// 2026 every page on examples.rxinfer.com shipped with raw `$...$` math and uncoloured code while
// the HTML was perfectly valid. The cause was one script: DocumenterMermaid injected
// `<script type="module">import mermaid from '.../mermaid@11/...'` into every page, mermaid inlines
// fastdom, and fastdom's UMD wrapper checks `typeof define == "function"`. With require.js on the
// page it took the AMD branch and fired an anonymous `define()`, which require.js attributed to the
// module it was fetching - jQuery. `$` stopped being a function, so every `$(document).ready(...)`
// in documenter.js threw and neither KaTeX nor highlight.js ever ran.
//
// `check-rendered-math.mjs` reported zero problems throughout, because the LaTeX was fine. Only
// executing the page finds this class of bug. `docs/make.jl` also guards the specific mechanism
// (`check_amd_conflicts`); this check is the backstop for whatever breaks documenter.js next.

import { createServer } from 'node:http';
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const positional = args.filter((a) => !a.startsWith('--'));
const flag = (name) => args.find((a) => a.startsWith(`--${name}=`))?.split('=')[1];
const BUILD_DIR = positional[0] ?? join(HERE, '..', 'build');
const LIMIT = Number(flag('limit') ?? Infinity);

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch {
  console.error(
    `SKIP  check-rendered-pages: the 'playwright' package is not installed.\n` +
    `      Run 'npm install --prefix docs/tools && npx --prefix docs/tools playwright install chromium'.`
  );
  process.exit(0);
}

let browser;
try {
  browser = await chromium.launch();
} catch (error) {
  console.error(
    `SKIP  check-rendered-pages: could not launch Chromium (${error.message.split('\n')[0]}).\n` +
    `      Run 'npx --prefix docs/tools playwright install chromium' to install the browser.`
  );
  process.exit(0);
}

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.gif': 'image/gif', '.woff': 'font/woff', '.woff2': 'font/woff2', '.ico': 'image/x-icon',
  '.map': 'application/json', '.xml': 'application/xml',
};

// Serve over http rather than file:// - a null origin changes how the browser treats the page's
// script loads, and we want this to look like the real site as closely as possible.
const server = createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  let path = join(BUILD_DIR, url);
  if (existsSync(path) && statSync(path).isDirectory()) path = join(path, 'index.html');
  if (!path.startsWith(BUILD_DIR) || !existsSync(path)) {
    res.writeHead(404).end('not found');
    return;
  }
  res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' });
  res.end(readFileSync(path));
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) walk(path, out);
    else if (entry === 'index.html') out.push(path);
  }
  return out;
}

// Console noise we must not fail on: the pages embed Google Analytics and a Google search widget,
// and those requests are routinely blocked or unavailable in CI. Failing on them would make this
// check flaky, which is worse than not having it.
const IGNORED_CONSOLE = [
  /googletagmanager|google-analytics|gstatic|googleapis|gen-search-widget/i,
  /favicon/i,
];
// Anything matching these is a genuine break in Documenter's own client-side setup.
const FATAL_CONSOLE = [
  /is not a function/i,
  /Mismatched anonymous define/i,
  /Uncaught/i,
  /documenter\.js/i,
  /require\.js/i,
];

const pages = walk(BUILD_DIR).slice(0, LIMIT);
const failures = [];
let checked = 0;

async function checkPage(file) {
  const rel = relative(BUILD_DIR, file);
  const html = readFileSync(file, 'utf8');
  const expectsMath = html.includes('class="math-container"') || html.includes('<span>$');
  const expectsHighlight = html.includes('language-julia');

  const page = await browser.newPage();
  const consoleErrors = [];
  const pageErrors = [];
  const failedAssets = [];

  page.on('console', (msg) => {
    if (msg.type() !== 'error') return;
    const text = msg.text();
    if (IGNORED_CONSOLE.some((re) => re.test(text))) return;
    consoleErrors.push(text);
  });
  page.on('pageerror', (error) => pageErrors.push(error.message.split('\n')[0]));
  page.on('requestfailed', (request) => {
    const url = request.url();
    if (IGNORED_CONSOLE.some((re) => re.test(url))) return;
    failedAssets.push(`${url} (${request.failure()?.errorText ?? 'failed'})`);
  });

  const problems = [];
  try {
    await page.goto(`${origin}/${rel}`, { waitUntil: 'load', timeout: 30_000 });
    // documenter.js pulls jQuery/KaTeX/highlight.js through require.js, so the DOM is still
    // changing after `load`. Give those a moment to resolve.
    if (expectsMath) {
      await page.waitForFunction(() => document.querySelector('.katex') !== null, { timeout: 15_000 })
        .catch(() => { /* asserted below with a better message */ });
    }
    await page.waitForTimeout(300);

    if (expectsMath) {
      const katexCount = await page.locator('.katex').count();
      if (katexCount === 0) {
        const katexAssetFailed = failedAssets.some((a) => /katex/i.test(a));
        problems.push(katexAssetFailed
          ? `no math was typeset, and a KaTeX asset failed to load - check network access to the CDN:\n        ${failedAssets.filter((a) => /katex/i.test(a)).join('\n        ')}`
          : 'the page contains math but KaTeX typeset none of it. Something threw before ' +
            'renderMathInElement ran - check the console errors above, and see check_amd_conflicts in docs/make.jl.');
      }
      // Read the prose only. Code blocks are full of Julia string interpolation (`"step $k"`),
      // which is not math and which KaTeX itself skips.
      const raw = await page.evaluate(() => {
        const article = document.querySelector('article');
        if (!article) return [];
        const clone = article.cloneNode(true);
        clone.querySelectorAll('pre, code, script, style').forEach((n) => n.remove());
        const text = clone.innerText || clone.textContent || '';
        const patterns = [/\\\[[\s\S]{0,60}/g, /\$\\[A-Za-z]{2,}[^$\n]{0,60}\$/g];
        return patterns.flatMap((re) => Array.from(text.matchAll(re), (m) => m[0])).slice(0, 3);
      });
      if (raw.length) problems.push(`raw LaTeX is visible in the rendered text: ${raw.map((r) => JSON.stringify(r)).join(', ')}`);
    }

    if (expectsHighlight) {
      const spans = await page.locator('pre code.language-julia span[class^="hljs-"]').count();
      if (spans === 0) {
        problems.push('Julia code blocks carry no highlight.js spans. `docs/make.jl` builds with ' +
          'prerender=true, so these should be in the HTML already - check that `node` was available at build time.');
      }
    }
  } catch (error) {
    problems.push(`page failed to load: ${error.message.split('\n')[0]}`);
  }

  for (const message of pageErrors) problems.push(`uncaught exception: ${message}`);
  for (const message of consoleErrors) {
    if (FATAL_CONSOLE.some((re) => re.test(message))) problems.push(`console error: ${message}`);
  }

  await page.close();
  checked++;
  return { page: rel, problems };
}

// A page load is mostly waiting on the CDN, so run a few at a time - sequentially this takes over
// two minutes for the full site.
const CONCURRENCY = Number(flag('concurrency') ?? 4);
const queue = [...pages];
await Promise.all(Array.from({ length: Math.min(CONCURRENCY, queue.length) }, async () => {
  while (queue.length) {
    const result = await checkPage(queue.shift());
    if (result.problems.length) failures.push(result);
  }
}));

failures.sort((a, b) => a.page.localeCompare(b.page));

await browser.close();
server.close();

if (failures.length === 0) {
  console.log(`OK    check-rendered-pages: ${checked} page(s) rendered math and highlighting correctly in Chromium.`);
  process.exit(0);
}

console.error(`\nFAIL  check-rendered-pages: ${failures.length} of ${checked} page(s) did not render correctly.\n`);
for (const failure of failures) {
  console.error(`  ${failure.page}`);
  for (const problem of failure.problems) console.error(`      ${problem}`);
  console.error('');
}
console.error(
  `A page can ship valid HTML and still render nothing: Documenter typesets math and colours code\n` +
  `with client-side JavaScript, so one script that throws early takes the whole page with it.\n`
);
process.exit(1);
