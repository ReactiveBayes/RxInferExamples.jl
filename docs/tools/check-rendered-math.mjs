// Static check: re-render every formula in the built HTML with the same KaTeX that the reader's
// browser will run, and fail the build if any of it cannot be typeset.
//
// Why this is needed at all: Documenter does not pre-render math. It emits inline math as
// `<span>$...$</span>` and display math as `<p class="math-container">\[...\]</p>`, and KaTeX's
// `auto-render` typesets it client-side. So invalid LaTeX produces a perfectly successful build
// and a page showing raw dollar signs.
//
// Two failure modes, and the difference matters a lot:
//   * ParseError            - that one formula degrades, the rest of the page is fine.
//   * anything else thrown  - `renderElem` in KaTeX's auto-render only catches ParseError
//                             (contrib/auto-render.js), so the exception escapes its DOM walk and
//                             *every formula after it on the page* stays raw. One bad brace can
//                             blank out a whole tutorial.
//
// What this check canNOT see: valid math that never gets typeset because something else on the
// page broke `documenter.js` before KaTeX ran. That was the actual cause of the outage this
// tooling came from, and only a real browser catches it - see `check-rendered-pages.mjs`.

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const BUILD_DIR = process.argv[2] ?? join(HERE, '..', 'build');

let katex;
try {
  katex = (await import('katex')).default;
} catch {
  console.error(
    `SKIP  check-rendered-math: the 'katex' package is not installed.\n` +
    `      Run 'npm install --prefix docs/tools' to enable this check.`
  );
  process.exit(0);
}

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) walk(path, out);
    else if (entry.endsWith('.html')) out.push(path);
  }
  return out;
}

// KaTeX's auto-render skips these, so we must too - Julia code is full of `$` interpolation.
const IGNORED_TAGS = new Set(['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option']);
const VOID_TAGS = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
  'meta', 'param', 'source', 'track', 'wbr']);

const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ' };
const unescapeHtml = (s) => s.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g, (m, g) => {
  if (g[0] === '#') return String.fromCodePoint(parseInt(g[1] === 'x' || g[1] === 'X' ? g.slice(2) : g.slice(1), g[1] === 'x' || g[1] === 'X' ? 16 : 10));
  return ENTITIES[g] ?? m;
});

// Collect the text nodes a browser would hand to auto-render, with the line they start on so the
// failure can be located in the file.
function textNodes(html) {
  const nodes = [];
  const stack = [];
  const tagRe = /<\/?([a-zA-Z][a-zA-Z0-9-]*)\b[^>]*?(\/?)>|<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<![^>]*>/g;
  let last = 0;
  let match;

  const pushText = (from, to) => {
    if (to <= from) return;
    const raw = html.slice(from, to);
    if (!raw.trim()) return;
    if (stack.some((t) => IGNORED_TAGS.has(t))) return;
    nodes.push({ text: unescapeHtml(raw), offset: from });
  };

  while ((match = tagRe.exec(html)) !== null) {
    pushText(last, match.index);
    last = tagRe.lastIndex;
    const name = match[1]?.toLowerCase();
    if (!name) continue;
    if (match[0].startsWith('</')) {
      const at = stack.lastIndexOf(name);
      if (at !== -1) stack.length = at;
    } else if (!VOID_TAGS.has(name) && !match[2]) {
      stack.push(name);
    }
  }
  pushText(last, html.length);
  return nodes;
}

// Read the delimiters and the KaTeX version straight out of the built documenter.js, so this check
// keeps following Documenter across upgrades instead of asserting against a hardcoded copy.
function readDocumenterConfig() {
  const path = join(BUILD_DIR, 'assets', 'documenter.js');
  let js;
  try {
    js = readFileSync(path, 'utf8');
  } catch {
    return { delimiters: null, version: null };
  }
  const version = js.match(/KaTeX\/(\d+\.\d+\.\d+)\//)?.[1] ?? null;
  let delimiters = null;
  const block = js.match(/"delimiters":\s*(\[[\s\S]*?\n\s*\])/);
  if (block) {
    try {
      delimiters = JSON.parse(block[1]).map((d) => ({ left: d.left, right: d.right, display: d.display }));
    } catch { /* fall through to the default below */ }
  }
  return { delimiters, version };
}

const { delimiters: configured, version: siteKatexVersion } = readDocumenterConfig();
const delimiters = configured ?? [
  { left: '$', right: '$', display: false },
  { left: '$$', right: '$$', display: true },
  { left: '\\[', right: '\\]', display: true },
];

if (siteKatexVersion && siteKatexVersion !== katex.version) {
  console.warn(
    `WARN  the built pages load KaTeX ${siteKatexVersion} but this check runs ${katex.version}.\n` +
    `      Update the 'katex' pin in docs/tools/package.json so the check matches the site.`
  );
}

// Ported from KaTeX contrib/auto-render/splitAtDelimiters.js. Keep it faithful: the point of this
// check is to see exactly what the browser sees, including the brace-depth rule that decides where
// a formula ends.
function findEndOfMath(delimiter, text, startIndex) {
  let index = startIndex;
  let braceLevel = 0;
  while (index < text.length) {
    const character = text[index];
    if (braceLevel <= 0 && text.slice(index, index + delimiter.length) === delimiter) return index;
    else if (character === '\\') index++;
    else if (character === '{') braceLevel++;
    else if (character === '}') braceLevel--;
    index++;
  }
  return -1;
}

const escapeRegex = (s) => s.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
const leftRe = new RegExp('(' + delimiters.map((d) => escapeRegex(d.left)).join('|') + ')');

function splitAtDelimiters(text) {
  const data = [];
  let before = '';
  while (true) {
    const found = text.search(leftRe);
    if (found === -1) break;
    if (found > 0) {
      before = text.slice(0, found);
      data.push({ type: 'text', data: before });
      text = text.slice(found);
    }
    const i = delimiters.findIndex((d) => text.startsWith(d.left));
    const end = findEndOfMath(delimiters[i].right, text, delimiters[i].left.length);
    if (end === -1) {
      data.push({ type: 'unterminated', data: text, before, display: delimiters[i].display });
      break;
    }
    data.push({
      type: 'math',
      data: text.slice(delimiters[i].left.length, end),
      before,
      display: delimiters[i].display,
    });
    text = text.slice(end + delimiters[i].right.length);
  }
  if (text) data.push({ type: 'text', data: text });
  return data;
}

// A `\command`, `\[`, `\{` or a `_{`/`^{` subscript group sitting in ordinary prose means some
// LaTeX never became math - the delimiters around it were not recognised, so the reader sees the
// source instead of a formula. Two spellings cause almost all of it: a newline or space directly
// inside `$`/`$$` (`$$\nx = 1\n$$`), and `$$...$$` used where `$...$` was meant.
const LEAKED_LATEX_RE = /\\[A-Za-z]{2,}|\\[[\]{}]|[_^]\{/;

const problems = [];
const pages = walk(BUILD_DIR);
let formulas = 0;

for (const page of pages) {
  const rel = relative(BUILD_DIR, page);
  for (const node of textNodes(readFileSync(page, 'utf8'))) {
    for (const chunk of splitAtDelimiters(node.text)) {
      const context = (chunk.before ?? '').trim().slice(-70);
      if (chunk.type === 'text' || chunk.type === 'unterminated') {
        // An unterminated `$` on its own is harmless: the browser leaves it as text, which is
        // exactly what a correctly escaped `\$` in prose should look like. What is not harmless is
        // LaTeX sitting in that text.
        const leaked = chunk.data.match(LEAKED_LATEX_RE);
        if (leaked) {
          problems.push({
            page: rel, context, fatal: false, kind: 'LaTeX never became math',
            snippet: chunk.data.slice(0, 200),
            message: `found \`${leaked[0]}\` in ordinary prose, so this formula ships as raw source. ` +
              'Almost always a space or newline directly inside the `$`/`$$` delimiters, or `$$...$$` ' +
              'used inline where `$...$` was meant.',
          });
        }
        continue;
      }
      formulas++;
      try {
        katex.renderToString(chunk.data, { displayMode: chunk.display, throwOnError: true });
      } catch (error) {
        const fatal = !(error instanceof katex.ParseError);
        problems.push({
          page: rel, context, fatal,
          kind: fatal ? `${error.constructor.name} (aborts the whole page)` : 'ParseError',
          snippet: chunk.data.slice(0, 200),
          message: error.message,
        });
      }
    }
  }
}

// One broken equation leaks across several text nodes, so collapse runs of the same kind on the
// same page into a single report - otherwise a single bad `$$` looks like six separate failures.
const collapsed = problems.filter((p, i) => {
  const prev = problems[i - 1];
  return !(prev && prev.page === p.page && prev.kind === p.kind);
});

if (problems.length === 0) {
  console.log(`OK    check-rendered-math: ${formulas} formulas across ${pages.length} pages typeset cleanly (KaTeX ${katex.version}).`);
  process.exit(0);
}

console.error(`\nFAIL  check-rendered-math: ${collapsed.length} broken formula(s) in the built documentation.\n`);
for (const p of collapsed) {
  console.error(`  ${p.page}  [${p.kind}]`);
  if (p.context) console.error(`      after: ...${p.context.replace(/\s+/g, ' ')}`);
  console.error(`      ${p.message}`);
  console.error(`      in: ${p.snippet.replace(/\s*\n\s*/g, ' ⏎ ')}`);
  if (p.fatal) {
    console.error(`      NOTE: KaTeX's auto-render only catches ParseError, so this exception escapes its`);
    console.error(`            DOM walk and leaves EVERY later formula on this page unrendered too.`);
  }
  console.error('');
}
console.error(`Fix the LaTeX in the source notebook, not in docs/build - that directory is generated.`);
console.error(`See the "Mathematical Content" rules in docs/src/how_to_contribute.md.\n`);
process.exit(1);
