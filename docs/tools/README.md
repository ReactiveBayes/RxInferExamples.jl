# Post-build documentation checks

Run these after `make docs`:

```bash
make docs-check          # installs deps if needed, then runs both checks
```

They exist because of a specific outage. In August 2026 every page on
examples.rxinfer.com published raw `$...$` and `\[...\]` instead of typeset math, and Julia code
blocks lost their colours. The build was green, the generated markdown was correct, and the HTML
was valid — the LaTeX in it re-rendered cleanly under KaTeX. Nothing in the pipeline could see the
problem, because Documenter typesets math *in the reader's browser*.

The cause was a single injected script. `DocumenterMermaid` put
`<script type="module">import mermaid from '.../mermaid@11/...'</script>` on every page. Mermaid
inlines `fastdom`, whose UMD wrapper is:

```js
typeof define=="function"?define(function(){return c}):typeof H=="object"&&(H.exports=c)
```

Documenter loads jQuery, KaTeX and highlight.js through require.js, which installs a global AMD
`define`. So that wrapper took the AMD branch and fired an *anonymous* `define()`. require.js
attributes an anonymous define to whichever module it is currently fetching — jQuery. `$` stopped
being a function, every `$(document).ready(...)` in `documenter.js` threw, and neither
`renderMathInElement` nor `hljs.highlightAll()` ever ran. Because it depended on which fetch was in
flight, it came and went between reloads.

## The checks

### `check-rendered-math.mjs`

Static. Walks the built HTML, extracts the text nodes a browser would hand to KaTeX's
`auto-render` (skipping `pre`/`code`/`script`, since Julia code is full of `$` interpolation),
splits them on the delimiters, and re-renders every formula with `throwOnError: true`.

The KaTeX version and the delimiter list are read out of the built `assets/documenter.js`, so the
check follows Documenter across upgrades instead of asserting against a hardcoded copy. Keep the
`katex` pin in `package.json` matching what the site loads; the script warns when they diverge.

Two failure modes, and the difference matters:

- **ParseError** — that one formula degrades, the rest of the page is fine.
- **anything else thrown** — `renderElem` in KaTeX's `auto-render` catches only `ParseError`, so the
  exception escapes its DOM walk and *every formula after it on the page* stays raw. One unbalanced
  brace can blank out a whole tutorial.

It also reports LaTeX left sitting in ordinary prose, which is what a `$$\nx = 1\n$$` or a
`$ A $` turns into: Documenter never reads those as math.

### `check-rendered-pages.mjs`

Runtime, and the only check that can catch the outage above. Serves `docs/build` over HTTP, loads
each page in headless Chromium, and asserts:

- no uncaught exceptions, and no console errors matching `is not a function`,
  `Mismatched anonymous define`, `Uncaught`, `documenter.js` or `require.js`
- pages containing math end up with `.katex` nodes, and no raw LaTeX is visible in the prose
- pages containing Julia end up with `hljs-*` spans

Google Analytics and the Gemini search widget are routinely blocked in CI, so failures from those
origins are ignored on purpose — a flaky check is worse than no check.

Both scripts exit 0 with a `SKIP` message when their dependencies are missing, so `make docs` stays
usable without node tooling.

## Related guards

- `check_amd_conflicts` in `docs/make.jl` fails the build if any page loads require.js *and* a
  third-party `<script type="module">`. That is the specific mechanism above, caught deterministically
  and with no dependencies.
- `examples/math_lint.jl` runs during `make examples` and rejects math that cannot render, reporting
  the notebook and line so authors get the error at the source rather than in generated output.
