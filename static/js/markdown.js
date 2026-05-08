/* ============================================================
   markdown.js — Markdown, KaTeX & code-highlight init
   Part of the paper-site component library.

   Auto-discovers .markdown-block[data-markdown] elements and:
     1. Renders Markdown → HTML via marked
     2. Renders KaTeX math ($...$ and $$...$$)
     3. Applies highlight.js to <pre><code> blocks

   Requires (loaded before this script):
     - marked       (https://cdn.jsdelivr.net/npm/marked/marked.min.js)
     - katex        (https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js)
     - katex auto-render extension
     - highlight.js (https://cdn.jsdelivr.net/gh/highlightjs/cdn-release/build/highlight.min.js)

   Usage in HTML:
     <div class="markdown-block" data-markdown>
       # My Heading
       Math: $E=mc^2$
       ```python
       def hello():
           print("world")
       ```
     </div>

   Optional attributes on the container:
     data-markdown-theme="dark|light" — override the code-highlight theme
   ============================================================ */
(function () {
  'use strict';

  /* ---------- Helpers ---------- */
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  /* ---------- Math protection ----------
     Markdown parsers interpret underscores inside LaTeX (e.g.
     \mathcal{L}_{\text{total}}) as italic markers, corrupting the
     math. We replace $...$ and $$...$$ with safe ASCII placeholders
     before parsing, then restore them afterwards.

     Delimiters use only [A-Za-z0-9_] so they survive any HTML/text
     pipeline unescaped.
     --------------------------------------------------------------- */
  // NOTE: placeholders must use only alphanumerics — markdown interprets
  // double underscores as **bold**, which would corrupt the tokens.
  var MATH_D = 'XXMATHDX';    // display math placeholder prefix
  var MATH_I = 'XXMATHIX';    // inline  math placeholder prefix
  var MATH_E = 'XEND';        // placeholder suffix

  function protectMath(text) {
    var displayBlocks = [];
    var inlineBlocks = [];

    // Protect display math first ($$ ... $$)
    text = text.replace(/\$\$([\s\S]*?)\$\$/g, function (_, math) {
      displayBlocks.push(math.trim());
      return MATH_D + (displayBlocks.length - 1) + MATH_E;
    });

    // Protect inline math ($ ... $) — avoid matching $$ remnants
    text = text.replace(/\$(.+?)\$/g, function (_, math) {
      inlineBlocks.push(math.trim());
      return MATH_I + (inlineBlocks.length - 1) + MATH_E;
    });

    return { text: text, display: displayBlocks, inline: inlineBlocks };
  }

  function restoreMath(html, display, inline) {
    // Restore display math
    display.forEach(function (math, i) {
      html = html.split(MATH_D + i + MATH_E).join('$$' + math + '$$');
    });
    // Restore inline math
    inline.forEach(function (math, i) {
      html = html.split(MATH_I + i + MATH_E).join('$' + math + '$');
    });
    return html;
  }

  /* ---------- Markdown parser setup ---------- */
  const depsOk = () => {
    const missing = [];
    if (typeof marked === 'undefined' || typeof marked.parse !== 'function') missing.push('marked');
    if (typeof katex  === 'undefined') missing.push('katex');
    if (typeof hljs   === 'undefined') missing.push('highlight.js');
    if (missing.length) {
      console.warn('[markdown.js] Missing dependencies: ' + missing.join(', ') +
        '. Markdown blocks will not render. Add the CDN <script> tags before this file.');
      return false;
    }
    return true;
  };

  /* ---------- Markdown parser setup ---------- */
  function createRenderer() {
    const renderer = new marked.Renderer();

    // Let KaTeX handle math — pass $...$ and $$...$$ through as raw HTML.
    // marked's default escapes $, so we need to preserve them.
    // NOTE: marked may call these renderers with either positional string args
    // (legacy / v12 default) or a token object (new tokenized API). Support both.
    renderer.codespan = function (arg) {
      const text = (typeof arg === 'string') ? arg : (arg && arg.text) || '';
      // If it looks like inline math ($...$), pass through to KaTeX
      if (/^\$.+\$$/.test(text.trim()) || /^\\\(.+\\\)$/.test(text.trim())) {
        return '<code class="math-inline">' + text + '</code>';
      }
      return '<code>' + text + '</code>';
    };

    renderer.code = function (arg, infoString) {
      let lang, text;
      if (typeof arg === 'string') {
        text = arg;
        lang = infoString || '';
      } else {
        text = (arg && arg.text) || '';
        lang = (arg && arg.lang) || '';
      }
      // If it's a math block ($$...$$), let KaTeX handle it
      if (lang === 'math' || lang === 'latex' || lang === 'katex') {
        try {
          return '<p class="katex-block">' + katex.renderToString(text, { displayMode: true, throwOnError: false }) + '</p>';
        } catch (e) {
          return '<pre><code>' + text + '</code></pre>';
        }
      }
      // Regular code block — highlight.js will handle later
      const cls = lang ? ' class="language-' + lang + '"' : '';
      return '<pre><code' + cls + '>' + text + '</code></pre>';
    };

    // Pass through raw HTML (needed for KaTeX $$...$$ blocks in text)
    renderer.html = function (arg) {
      return (typeof arg === 'string') ? arg : (arg && arg.text) || '';
    };

    return renderer;
  }

  /* ---------- KaTeX auto-render ---------- */
  function renderMathInBlock(el) {
    if (typeof window.renderMathInElement === 'function') {
      try {
        window.renderMathInElement(el, {
          delimiters: [
            { left: '$$',  right: '$$',  display: true  },
            { left: '$',   right: '$',   display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true  },
          ],
          throwOnError: false,
        });
      } catch (e) {
        console.warn('[markdown.js] KaTeX auto-render failed:', e.message);
      }
    } else {
      console.warn('[markdown.js] KaTeX auto-render extension (renderMathInElement) not available.');
    }
  }

  /* ---------- Highlight code ---------- */
  function highlightCodeInElement(el) {
    if (typeof hljs === 'undefined') return;
    $$('pre code', el).forEach(function (block) {
      try {
        hljs.highlightElement(block);
      } catch (e) {
        // language not supported — that's fine
      }
    });
  }

  /* ---------- Main init ---------- */
  function initMarkdown(root) {
    if (!depsOk()) return;

    var containers = root
      ? [root]
      : $$('.markdown-block[data-markdown]');

    if (!containers.length) return;

    containers.forEach(function (container) {
      // Skip already-rendered blocks
      if (container.dataset.markdownRendered === 'true') return;

      // Get raw markdown text.  Use .trim() — the outer whitespace is
      // just HTML indentation and is never meaningful markdown content.
      var raw = container.textContent.trim();

      if (!raw) return;

      // Protect LaTeX math from being corrupted by markdown parsing
      // (underscores in LaTeX like \mathcal{L}_{...} trigger italic).
      var protected_ = protectMath(raw);
      raw = protected_.text;

      // Render markdown → HTML.  Inline mode (no <p>/<h1>/… wrappers) is
      // used when the container is a <span> or explicitly opts in via
      // data-markdown="inline". This makes it valid to embed markdown
      // inside phrasing-only contexts.
      try {
        var renderer = createRenderer();
        var inline = container.tagName === 'SPAN' ||
                     container.dataset.markdown === 'inline';
        var html = inline
          ? marked.parseInline(raw, { renderer: renderer })
          : marked.parse(raw, { renderer: renderer, breaks: false });

        // Restore math expressions
        html = restoreMath(html, protected_.display, protected_.inline);

        container.innerHTML = html;
      } catch (e) {
        console.error('[markdown.js] Markdown parse failed:', e.message);
        return;
      }

      // Apply KaTeX to math expressions
      renderMathInBlock(container);

      // Apply code highlighting
      highlightCodeInElement(container);

      // Mark as rendered
      container.dataset.markdownRendered = 'true';
    });
  }

  // Expose globally so paper.js can call it
  window.initMarkdown = initMarkdown;

  // Also expose a tiny namespace
  window.PaperMarkdown = {
    init: initMarkdown,
    highlightCodeInElement: highlightCodeInElement,
    renderMathInBlock: renderMathInBlock,
  };
})();
