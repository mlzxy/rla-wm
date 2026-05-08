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

      // Get raw markdown text.  Use innerHTML (not textContent) so that
      // inline HTML like <span class="annot"> survives into marked.
      var raw = container.innerHTML.trim();

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

      // Activate any .annot elements inside the rendered markdown.
      // initAnnotations() runs before initMarkdown(), so new <span>s
      // injected here were never processed.
      if (typeof ScrollTrigger !== 'undefined') {
        $$('.annot', container).forEach(function (el) {
          if (el.classList.contains('hero-annot')) return;
          if (el.closest('.scrolly')) return;
          if (el.dataset.color) el.style.setProperty('--annot-bg', el.dataset.color);
          ScrollTrigger.create({
            trigger: el, start: 'top 85%',
            onEnter:     function () { el.classList.add('is-active'); },
            onLeaveBack: function () { el.classList.remove('is-active'); },
          });
        });
      }

      // Mark as rendered
      container.dataset.markdownRendered = 'true';
    });
  }

  /* ============================================================
     Markdown-table → .results-table converter
     Finds [data-table] elements, parses markdown pipe-table syntax,
     and replaces the source element with a styled <table>.

     Special markers (processed BEFORE marked parses them):
       *value  → cell gets class "best"      (asterisk stripped)
       ~value  → cell gets class "second"    (tilde stripped)
       ---     → row  gets class "row-divider" (entire row, single cell)

     Auto-detected:
       Headers containing ↑ or ↓ → class "num"
       **text**                  → <strong>text</strong> (standard markdown)

     Usage:
       <div class="markdown-table reveal" data-table
            data-caption="Optional caption text.">
     | Method | Acc. ↑ | FLOPs ↓ |
     |--------|--------|---------|
     | Baseline-A | 71.2 | 9.8B |
     | Baseline-B | ~73.4 | 12.1B |
     | --- |
     | **Ours** | *76.1 | *4.8B |
       </div>
     ============================================================ */
  function initMarkdownTables(root) {
    var containers = root
      ? [root]
      : $$('.markdown-table[data-table]');

    containers.forEach(function (container) {
      if (container.dataset.tableRendered === 'true') return;

      var raw = container.textContent.trim();
      if (!raw) return;

      // ---------- Extract caption (anything after the last pipe-table line) ----------
      // Split into lines; the last non-empty, non-pipe line(s) become the caption.
      var lines = raw.split(/\r?\n/);
      var captionLines = [];
      while (lines.length && !/\|/.test(lines[lines.length - 1].trim())) {
        captionLines.unshift(lines.pop().trim());
      }
      var captionText = captionLines.join(' ').trim();
      // Rejoin remaining table rows
      raw = lines.join('\n').trim();
      if (!raw) return;

      // ---------- Pre-process: protect special markers ----------
      // Replace *value and ~value with marker tokens that survive
      // marked's parsing, then restore them after.
      var BEST  = 'XXBESTXX';
      var SCND  = 'XXSCNDXX';
      var DIV   = 'XXDIVXX';

      // Protect --- divider rows (a row whose only cell is exactly ---)
      // Markdown: | --- |  → after trim becomes "---"
      // We replace standalone "---" rows with a token row.
      raw = raw.replace(/^\s*\|\s*---\s*\|\s*$/gm, '| ' + DIV + ' |');
      // Also handle "--- |" or "| ---" variants
      raw = raw.replace(/^\s*\|\s*---\s*$/gm, '| ' + DIV);
      raw = raw.replace(/^\s*---\s*\|\s*$/gm, DIV + ' |');

      // Protect *value and ~value inside table cells
      // Use lookahead (?=\|) so the shared pipe separator is not consumed,
      // allowing adjacent cells to also be matched.
      raw = raw.replace(/\|\s*\*(?!\*)\s*([^|\n]+?)\s*(?=\|)/g, '| ' + BEST + ' $1 ');
      raw = raw.replace(/\|\s*~\s*([^|\n]+?)\s*(?=\|)/g, '| ' + SCND + ' $1 ');

      // ---------- Parse with marked ----------
      var renderer = createRenderer();
      var html;
      try {
        html = marked.parse(raw, { renderer: renderer, breaks: false });
      } catch (e) {
        console.error('[markdown.js] Table parse failed:', e.message);
        return;
      }

      // ---------- Post-process: convert <table> → .results-table ----------
      // Replace <table> with our styled version
      html = html.replace(/<table>/g, '<table class="results-table">');

      // Process header cells: detect arrows → add .num
      html = html.replace(/<th(.*?)>(.*?)<\/th>/g, function (m, attrs, content) {
        var cls = attrs.includes('class="') ? attrs : attrs + ' class=""';
        if (/[↑↓]/.test(content)) {
          cls = cls.replace(/class="([^"]*)"/, 'class="$1 num"');
        }
        return '<th' + cls + '>' + content + '</th>';
      });

      // Process body cells: restore BEST/SCND tokens
      html = html.replace(/<td(.*?)>(.*?)<\/td>/g, function (m, attrs, content) {
        var cls = attrs.includes('class="') ? attrs : attrs + ' class=""';
        if (content.indexOf(BEST) !== -1) {
          content = content.replace(new RegExp(BEST + '\\s*', 'g'), '');
          cls = cls.replace(/class="([^"]*)"/, 'class="$1 num best"');
        } else if (content.indexOf(SCND) !== -1) {
          content = content.replace(new RegExp(SCND + '\\s*', 'g'), '');
          cls = cls.replace(/class="([^"]*)"/, 'class="$1 num second"');
        } else {
          // Auto-detect numeric cells: if content is a pure number (or B/M)
          if (/^[\d.]+[BMKk]?$/.test(content.trim()) || /^[↑↓]/.test(content.trim())) {
            cls = cls.replace(/class="([^"]*)"/, 'class="$1 num"');
          }
        }
        return '<td' + cls + '>' + content + '</td>';
      });

      // Wrap table (+ optional caption) in .results-table-wrapper
      var captionHtml = '';
      if (captionText) {
        captionHtml = '<p class="results-table-caption">' + captionText + '</p>';
      }
      html = '<div class="results-table-wrapper">' + html + captionHtml + '</div>';

      // Apply data-width (sets --table-max-width on the wrapper)
      var tableWidth = container.dataset.width;
      if (tableWidth) {
        html = html.replace(
          '<div class="results-table-wrapper">',
          '<div class="results-table-wrapper" style="--table-max-width:' + tableWidth + ';">'
        );
      }

      // ---------- Replace source element ----------
      var tmp = document.createElement('div');
      tmp.innerHTML = html;
      var tableWrapper = tmp.firstChild;  // .results-table-wrapper
      var table = tableWrapper && tableWrapper.querySelector('table.results-table');

      if (table) {
        // --- Handle divider row via DOM ---
        var rows = table.querySelectorAll('tr');
        for (var i = 0; i < rows.length; i++) {
          if (rows[i].textContent.indexOf(DIV) !== -1) {
            var nextRow = rows[i + 1];
            if (nextRow) nextRow.classList.add('row-divider');
            rows[i].remove();
            break;
          }
        }

        // Copy classes from source container (e.g. "reveal") to the wrapper
        if (container.className) {
          var srcClasses = container.className.replace(/\bmarkdown-table\b/g, '').trim();
          if (srcClasses) {
            tableWrapper.className = (tableWrapper.className + ' ' + srcClasses).trim();
          }
        }

        container.parentNode.insertBefore(tableWrapper, container);
        container.remove();

        // Apply reveal animation to the wrapper (not just the table)
        if (tableWrapper.classList.contains('reveal') &&
            typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined') {
          gsap.to(tableWrapper, {
            opacity: 1, y: 0, duration: 0.8, ease: 'power2.out',
            scrollTrigger: {
              trigger: tableWrapper,
              start: 'top 85%',
              end: 'bottom 15%',
              toggleActions: 'play none none reverse',
            },
          });
        }
      }
    });
  }

  // Expose globally so paper.js can call it
  window.initMarkdown = initMarkdown;
  window.initMarkdownTables = initMarkdownTables;

  // Also expose a tiny namespace
  window.PaperMarkdown = {
    init: initMarkdown,
    initMarkdownTables: initMarkdownTables,
    highlightCodeInElement: highlightCodeInElement,
    renderMathInBlock: renderMathInBlock,
  };
})();
