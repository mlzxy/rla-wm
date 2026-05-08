/* ============================================================
   themes.js — Theme manager for the paper-site library

   Manages the `data-theme` attribute on <html>, persists the
   choice to localStorage, and exposes a clean API.

   BUILT-IN THEMES
     light  — modern clean look (default)
     dark   — dark background, light text
     paper  — LaTeX-inspired minimal academic look

   USAGE (programmatic)
     PaperThemes.set('dark');           // switch to dark
     PaperThemes.next();                // cycle to next theme
     PaperThemes.current();             // → 'light' | 'dark' | 'paper'
     PaperThemes.list();                // → ['light', 'dark', 'paper']
     PaperThemes.register('ocean', {    // add your own theme
       label: 'Ocean',
       apply: function() { … },
     });

   USAGE (HTML)
     <!-- Button that cycles themes -->
     <button onclick="PaperThemes.next()">Toggle theme</button>

     <!-- Buttons for specific themes -->
     <button onclick="PaperThemes.set('light')">Light</button>
     <button onclick="PaperThemes.set('dark')">Dark</button>
     <button onclick="PaperThemes.set('paper')">Paper</button>

   HOW TO DESIGN YOUR OWN THEME
   ────────────────────────────
   1. In themes.css, add:
        [data-theme="my-theme"] {
          --fg: …;
          --bg: …;
          … other CSS variables …
        }
   2. Register it (so the switcher knows about it):
        PaperThemes.register('my-theme', { label: 'My Theme' });
   3. Optionally provide an `apply` callback for imperative
      changes (e.g. swapping a logo, changing a font CDN):

        PaperThemes.register('my-theme', {
          label: 'My Theme',
          apply: function() {
            document.body.classList.add('my-custom-class');
          },
          remove: function() {
            document.body.classList.remove('my-custom-class');
          }
        });
   ============================================================ */
(function () {
  'use strict';

  var STORAGE_KEY = 'paper-site-theme';
  var DEFAULT_THEME = 'light';

  // Internal registry.  Each entry: { label, apply?, remove? }
  var registry = {
    light: { label: 'Light' },
    dark:  { label: 'Dark' },
    paper: { label: 'Paper' },
  };

  var currentTheme = DEFAULT_THEME;

  /* ---------- Apply a theme ---------- */
  function applyTheme(name) {
    var entry = registry[name];
    if (!entry) {
      console.warn('[themes.js] Unknown theme "' + name + '". Available: ' + Object.keys(registry).join(', '));
      return;
    }

    // Remove previous theme's imperative effects
    var prev = registry[currentTheme];
    if (prev && typeof prev.remove === 'function') {
      try { prev.remove(); } catch (e) { console.warn('[themes.js] remove() failed for ' + currentTheme, e); }
    }

    // Set the data-theme attribute on <html>
    document.documentElement.setAttribute('data-theme', name);

    // Run the new theme's imperative apply hook
    if (typeof entry.apply === 'function') {
      try { entry.apply(); } catch (e) { console.warn('[themes.js] apply() failed for ' + name, e); }
    }

    currentTheme = name;
  }

  /* ---------- Persist ---------- */
  function persist(name) {
    try {
      localStorage.setItem(STORAGE_KEY, name);
    } catch (e) {
      // localStorage may be unavailable (private browsing, quota, etc.)
    }
  }

  /* ---------- Public API ---------- */
  function setTheme(name) {
    applyTheme(name);
    persist(name);
  }

  function getTheme() {
    return currentTheme;
  }

  function getThemeLabel() {
    var entry = registry[currentTheme];
    return entry ? entry.label : currentTheme;
  }

  function nextTheme() {
    var names = Object.keys(registry);
    var idx   = names.indexOf(currentTheme);
    var next  = names[(idx + 1) % names.length];
    setTheme(next);
    return next;
  }

  function listThemes() {
    return Object.keys(registry);
  }

  function registerTheme(name, opts) {
    if (!name || typeof name !== 'string') {
      console.error('[themes.js] register() requires a string name.');
      return;
    }
    registry[name] = {
      label:  (opts && opts.label)  || name,
      apply:  (opts && opts.apply)  || null,
      remove: (opts && opts.remove) || null,
    };
  }

  /* ---------- Boot: restore saved theme ---------- */
  function boot() {
    var saved = null;
    try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) { /* ignore */ }

    // Validate saved theme exists in registry
    if (saved && registry[saved]) {
      applyTheme(saved);
    } else {
      // No saved theme or invalid — use default (data-theme not set,
      // so :root variables apply = light theme)
      currentTheme = DEFAULT_THEME;
    }
  }

  // Run immediately so the correct theme is set before first paint.
  // We only apply if a saved theme exists AND differs from default.
  // The default (light) is handled by :root — no data-theme needed.
  (function () {
    var saved = null;
    try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) { /* ignore */ }
    if (saved && registry[saved] && saved !== DEFAULT_THEME) {
      document.documentElement.setAttribute('data-theme', saved);
      currentTheme = saved;
      // Run apply hook if any
      var entry = registry[saved];
      if (entry && typeof entry.apply === 'function') {
        try { entry.apply(); } catch (e) { /* ignore */ }
      }
    }
  })();

  // Expose
  window.PaperThemes = {
    set:      setTheme,
    get:      getTheme,
    label:    getThemeLabel,
    next:     nextTheme,
    list:     listThemes,
    register: registerTheme,
  };
})();
