# Paper Site Template

A minimal, declarative template for paper / project websites. Drop content into `template.html`, and the auto-init library (`paper.js` + `paper.css`) wires up scroll animations, carousels, annotations, tooltips, comparison sliders, code blocks, and more — no per-page JavaScript required.

---

## Quick start

```bash
# From the repo root, run a local server (required for the Inkscape SVG fetch)
python3 -m http.server 8000

# then open
open http://localhost:8000/template.html
```

Opening `template.html` directly via `file://` works for everything *except* the external Inkscape SVG (CORS blocks `fetch()`); that component falls back to a plain `<img>` automatically.

---

## File layout

```
template.html         # your page — class names + data-* attributes only
static/
├── css/
│   └── paper.css     # the component library (styles)
├── js/
│   └── paper.js      # the component library (auto-init)
└── images/           # your figures / SVGs / videos
```

To author a page, **only edit `template.html`**. Everything else is library code.

---

## How it works

`paper.js` registers a `DOMContentLoaded` handler that scans the document and initialises every component it finds — by class name (e.g. `.reveal`, `.scrolly`, `.compare`) and `data-*` attributes (e.g. `data-embla`, `data-svg-src`, `data-explorer`'s embedded JSON). Nothing is wired by hand.

Each `init*` function is also exposed under `window.Paper` if you ever need to re-run a module after manually injecting markup:

```js
window.Paper.initCodeBlocks();
```

---

## Components

### Reveal-on-scroll

Any element with class `reveal` fades + translates in when it enters the viewport, and reverses when scrolled back past.

```html
<h2 class="reveal">Method</h2>
<p class="reveal">…</p>
```

### Hero with delay-animated annotations + side cards

The hero title can contain `<span class="hero-annot">` phrases. After a delay (no scroll), each phrase gets a hand-drawn underline (or other Rough Notation mark) and its paired side card slides in from the chosen side.

```html
<h1 class="reveal">
  Your Paper Title:
  <span class="hero-annot" data-annot-id="a"
        data-color="#ef4444" data-delay="1.0" data-side="right">
    First Key Idea
  </span>
  meets
  <span class="hero-annot" data-annot-id="b"
        data-color="#3b82f6" data-delay="2.6" data-side="left">
    Second Key Idea
  </span>
</h1>

<aside class="side-card" data-annot-id="a"
       style="top:26%; right:4%;">
  <div class="kicker">First Key Idea</div>
  <div class="body"><p>…</p></div>
</aside>
```

| Attribute on `.hero-annot` | Default | Notes |
|---|---|---|
| `data-annot-id` | — | Pairs the phrase with `<aside data-annot-id>` |
| `data-type` | `underline` | `underline` / `highlight` / `circle` / any Rough-Notation type |
| `data-color` | `#ef4444` | Stroke colour |
| `data-delay` | `1.0` | Seconds before this annotation fires |
| `data-side` | `right` | `left` / `right` — direction the card flies in from |
| `data-stroke`, `data-padding`, `data-duration` | `2.5` / `3` / `700` | Rough Notation tuning |

The side card's size, padding, accent colour, and font size are all CSS variables — override per card with inline `style="--card-width:320px; --card-font-size:15px;"`. See `:root .side-card` in `paper.css` for the full list.

### Floating background geometry

Decorative shapes scattered around the hero, each with its own slow drift. The wrapper is `aria-hidden` and `pointer-events: none`, so it never interferes with the page.

```html
<section class="hero">
  <div class="hero-bg" aria-hidden="true">
    <!-- built-in outlined shapes (currentColor; .outline gives them
         the fill:none + stroke styling) -->
    <svg class="shape outline shape-1" viewBox="0 0 100 100">…</svg>
    <svg class="shape outline shape-2" viewBox="0 0 100 100">…</svg>
    …
    <!-- your own SVG (no .outline → keeps your fills / strokes) -->
    <svg class="shape shape-10" viewBox="0 0 100 100">
      <polygon points="…" fill="#10b981"/>
    </svg>
    <!-- or as an <img> -->
    <img class="shape shape-9" src="static/images/your-mark.svg" alt="">
  </div>
  …
</section>
```

**Slots `.shape-1` … `.shape-10`** — each defines a position, size, colour, opacity, and animation. Pick whichever slot matches where you want the shape; swap the inner SVG freely without touching CSS.

| Class on the `<svg>`/`<img>` | Effect |
|---|---|
| `shape` | required — applies absolute positioning + animation |
| `shape-N` | required — picks one of 10 positioning slots |
| `outline` | (optional) makes inner `circle` / `polygon` / `rect` / `line` / `path` use `fill: none; stroke: currentColor;` so they pick up the slot's `color` |
| `dotgrid` | (optional) makes inner `circle`s filled (used by the dot-grid shape) |
| `circle.dot` | inside an `outline` shape, marks individual circles as filled (used by the node-graph) |

For a custom SVG with your own fills/strokes, just *omit* `.outline`. For an `<img>`, the slot's `color` is irrelevant — the rendered SVG/PNG uses its own colours.

**Per-instance size / opacity overrides.** Each slot exposes three CSS variables you can override inline (no CSS edit required):

| Variable | What it does | Default |
|---|---|---|
| `--shape-w` | Width  | per slot (`60px` if unset) |
| `--shape-h` | Height | per slot (`60px` if unset) |
| `--shape-opacity` | 0–1 visibility | `0.18` (varies per slot) |

```html
<!-- A landscape SVG: stretch the slot to match its aspect, bump opacity -->
<img class="shape shape-10"
     style="--shape-w:200px; --shape-h:60px; --shape-opacity:.55;"
     src="static/images/your-mark.svg" alt="">
```

To change *position* or *animation*, edit the slot's rule in `paper.css`:

```css
.hero-bg .shape-3 {
  top: 42%; left: 3%;
  width: 56px; height: 56px;
  color: var(--accent-3);     /* used by outlined shapes via currentColor */
  opacity: 0.18;
  animation: float-c 26s ease-in-out infinite;
}
```

Animations (`float-a` … `float-j`) are simple `translate + rotate` keyframes; pick whichever drift direction looks right. All animations are disabled under `prefers-reduced-motion`. On mobile (`≤720px`), opacity drops to `0.11` and four slots (3, 4, 7, 10) hide entirely to reduce clutter.

### Author list + affiliations

```html
<div class="author-list">
  <span class="author equal"><a href="#">A. Anonymous</a><sup>1</sup></span>
  <span class="author equal"><a href="#">B. Anonymous</a><sup>1,2</sup></span>
  <span class="author"><a href="#">C. Anonymous</a><sup>2</sup></span>
</div>

<div class="affiliations">
  <span><sup>1</sup><img class="affil-logo" src="static/images/logo-mit.png" alt="">MIT</span>
  <span><sup>2</sup><img class="affil-logo" src="static/images/logo-stanford.png" alt="">Stanford University</span>
</div>

<p class="eq-note">* Equal contribution</p>
```

- `.equal` adds an asterisk after the name.
- `.affil-logo` is optional. Drop the `<img>` to show just text. Logos are constrained to `height: 24px` (max-width 80px) and align to the institution name baseline.
- Affiliations use a readable dark-gray (not muted) at 14.5px so they're easy to see.

### Conference / venue line

```html
<!-- Conference / venue line. Delete this <p> for an arXiv preprint. -->
<p class="muted">NeurIPS 2026 (under review)</p>
```

Just remove the line (or comment it out) for arXiv preprints — there's no special class to toggle.

### Action buttons

```html
<div class="actions">
  <a class="btn-link arxiv" href="https://arxiv.org/abs/0000.00000" target="_blank" rel="noopener">
    <i class="ai ai-arxiv" aria-hidden="true"></i> arXiv
  </a>
  <a class="btn-link github" href="https://github.com/your-org/your-repo" target="_blank" rel="noopener">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="…"/></svg> Code
  </a>
  <a class="btn-link linkedin" href="https://www.linkedin.com/in/your-handle" target="_blank" rel="noopener">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="…"/></svg> LinkedIn
  </a>
</div>
```

Outline buttons branded in each service's signature colour:

| Variant class | Colour | Icon source |
|---|---|---|
| `.btn-link.arxiv`    | `#B31B1B` (Cornell red) | [academicons](https://jpswalsh.github.io/academicons/) `ai ai-arxiv` |
| `.btn-link.github`   | `#181717`               | inline SVG (simpleicons GitHub mark) |
| `.btn-link.linkedin` | `#0A66C2`               | inline SVG (simpleicons LinkedIn mark) |

The border, label, and icon all use the brand colour; on hover the button fills in. To add a new variant, add a class:

```css
.btn-link.openreview { --btn-color: #8c1b13; }
.btn-link.semantic   { --btn-color: #2e6cb6; }
```

Academicons (loaded from CDN in the `<head>`) gives you `ai-openreview`, `ai-google-scholar`, `ai-orcid`, `ai-semantic-scholar`, `ai-researchgate`, etc. — see their [cheatsheet](https://jpswalsh.github.io/academicons/).

### Inline annotations + tooltips

Wrap any in-text phrase in `<span class="annot">` to give it a Rough Notation mark on scroll-in. Add `data-tippy="…"` for a hover tooltip.

```html
<span class="annot" data-color="#ef4444"
      data-tippy="A short tooltip describing this phrase.">
  a new method
</span>
```

| Attribute | Default | Options |
|---|---|---|
| `data-type` | `underline` | `underline` / `highlight` / `box` / `circle` / `bracket` |
| `data-color` | `#3b82f6` | Any CSS colour |
| `data-stroke`, `data-padding`, `data-duration` | `2` / `2` / `600` | |

Annotations inside the scrollytelling section (`.scrolly`) are managed automatically — they all draw together when the section enters view, and gain a yellow-highlight `.is-active` class while their bound slide is active.

### Callouts

```html
<div class="callout" data-kind="tldr">
  <span class="callout-label">TL;DR</span>
  …
</div>
```

`data-kind`: `tldr` (blue) · `note` (gray) · `warning` (amber) · `key` (red).

### Scrollytelling

A pinned section where scroll progress drives an active-slide index. The text column on the left is bound to slides on the right; each `[data-slide]` paragraph lights up when its slide is active. Hovering a paragraph (desktop) jumps to that slide.

```html
<div class="scrolly">
  <div class="stage">
    <div class="text-col">
      <p data-slide="0">Step 1. <span class="annot">extract features</span> …</p>
      <p data-slide="1">Step 2. …</p>
      <p data-slide="2">Step 3. …</p>
    </div>
    <div class="figure-col">
      <div class="figure-frame">
        <div class="figure-slide is-active">…</div>
        <div class="figure-slide">…</div>
        <div class="figure-slide">…</div>
        <!-- progress dots are auto-injected -->
      </div>
    </div>
  </div>
</div>
```

On desktop the whole `.scrolly` is pinned and scroll-scrubbed. On mobile (`≤720px`) the figure becomes CSS-sticky at the top, and per-paragraph `ScrollTrigger`s switch the active slide as paragraphs cross mid-viewport.

### Image + text split

```html
<div class="split" data-layout="img-left">  <!-- or "img-right" -->
  <div class="split-media">
    <img src="…" alt=""> <!-- or a video, or a placeholder -->
  </div>
  <div class="split-text">
    <h3>Component One</h3>
    <p>…</p>
  </div>
</div>
```

Two columns on desktop, single column with image on top on mobile.

### Figure grid

```html
<div class="figure-grid" style="--figure-grid-cols:3;">
  <figure>
    <img src="…" alt="">
    <figcaption>Caption.</figcaption>
  </figure>
  …
</div>
```

`--figure-grid-cols` controls the column count (defaults to `3`). Reduces to 2 columns on mobile. `<figure>` children can hold `<img>`, `<video>`, or a `.placeholder` div.

### Inline SVG diagram

Any `<svg class="svg-diagram">` containing `<path class="flow">` lines gets a stroke-dashoffset draw animation when the SVG enters view. Use `.flow.alt` for the secondary colour.

```html
<svg class="svg-diagram" viewBox="0 0 720 280">
  <rect class="node"  x="20" y="110" width="120" height="60" rx="8"/>
  <text class="node-label" x="80" y="145" text-anchor="middle">Input</text>
  <path class="flow" d="M 140 140 C 170 140, 170 70, 200 70"/>
  <path class="flow alt" d="…"/>
</svg>
```

### External Inkscape SVG

```html
<div class="svg-holder" data-svg-src="static/images/wm.svg"></div>
```

Fetches the file at runtime, injects it inline, fades in each top-level child with a stagger, and additionally draws every stroked `<path>` via `stroke-dashoffset`. Safe with Inkscape's `transform="..."` attributes — only `opacity` is animated on children to avoid blowing up SVG layouts.

| Attribute | Default | Notes |
|---|---|---|
| `data-svg-src` | — | Path to the SVG file |
| `data-svg-stagger` | `0.025` | Seconds between siblings |
| `data-svg-fade` | `0.6` | Fade duration |
| `data-svg-draw` | `1.2` | Stroke-draw duration |
| `data-svg-start` | `top 75%` | ScrollTrigger `start` |

Falls back to a static `<img>` if `fetch()` fails (e.g. opening via `file://`).

### Embla carousel

```html
<div class="embla" data-embla data-loop="true">
  <div class="embla__container">
    <div class="embla__slide"><div class="card">…</div></div>
    <div class="embla__slide"><div class="card">…</div></div>
    <div class="embla__slide"><div class="card">…</div></div>
  </div>
</div>
```

Prev / Next buttons and pagination dots are auto-injected after the carousel.

| Attribute | Default | Notes |
|---|---|---|
| `data-embla` | — | Marker (must be present) |
| `data-loop` | `false` | Set `"true"` to loop |
| `data-align` | `start` | `start` / `center` / `end` |
| `data-drag-free` | `false` | Free-scroll vs. snap |

### Comparison slider (before / after)

```html
<div class="compare"
     data-before="path/to/before.jpg"
     data-after="path/to/after.jpg"
     data-before-label="Input"
     data-after-label="Ours"></div>
```

Drag (or touch-drag) to wipe between the two images. The `before` image is `clip-path`'d so the `after` image is revealed underneath as you slide.

### Explorer (thumb strip + grid sub-component)

A configurable component for showing many figure variants. The entire DOM is built from an embedded JSON config:

```html
<div class="explorer">
  <script type="application/json">
[
  {
    "thumb":  { "label": "A", "color": "#ef4444" },
    "title":  "Configuration A — irregular grid, row labels, cell-by-cell",
    "rowLabels": ["Input", "Ours"],
    "colLabels": null,
    "animation": { "mode": "cell", "stagger": 0.08 },
    "rows": [
      { "cells": [
        { "kind": "color", "color": "#ef4444", "caption": "orig" },
        { "kind": "image", "src": "static/images/x.png", "caption": "x" }
      ] },
      { "cells": [
        { "kind": "sequence",
          "frames": [{"color":"#ef4444"},{"color":"#3b82f6"}],
          "interval": 350,
          "caption": "auto-play" }
      ] }
    ]
  }
]
  </script>
</div>
```

**Cell kinds**

| `kind` | Required fields | Notes |
|---|---|---|
| `color` | `color` | Solid colour tile (great for placeholders); optional `label` |
| `image` | `src` | Static image |
| `video` | `src` | Auto-play, loop, muted, inline |
| `sequence` | `frames: [{color\|src}, …]`, `interval` | Cycles through frames on a timer |

**Animation modes** (`animation.mode`)

| Mode | Behaviour |
|---|---|
| `"cell"` | Stagger by DOM order |
| `"row"` | All cells in row 0, then row 1, … |
| `"col"` | Column 0 across rows, then column 1, … |
| `[2, 0, 3, 1]` | Explicit indices into the flat cell list |

`stagger` is in seconds. Rows can have unequal cell counts; any cell not referenced by an explicit index list still appears, just at the end.

### Code block with copy button

```html
<div class="code-block" data-lang="BibTeX">
<pre>@inproceedings{anonymous2026paper,
  title  = {…},
  author = {…},
  year   = {2026},
}</pre>
</div>
```

A "Copy" button is auto-injected and copies the `<pre>` contents to the clipboard via the modern API (with an `execCommand` fallback). Set `data-lang` to label the block.

### Results table

```html
<table class="results-table">
  <thead>
    <tr><th>Method</th><th class="num">Acc. ↑</th><th class="num">FLOPs ↓</th></tr>
  </thead>
  <tbody>
    <tr><td>Baseline-A</td>            <td class="num">71.2</td><td class="num">9.8B</td></tr>
    <tr><td>Baseline-B</td>            <td class="num second">73.4</td><td class="num">12.1B</td></tr>
    <tr class="row-divider"><td>Prior SOTA</td><td class="num">74.0</td><td class="num">15.6B</td></tr>
    <tr><td><strong>Ours</strong></td> <td class="num best">76.1</td><td class="num best">4.8B</td></tr>
  </tbody>
  <caption>Top-1 accuracy on the held-out test set.</caption>
</table>
```

| Class | Effect |
|---|---|
| `.num` | Right-align, tabular numerals |
| `.best` | Bold + accent colour |
| `.second` | Underlined (runner-up) |
| `tr.row-divider` | Heavier top border (separates model families) |

### Plain video frame

```html
<video class="paper-video" autoplay loop muted playsinline src="…"></video>
```

CSS-only. Rounded, bordered, full-width.

### Video player (local file or YouTube)

A drop-in player that auto-detects the source type. Just give it a `data-src` and it figures out whether to render a `<video>` (local file) or a YouTube `<iframe>`.

```html
<!-- YouTube — any URL form is accepted -->
<div class="video-player"
     data-src="https://www.youtube.com/watch?v=aqz-KE-bpKQ"></div>

<!-- Local MP4 -->
<div class="video-player"
     data-src="static/videos/teaser.mp4"
     data-poster="static/images/teaser-thumb.jpg"></div>
```

| Attribute | Default | Notes |
|---|---|---|
| `data-src`      | — (required) | Local path **or** any YouTube URL: `youtube.com/watch?v=…`, `youtu.be/…`, `/embed/…`, `/shorts/…` |
| `data-aspect`   | `16/9` | `4/3`, `1/1`, `21/9`, `9/16`, … (sets `aspect-ratio` on the container) |
| `data-autoplay` | `false` | Forces `muted` per browser policy |
| `data-loop`     | `false` | Loops indefinitely |
| `data-muted`    | `false` (or `true` if autoplay) | |
| `data-controls` | `true` | Set `"false"` to hide controls (autoplay loop hero clips) |
| `data-poster`   | — | **Local video only** — thumbnail before play |
| `data-title`    | `"Video player"` | YouTube iframe `title` attribute |

The container has `aspect-ratio: 16/9` and rounded corners by default. To constrain its width inline:

```html
<div class="video-player"
     style="max-width:480px; margin:28px auto 0;"
     data-src="…"></div>
```

### Footer bar

```html
<footer class="footer-bar">
  Feel free to borrow the source code of this website — just link back to
  <a href="#">this page</a> in the footer.
</footer>
```

---

## Customising the look

Global tokens live in `:root` at the top of `paper.css`:

```css
:root {
  --fg:        #1a1a1a;
  --muted:     #6b7280;
  --bg:        #ffffff;
  --bg-soft:   #f7f7f8;
  --accent-1:  #ef4444;
  --accent-2:  #3b82f6;
  --accent-3:  #10b981;
  --accent-4:  #f59e0b;
  --rule:      #e5e7eb;
}
```

Override per-component via inline CSS variables — e.g. on `.side-card`:

```html
<aside class="side-card" data-annot-id="a"
       style="--card-width:320px; --card-font-size:15px;">
  …
</aside>
```

See the `.side-card { --card-* … }` block in `paper.css` for every knob.

---

## Page-specific JavaScript

Anything not covered by a built-in component is just a plain `<script>` after `paper.js`:

```html
<script src="static/js/paper.js"></script>
<script>
  document.getElementById('runBtn').addEventListener('click', () => { … });
</script>
```

If you inject markup *after* page load and need a component to re-init, call `window.Paper.initEmblas()` (etc.) — they're idempotent on already-initialised elements.

---

## Layout helpers

| Class | What it does |
|---|---|
| `.container` | Centred, max-width 980px, horizontal padding |
| `.narrow` | Add to `.container` to cap at 720px |
| `.muted` | Apply the muted gray text colour |
| `section` | 96px vertical padding by default (56px on mobile) |
| `section.alt` | Soft-gray section background — alternate to break up the page |
| `.reveal` | Hidden+offset until scrolled into view |

---

## Mobile

A single breakpoint at `≤720px` collapses everything sensibly:

- Splits stack to one column (image on top)
- Figure grid drops to 2 columns
- Scrollytelling unpins; figure becomes sticky on top
- Embla, demo form, table, and Explorer reflow with smaller paddings/fonts
- Side cards collapse inline below the title

---

## Dependencies

Loaded from CDN at the bottom of `template.html`:

- [GSAP 3.12](https://greensock.com/gsap/) + [ScrollTrigger](https://greensock.com/scrolltrigger/) — animation engine
- [Lenis 1.0](https://github.com/studio-freight/lenis) — smooth scroll
- [Embla Carousel](https://www.embla-carousel.com/) — headless carousel
- [Rough Notation](https://roughnotation.com/) — hand-drawn annotations
- [Tippy.js 6](https://atomiks.github.io/tippyjs/) (+ Popper) — tooltips

All MIT or BSD-licensed. No build step required.

---

## Browser support

Modern evergreen browsers (Chrome, Firefox, Safari, Edge). Components rely on:

- CSS Grid + custom properties
- `clip-path: inset()` (compare slider)
- `navigator.clipboard.writeText` (code copy — falls back to `execCommand`)
- Pointer events (compare slider — works on touch)
- `gsap.matchMedia()` for mobile/desktop branches

---

## Editing tips

- **Keep `template.html` minimal.** All visual styling lives in `paper.css`; behaviour lives in `paper.js`. The template should only contain content + class names + data attributes.
- **Reorder sections freely** — there's no implicit dependency between them.
- **Per-instance tweaks** belong in inline `style="--variable:..."` attributes, not in `paper.css`. The variables are the API.
- **Need a new component?** Add CSS to `paper.css`, add an `init*` function in `paper.js` (and call it from `boot()`), and use it via class names + data attributes in `template.html`.
