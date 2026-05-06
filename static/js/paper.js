/* ============================================================
   paper.js — paper-site component library

   Auto-discovers components from the DOM on DOMContentLoaded.
   Pairs with paper.css. Required globals (loaded via <script>):
     gsap, ScrollTrigger, Lenis, EmblaCarousel, RoughNotation, tippy
   ============================================================ */
(function () {
  'use strict';

  /* ---------- helpers ---------- */
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const $  = (sel, root) => (root || document).querySelector(sel);
  const num = (v, d) => (v == null || v === '' ? d : +v);

  /* ============================================================
     Smooth scroll (Lenis) + ScrollTrigger ticker.
     ============================================================ */
  let lenis;
  let scrollingNow = false;
  let scrollIdleTimer = null;
  function initSmoothScroll() {
    lenis = new Lenis({ duration: 1.1, smoothWheel: true });
    lenis.on('scroll', () => {
      ScrollTrigger.update();
      scrollingNow = true;
      clearTimeout(scrollIdleTimer);
      scrollIdleTimer = setTimeout(() => { scrollingNow = false; }, 150);
    });
    gsap.ticker.add((t) => lenis.raf(t * 1000));
    gsap.ticker.lagSmoothing(0);
  }

  /* ============================================================
     Reveal-on-scroll. Anything with .reveal fades+slides in.
     ============================================================ */
  function initReveal() {
    $$('.reveal').forEach((el) => {
      gsap.to(el, {
        opacity: 1, y: 0, duration: 0.8, ease: 'power2.out',
        scrollTrigger: {
          trigger: el,
          start: 'top 85%',
          end: 'bottom 15%',
          toggleActions: 'play none none reverse',
        },
      });
    });
  }

  /* ============================================================
     In-text annotations. Markup:
       <span class="annot" data-type="underline" data-color="#ef4444">…</span>
     Optional: data-stroke, data-padding, data-duration.
     Annotations inside scrolly paragraphs are skipped here — they
     are activated by the scrolly module instead (active-slide bound).
     ============================================================ */
  function initAnnotations() {
    const RN = window.RoughNotation;
    $$('.annot').forEach((el) => {
      // Skip hero phrases (handled by initHero) and scrolly-bound phrases
      // (handled by initScrolly via the active-slide highlight).
      if (el.classList.contains('hero-annot')) return;
      if (el.closest('.scrolly')) return;

      const type    = el.dataset.type    || 'underline';
      const color   = el.dataset.color   || '#3b82f6';
      const stroke  = num(el.dataset.stroke,   2);
      const padding = num(el.dataset.padding,  2);
      const dur     = num(el.dataset.duration, 600);

      const ann = RN.annotate(el, { type, color, strokeWidth: stroke, padding, animationDuration: dur });
      ScrollTrigger.create({
        trigger: el, start: 'top 85%',
        onEnter:     () => ann.show(),
        onLeaveBack: () => ann.hide(),
      });
    });
  }

  /* ============================================================
     Tippy tooltips. Anything with [data-tippy] gets one.
     ============================================================ */
  function initTooltips() {
    if (typeof tippy !== 'function') return;
    tippy('[data-tippy]', {
      content: (ref) => ref.getAttribute('data-tippy'),
      theme: 'light-border',
      animation: 'shift-away',
    });
  }

  /* ============================================================
     Hero — delay-driven annotations + side cards.
     Markup:
       <span class="hero-annot" data-annot-id="a"
             data-type="underline" data-color="#ef4444"
             data-delay="1.0">phrase</span>
       <aside class="side-card" data-annot-id="a" data-side="right">…</aside>
     The card slides in from `data-side` (`left`|`right`) when its
     phrase fires. Per-card knobs via inline style="--card-*: …".
     ============================================================ */
  function initHero() {
    const RN = window.RoughNotation;
    const phrases = $$('.hero-annot[data-annot-id]');
    if (!phrases.length) return;

    // Pre-hide each card and offset it in the direction it'll fly in.
    phrases.forEach((phrase) => {
      const id   = phrase.dataset.annotId;
      const side = phrase.dataset.side || 'right';
      const card = $(`.side-card[data-annot-id="${id}"]`);
      if (card) gsap.set(card, { opacity: 0, x: side === 'right' ? 30 : -30 });
    });

    phrases.forEach((phrase) => {
      const id     = phrase.dataset.annotId;
      const type   = phrase.dataset.type   || 'underline';
      const color  = phrase.dataset.color  || '#ef4444';
      const stroke = num(phrase.dataset.stroke,   2.5);
      const pad    = num(phrase.dataset.padding,  3);
      const dur    = num(phrase.dataset.duration, 700);
      const delay  = num(phrase.dataset.delay,    1.0);
      const card   = $(`.side-card[data-annot-id="${id}"]`);

      const ann = RN.annotate(phrase, { type, color, strokeWidth: stroke, padding: pad, animationDuration: dur });
      gsap.delayedCall(delay, () => {
        ann.show();
        if (card) gsap.to(card, { opacity: 1, x: 0, duration: 0.7, ease: 'power3.out' });
      });
    });
  }

  /* ============================================================
     Scrollytelling — pinned figure column + bound text.
     Markup:
       <div class="scrolly">
         <div class="stage">
           <div class="text-col">
             <p data-slide="0">… <span class="annot">phrase</span> …</p>
             …
           </div>
           <div class="figure-col">
             <div class="figure-frame">
               <div class="figure-slide is-active">…</div>
               …
             </div>
           </div>
         </div>
       </div>
     The progress dots are auto-generated (one per .figure-slide).
     Annotations inside paragraphs auto-bind to the active slide.
     ============================================================ */
  function initScrolly() {
    const RN = window.RoughNotation;
    $$('.scrolly').forEach((scrolly) => {
      const figureSlides   = $$('.figure-slide', scrolly);
      const textParagraphs = $$('.text-col p[data-slide]', scrolly);
      const NUM_SLIDES     = figureSlides.length;
      if (!NUM_SLIDES) return;

      // Auto-generate progress dots inside each .figure-frame if missing.
      const frame = $('.figure-frame', scrolly);
      let progress = $('.figure-progress', scrolly);
      if (frame && !progress) {
        progress = document.createElement('div');
        progress.className = 'figure-progress';
        for (let i = 0; i < NUM_SLIDES; i++) progress.appendChild(document.createElement('span'));
        frame.appendChild(progress);
      }
      const progressDots = progress ? Array.from(progress.children) : [];

      // Auto-annotate each phrase inside paragraphs (color/type configurable
      // via data-* on the .annot element). Add .is-active when its slide is.
      const phraseAnnots = []; // [{ el, ann, slideIdx }]
      textParagraphs.forEach((p) => {
        const slideIdx = +p.dataset.slide;
        $$('.annot', p).forEach((el) => {
          const type   = el.dataset.type   || 'underline';
          const color  = el.dataset.color  || '#3b82f6';
          const stroke = num(el.dataset.stroke, 2);
          const pad    = num(el.dataset.padding, 2);
          const dur    = num(el.dataset.duration, 600);
          const ann = RN.annotate(el, { type, color, strokeWidth: stroke, padding: pad, animationDuration: dur });
          phraseAnnots.push({ el, ann, slideIdx });
        });
      });
      // Draw all phrase annotations once the section enters view.
      if (phraseAnnots.length) {
        ScrollTrigger.create({
          trigger: scrolly, start: 'top 80%',
          onEnter:     () => phraseAnnots.forEach(({ ann }) => ann.show()),
          onLeaveBack: () => phraseAnnots.forEach(({ ann }) => ann.hide()),
        });
      }

      function setActiveSlide(idx) {
        figureSlides.forEach((s, i)  => s.classList.toggle('is-active', i === idx));
        progressDots.forEach((d, i)  => d.classList.toggle('is-active', i === idx));
        textParagraphs.forEach((p, i) => p.classList.toggle('is-active-para', i === idx));
        phraseAnnots.forEach(({ el, slideIdx }) => el.classList.toggle('is-active', slideIdx === idx));
      }

      gsap.matchMedia().add({
        isDesktop: '(min-width: 721px)',
        isMobile:  '(max-width: 720px)',
      }, (ctx) => {
        const { isDesktop, isMobile } = ctx.conditions;

        if (isDesktop) {
          const trig = ScrollTrigger.create({
            trigger: scrolly,
            start: 'top top',
            end: () => '+=' + (NUM_SLIDES * window.innerHeight),
            pin: scrolly,
            pinSpacing: true,
            scrub: true,
            onUpdate: (self) => {
              const idx = Math.min(NUM_SLIDES - 1, Math.floor(self.progress * NUM_SLIDES * 0.9999));
              setActiveSlide(idx);
            },
          });
          // Hover-to-jump. Suppressed while a scroll is in flight, so wheeling
          // through the pinned section never fights with mouseenter handlers
          // that would otherwise yank the page back to a paragraph anchor.
          // Also requires real cursor movement since the last scroll (synthetic
          // mouseenter events fire when the pin engages and DOM gets reparented).
          let lastMouseMoveAt = 0;
          const onMove = () => { lastMouseMoveAt = performance.now(); };
          window.addEventListener('mousemove', onMove);

          const handlers = [];
          textParagraphs.forEach((p, i) => {
            const fn = () => {
              if (scrollingNow) return;
              if (performance.now() - lastMouseMoveAt > 120) return;
              if (p.classList.contains('is-active-para')) return;
              const targetY = trig.start + ((i + 0.5) / NUM_SLIDES) * (trig.end - trig.start);
              if (lenis) lenis.scrollTo(targetY, { duration: 0.6 });
              else window.scrollTo({ top: targetY, behavior: 'smooth' });
            };
            p.addEventListener('mouseenter', fn);
            handlers.push([p, fn]);
          });
          return () => {
            window.removeEventListener('mousemove', onMove);
            handlers.forEach(([p, fn]) => p.removeEventListener('mouseenter', fn));
          };
        }

        if (isMobile) {
          textParagraphs.forEach((p, i) => {
            ScrollTrigger.create({
              trigger: p,
              start: 'top 50%',
              end:   'bottom 50%',
              onEnter:     () => setActiveSlide(i),
              onEnterBack: () => setActiveSlide(i),
            });
          });
        }
      });
    });
  }

  /* ============================================================
     Embla carousels. Markup:
       <div class="embla" data-embla data-loop="true">
         <div class="embla__container">
           <div class="embla__slide">…</div>
           …
         </div>
       </div>
     Optional siblings, auto-detected if next to the carousel:
       <div class="embla-controls">…</div>  (prev/next auto-injected)
       <div class="embla-dots"></div>       (dots auto-injected)
     If they aren't present, paper.js appends them after the .embla.
     ============================================================ */
  function initEmblas() {
    $$('[data-embla]').forEach((node) => {
      // Skip thumb-strips — Explorer manages those.
      if (node.closest('.explorer')) return;

      const loop  = node.dataset.loop  === 'true';
      const align = node.dataset.align || 'start';
      const dragFree = node.dataset.dragFree === 'true';
      const embla = EmblaCarousel(node, { loop, align, dragFree });

      // Locate or create controls/dots near the carousel.
      const parent = node.parentElement;
      let controls = parent && $('.embla-controls', parent);
      let dots     = parent && $('.embla-dots',     parent);
      if (!controls) {
        controls = document.createElement('div');
        controls.className = 'embla-controls';
        controls.innerHTML = '<button data-embla-prev>← Prev</button><button data-embla-next>Next →</button>';
        node.insertAdjacentElement('afterend', controls);
      }
      if (!dots) {
        dots = document.createElement('div');
        dots.className = 'embla-dots';
        controls.insertAdjacentElement('afterend', dots);
      }
      const prev = $('[data-embla-prev]', controls) || controls.querySelector('button:first-child');
      const next = $('[data-embla-next]', controls) || controls.querySelector('button:last-child');
      if (prev) prev.addEventListener('click', () => embla.scrollPrev());
      if (next) next.addEventListener('click', () => embla.scrollNext());

      embla.scrollSnapList().forEach((_, i) => {
        const b = document.createElement('button');
        b.addEventListener('click', () => embla.scrollTo(i));
        dots.appendChild(b);
      });
      const updateDots = () => {
        const i = embla.selectedScrollSnap();
        dots.querySelectorAll('button').forEach((b, j) => b.classList.toggle('is-active', j === i));
      };
      embla.on('select', updateDots);
      updateDots();
    });
  }

  /* ============================================================
     Inline SVG diagrams. Any <svg class="svg-diagram"> with
     <path class="flow"> children gets a path-draw on scroll-in.
     ============================================================ */
  function initSvgFlows() {
    $$('svg.svg-diagram').forEach((svg) => {
      const flows = $$('.flow', svg);
      if (!flows.length) return;
      flows.forEach((p) => {
        const len = p.getTotalLength();
        p.style.strokeDasharray  = len;
        p.style.strokeDashoffset = len;
      });
      ScrollTrigger.create({
        trigger: svg,
        start: 'top 80%',
        onEnter: () => {
          gsap.to(flows, { strokeDashoffset: 0, duration: 1.2, ease: 'power2.out', stagger: 0.15 });
        },
        onLeaveBack: () => {
          flows.forEach((p) => gsap.set(p, { strokeDashoffset: p.getTotalLength() }));
        },
      });
    });
  }

  /* ============================================================
     External Inkscape SVGs. Markup:
       <div class="svg-holder" data-svg-src="static/images/wm.svg"></div>
     Optional: data-svg-stagger, data-svg-fade, data-svg-draw, data-svg-start.
     Fetch the file, inject it inline, fade+stagger top-level
     children in, and draw stroked paths via stroke-dashoffset.
     Falls back to <img> if fetch fails (e.g. file:// CORS).
     ============================================================ */
  function initExternalSvgs() {
    $$('.svg-holder[data-svg-src]').forEach((holder) => {
      const url     = holder.dataset.svgSrc;
      const stagger = num(holder.dataset.svgStagger, 0.025);
      const fade    = num(holder.dataset.svgFade,    0.6);
      const draw    = num(holder.dataset.svgDraw,    1.2);
      const start   = holder.dataset.svgStart || 'top 75%';

      fetch(url).then((r) => r.text()).then((text) => {
        holder.innerHTML = text;
        const svg = holder.querySelector('svg');
        if (!svg) return;
        animate(svg);
      }).catch((err) => {
        console.warn('[paper.js] external SVG fetch failed (' + err.message +
          '). Falling back to <img>. Run a local server (e.g. `python3 -m http.server`) for the animated version.');
        holder.innerHTML =
          '<img src="' + url + '" style="width:100%;height:auto;display:block;" alt="">' +
          '<p style="text-align:center;color:#9ca3af;font-size:12px;margin-top:8px;">' +
          '(static fallback — animation requires HTTP, not <code>file://</code>.)</p>';
      });

      function animate(svg) {
        svg.removeAttribute('width');
        svg.removeAttribute('height');

        const stage = svg.querySelector('#layer1') || svg;
        const SKIP  = ['defs', 'metadata', 'sodipodi:namedview', 'title', 'desc'];
        const children = Array.from(stage.children).filter((c) => !SKIP.includes(c.tagName.toLowerCase()));
        if (!children.length) return;

        // Prep stroked descendants for draw-in.
        const drawables = [];
        children.forEach((child) => {
          const paths = child.tagName.toLowerCase() === 'path' ? [child] : child.querySelectorAll('path');
          paths.forEach((p) => {
            const cs = window.getComputedStyle(p);
            const sw = parseFloat(cs.strokeWidth);
            if (cs.stroke && cs.stroke !== 'none' && sw > 0.1) {
              let len = 0;
              try { len = p.getTotalLength(); } catch (e) { return; }
              if (len > 1 && isFinite(len)) {
                p.style.strokeDasharray  = len;
                p.style.strokeDashoffset = len;
                drawables.push({ p, len });
              }
            }
          });
        });

        // Opacity-only initial state (CSS transforms on SVG children with
        // their own transform="..." would blow up the layout without
        // transform-box: fill-box).
        gsap.set(children, { opacity: 0 });

        ScrollTrigger.create({
          trigger: holder,
          start,
          onEnter: () => {
            gsap.to(children, { opacity: 1, duration: fade, stagger, ease: 'power2.out' });
            gsap.to(drawables.map((d) => d.p), { strokeDashoffset: 0, duration: draw, stagger, ease: 'power2.out' });
          },
          onLeaveBack: () => {
            gsap.set(children, { opacity: 0 });
            drawables.forEach(({ p, len }) => gsap.set(p, { strokeDashoffset: len }));
          },
        });

        ScrollTrigger.refresh();
      }
    });
  }

  /* ============================================================
     Explorer: thumb strip + grid sub-component.
     Markup:
       <div class="explorer">
         <script type="application/json">[ ...configs... ]</script>
       </div>
     The entire DOM (thumb strip, detail panel, controls) is built
     from the embedded JSON config. See template.html for the schema.
     ============================================================ */
  function initExplorers() {
    $$('.explorer').forEach((root) => {
      const cfgScript = $('script[type="application/json"]', root);
      if (!cfgScript) return;
      let configs = [];
      try { configs = JSON.parse(cfgScript.textContent); } catch (e) { console.error('[paper.js] explorer JSON parse failed', e); return; }
      if (!Array.isArray(configs) || !configs.length) return;

      // Build skeleton.
      root.innerHTML = `
        <div class="thumb-strip embla">
          <div class="embla__container"></div>
          <button class="thumb-nav prev" aria-label="prev">‹</button>
          <button class="thumb-nav next" aria-label="next">›</button>
        </div>
        <div class="detail">
          <div class="detail-title">—</div>
          <div class="col-labels-host"></div>
          <div class="detail-grid"></div>
        </div>
      `;

      const stripContainer = $('.thumb-strip .embla__container', root);
      const detailTitle    = $('.detail-title', root);
      const detailGrid     = $('.detail-grid',  root);
      const colLabelsEl    = $('.col-labels-host', root);

      configs.forEach((cfg, i) => {
        const slide = document.createElement('div');
        slide.className = 'embla__slide';
        slide.innerHTML = `<div class="thumb" data-thumb-idx="${i}" style="background:${cfg.thumb.color};">${cfg.thumb.label}</div>`;
        stripContainer.appendChild(slide);
      });

      const stripEl = $('.thumb-strip', root);
      const embla   = EmblaCarousel(stripEl, { dragFree: true, containScroll: 'trimSnaps' });
      $('.thumb-nav.prev', root).addEventListener('click', () => embla.scrollPrev());
      $('.thumb-nav.next', root).addEventListener('click', () => embla.scrollNext());

      let activeIntervals = [];
      function clearDetail() {
        activeIntervals.forEach(clearInterval);
        activeIntervals = [];
        detailGrid.innerHTML  = '';
        colLabelsEl.innerHTML = '';
        colLabelsEl.className = 'col-labels-host';
      }

      function buildCellMedia(cell) {
        const media = document.createElement('div');
        media.className = 'media';
        if (cell.kind === 'color') {
          media.style.background = cell.color;
          media.textContent = cell.label || '';
        } else if (cell.kind === 'image') {
          const img = document.createElement('img');
          img.src = cell.src; img.alt = cell.label || '';
          media.appendChild(img);
        } else if (cell.kind === 'video') {
          const v = document.createElement('video');
          v.src = cell.src; v.autoplay = true; v.loop = true; v.muted = true; v.playsInline = true;
          media.appendChild(v);
        } else if (cell.kind === 'sequence') {
          let f = 0;
          const apply = () => {
            const fr = cell.frames[f];
            if (fr.src) {
              media.style.background = ''; media.textContent = '';
              if (!media.querySelector('img')) media.innerHTML = '<img>';
              media.querySelector('img').src = fr.src;
            } else {
              media.style.background = fr.color;
              media.textContent = (f + 1) + '/' + cell.frames.length;
            }
            f = (f + 1) % cell.frames.length;
          };
          apply();
          activeIntervals.push(setInterval(apply, cell.interval || 400));
        }
        return media;
      }

      function renderDetail(cfg) {
        clearDetail();
        detailTitle.textContent = cfg.title || '';

        if (cfg.colLabels && cfg.colLabels.length) {
          colLabelsEl.className = 'col-labels';
          cfg.colLabels.forEach((label) => {
            const s = document.createElement('span'); s.textContent = label; colLabelsEl.appendChild(s);
          });
        }

        const cellNodes   = [];
        const rowsAsCells = [];
        cfg.rows.forEach((row, ri) => {
          const rowEl = document.createElement('div'); rowEl.className = 'detail-row';
          if (cfg.rowLabels) {
            const lab = document.createElement('div'); lab.className = 'row-label';
            lab.textContent = cfg.rowLabels[ri] || '';
            rowEl.appendChild(lab);
          }
          const cellsWrap = document.createElement('div'); cellsWrap.className = 'cells';
          const rowCells = [];
          row.cells.forEach((cell) => {
            const cellEl = document.createElement('div'); cellEl.className = 'detail-cell';
            cellEl.appendChild(buildCellMedia(cell));
            if (cell.caption) {
              const cap = document.createElement('div'); cap.className = 'caption';
              cap.textContent = cell.caption;
              cellEl.appendChild(cap);
            }
            cellsWrap.appendChild(cellEl);
            cellNodes.push(cellEl);
            rowCells.push(cellEl);
          });
          rowEl.appendChild(cellsWrap);
          detailGrid.appendChild(rowEl);
          rowsAsCells.push(rowCells);
        });

        const stagger = (cfg.animation && cfg.animation.stagger != null ? cfg.animation.stagger : 0.08) * 1000;
        const mode    = (cfg.animation && cfg.animation.mode) || 'cell';
        const showAfter = (el, ms) => setTimeout(() => el.classList.add('is-shown'), ms);

        if (Array.isArray(mode)) {
          mode.forEach((idx, k) => { if (cellNodes[idx]) showAfter(cellNodes[idx], k * stagger); });
          cellNodes.forEach((el, i) => { if (!mode.includes(i)) showAfter(el, mode.length * stagger); });
        } else if (mode === 'row') {
          rowsAsCells.forEach((rc, ri) => rc.forEach((el) => showAfter(el, ri * stagger)));
        } else if (mode === 'col') {
          const maxLen = Math.max(...rowsAsCells.map((r) => r.length));
          for (let c = 0; c < maxLen; c++) {
            rowsAsCells.forEach((r) => { if (r[c]) showAfter(r[c], c * stagger); });
          }
        } else {
          cellNodes.forEach((el, i) => showAfter(el, i * stagger));
        }
      }

      function activate(idx) {
        $$('.thumb', root).forEach((t) => t.classList.toggle('is-active', +t.dataset.thumbIdx === idx));
        renderDetail(configs[idx]);
      }
      stripContainer.addEventListener('mouseover', (e) => {
        const t = e.target.closest('.thumb');
        if (t) activate(+t.dataset.thumbIdx);
      });
      stripContainer.addEventListener('click', (e) => {
        const t = e.target.closest('.thumb');
        if (t) activate(+t.dataset.thumbIdx);
      });
      activate(0);
    });
  }

  /* ============================================================
     Code blocks with a copy button. Markup:
       <div class="code-block" data-lang="BibTeX">
         <pre>@inproceedings{...}</pre>
       </div>
     `data-lang` is shown as a small label in the corner. The copy
     button is auto-injected; clicking it copies the <pre> text
     (or the block's textContent if there's no <pre>).
     ============================================================ */
  function initCodeBlocks() {
    $$('.code-block').forEach((block) => {
      if (block.dataset.lang && !block.querySelector('.code-lang')) {
        const lang = document.createElement('span');
        lang.className = 'code-lang';
        lang.textContent = block.dataset.lang;
        block.appendChild(lang);
      }
      if (block.querySelector('.code-copy')) return;

      const btn = document.createElement('button');
      btn.className = 'code-copy';
      btn.type = 'button';
      btn.textContent = 'Copy';
      btn.addEventListener('click', async () => {
        const pre = block.querySelector('pre');
        const text = (pre ? pre.textContent : block.textContent).replace(/^\n+|\n+$/g, '');
        try {
          await navigator.clipboard.writeText(text);
          btn.textContent = '✓ Copied';
          btn.classList.add('is-ok');
        } catch (e) {
          // Fallback: select-and-copy via a temporary textarea.
          const ta = document.createElement('textarea');
          ta.value = text; document.body.appendChild(ta);
          ta.select();
          try { document.execCommand('copy'); btn.textContent = '✓ Copied'; btn.classList.add('is-ok'); }
          catch (_) { btn.textContent = 'Copy failed'; }
          document.body.removeChild(ta);
        }
        setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('is-ok'); }, 1500);
      });
      block.appendChild(btn);
    });
  }

  /* ============================================================
     Comparison sliders (before / after). Markup:
       <div class="compare"
            data-before="path/to/before.jpg"
            data-after="path/to/after.jpg"
            data-before-label="Input"
            data-after-label="Ours"></div>
     Drag (or touch-drag) the handle to wipe between the two.
     The "before" image is clip-path'd so you reveal the "after"
     underneath as you drag right.
     ============================================================ */
  function initCompareSliders() {
    $$('.compare').forEach((root) => {
      const before = root.dataset.before;
      const after  = root.dataset.after;
      if (!before || !after) return;
      const beforeLabel = root.dataset.beforeLabel;
      const afterLabel  = root.dataset.afterLabel;

      if (!root.querySelector('.compare-handle')) {
        root.innerHTML =
          `<img class="compare-after"  src="${after}"  alt="">` +
          `<img class="compare-before" src="${before}" alt="">` +
          `<div class="compare-handle"></div>` +
          (beforeLabel ? `<span class="compare-label before">${beforeLabel}</span>` : '') +
          (afterLabel  ? `<span class="compare-label after">${afterLabel}</span>`   : '');
      }
      const beforeImg = root.querySelector('.compare-before');
      const handle    = root.querySelector('.compare-handle');

      function setPct(pct) {
        const v = Math.max(0, Math.min(100, pct));
        beforeImg.style.clipPath = `inset(0 ${100 - v}% 0 0)`;
        handle.style.left = v + '%';
      }
      setPct(50);

      function ptToPct(clientX) {
        const r = root.getBoundingClientRect();
        return ((clientX - r.left) / r.width) * 100;
      }

      let dragging = false;
      root.addEventListener('pointerdown', (e) => {
        dragging = true;
        try { root.setPointerCapture(e.pointerId); } catch (_) {}
        setPct(ptToPct(e.clientX));
      });
      root.addEventListener('pointermove', (e) => {
        if (!dragging) return;
        e.preventDefault();
        setPct(ptToPct(e.clientX));
      });
      const end = (e) => {
        dragging = false;
        try { root.releasePointerCapture(e.pointerId); } catch (_) {}
      };
      root.addEventListener('pointerup', end);
      root.addEventListener('pointercancel', end);
    });
  }

  /* ============================================================
     Video player. Markup:
       <div class="video-player" data-src="…"
            data-aspect="16/9" data-autoplay="true"
            data-loop="true" data-muted="true"></div>
     Auto-detects local file vs. YouTube URL and renders <video>
     or <iframe> accordingly. data-aspect overrides the default
     16/9 via inline `aspect-ratio` on the container.
     ============================================================ */
  function extractYouTubeId(url) {
    const re = /(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
    const m = url.match(re);
    return m ? m[1] : null;
  }
  function initVideoPlayers() {
    $$('.video-player[data-src]').forEach((root) => {
      if (root.querySelector('video, iframe')) return; // already initialized

      const src      = root.dataset.src;
      const aspect   = root.dataset.aspect;
      const autoplay = root.dataset.autoplay === 'true';
      const loop     = root.dataset.loop     === 'true';
      // Browser autoplay policies require muted unless user interacted.
      const muted    = root.dataset.muted    === 'true' || autoplay;
      const controls = root.dataset.controls !== 'false';
      const poster   = root.dataset.poster;

      if (aspect) root.style.aspectRatio = aspect.replace(':', ' / ');

      const ytId = extractYouTubeId(src);
      if (ytId) {
        const params = new URLSearchParams();
        if (autoplay) params.set('autoplay', '1');
        if (loop)     { params.set('loop', '1'); params.set('playlist', ytId); }
        if (muted)    params.set('mute', '1');
        if (!controls) params.set('controls', '0');
        params.set('rel', '0');
        params.set('modestbranding', '1');
        params.set('playsinline', '1');

        const iframe = document.createElement('iframe');
        iframe.src = `https://www.youtube.com/embed/${ytId}?${params.toString()}`;
        iframe.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
        iframe.setAttribute('allowfullscreen', '');
        iframe.setAttribute('loading', 'lazy');
        iframe.setAttribute('title', root.dataset.title || 'Video player');
        root.appendChild(iframe);
      } else {
        const video = document.createElement('video');
        video.src = src;
        if (controls) video.controls = true;
        if (autoplay) video.autoplay = true;
        if (loop)     video.loop     = true;
        if (muted)    video.muted    = true;
        video.playsInline = true;
        if (poster)   video.poster   = poster;
        root.appendChild(video);
      }
    });
  }

  /* ============================================================
     Boot.
     ============================================================ */
  function boot() {
    if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') {
      console.error('[paper.js] gsap + ScrollTrigger are required. Did you include the <script> tags?');
      return;
    }
    gsap.registerPlugin(ScrollTrigger);

    if (typeof Lenis === 'function') initSmoothScroll();
    initReveal();
    initAnnotations();
    initTooltips();
    initHero();
    initScrolly();
    initEmblas();
    initSvgFlows();
    initExternalSvgs();
    initExplorers();
    initCodeBlocks();
    initCompareSliders();
    initVideoPlayers();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  // Expose a tiny namespace in case callers want to re-run a module.
  window.Paper = {
    initReveal, initAnnotations, initTooltips, initHero, initScrolly,
    initEmblas, initSvgFlows, initExternalSvgs, initExplorers,
    initCodeBlocks, initCompareSliders, initVideoPlayers,
  };
})();
