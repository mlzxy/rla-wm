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
      // Skip elements inside a scrolly — the scrolly's own pinned
      // ScrollTrigger manages visibility of .figure-slide children.
      // Set them immediately to visible so the CSS .reveal {opacity:0}
      // default doesn't hide them.
      if (el.closest('.scrolly')) {
        el.style.opacity = '1';
        el.style.transform = 'translateY(0)';
        return;
      }
      gsap.to(el, {
        opacity: 1, y: 0, duration: 0.8, ease: 'power2.out',
        scrollTrigger: {
          trigger: el,
          start: 'top 85%',
          end: 'bottom 15%',
          toggleActions: 'play none play reverse',
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
    $$('.annot').forEach((el) => {
      // Skip hero phrases (handled by initHero) and scrolly-bound phrases
      // (handled by initScrolly via the active-slide highlight).
      if (el.classList.contains('hero-annot')) return;
      if (el.closest('.scrolly')) return;

      if (el.dataset.color) el.style.setProperty('--annot-bg', el.dataset.color);
      ScrollTrigger.create({
        trigger: el, start: 'top 85%',
        onEnter:     () => el.classList.add('is-active'),
        onLeaveBack: () => el.classList.remove('is-active'),
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

    const isMobile = () => window.matchMedia('(max-width: 720px)').matches;

    // Pre-hide each card and offset it in the direction it'll fly in.
    // On mobile, skip the offset so the static-flow card layout isn't shifted.
    phrases.forEach((phrase) => {
      const id   = phrase.dataset.annotId;
      const side = phrase.dataset.side || 'right';
      const card = $(`.side-card[data-annot-id="${id}"]`);
      if (!card) return;
      // Bind card to its phrase text so mobile CSS can render a "↳ <phrase>" reference.
      card.dataset.annotRef = phrase.textContent.trim().replace(/\s+/g, ' ');
      if (isMobile()) {
        gsap.set(card, { opacity: 1, x: 0, clearProps: 'opacity,transform' });
      } else {
        gsap.set(card, { opacity: 0, x: side === 'right' ? 30 : -30 });
      }
    });

    const annotations = [];
    let isMobileActive = isMobile();

    function applyMobileCssUnderline(phrase, color, stroke, pad) {
      phrase.style.textDecoration = 'underline';
      phrase.style.textDecorationColor = color;
      phrase.style.textDecorationThickness = stroke + 'px';
      phrase.style.textUnderlineOffset = pad + 'px';
    }

    function clearMobileCssUnderline(phrase) {
      phrase.style.textDecoration = '';
      phrase.style.textDecorationColor = '';
      phrase.style.textDecorationThickness = '';
      phrase.style.textUnderlineOffset = '';
    }

    function removeAllRoughNotations() {
      annotations.forEach((a) => {
        if (a.ann) { try { a.ann.remove(); } catch (e) {} a.ann = null; }
      });
    }

    function createAnnotation(phrase, obj) {
      const { type, color, stroke, pad, dur } = obj;
      if (isMobileActive) {
        // CSS text-decoration handles multiline text wrapping natively
        applyMobileCssUnderline(phrase, color, stroke, pad);
        obj.ann = null;
        obj.isCss = true;
      } else {
        clearMobileCssUnderline(phrase);
        obj.ann = RN.annotate(phrase, { type, color, strokeWidth: stroke, padding: pad, animationDuration: dur });
        obj.isCss = false;
        gsap.delayedCall(obj.delay, () => {
          if (obj.ann) obj.ann.show();
          const card = $(`.side-card[data-annot-id="${obj.id}"]`);
          if (card && !isMobile()) gsap.to(card, { opacity: 1, x: 0, duration: 0.7, ease: 'power3.out' });
        });
      }
    }

    phrases.forEach((phrase) => {
      const id     = phrase.dataset.annotId;
      const type   = phrase.dataset.type   || 'underline';
      const color  = phrase.dataset.color  || '#ef4444';
      const stroke = num(phrase.dataset.stroke,   2.5);
      const pad    = num(phrase.dataset.padding,  3);
      const dur    = num(phrase.dataset.duration, 700);
      const delay  = num(phrase.dataset.delay,    1.0);

      const obj = { ann: null, phrase, id, type, color, stroke, pad, dur, delay, isCss: false };
      annotations.push(obj);
      createAnnotation(phrase, obj);
    });

    // When crossing the mobile/desktop breakpoint, switch between
    // CSS text-decoration (mobile) and rough-notation (desktop).
    window.addEventListener('resize', () => {
      const nowMobile = isMobile();
      if (nowMobile === isMobileActive) return;
      isMobileActive = nowMobile;
      removeAllRoughNotations();
      // Re-show side cards
      phrases.forEach((phrase) => {
        const id   = phrase.dataset.annotId;
        const side = phrase.dataset.side || 'right';
        const card = $(`.side-card[data-annot-id="${id}"]`);
        if (!card) return;
        if (nowMobile) {
          gsap.set(card, { opacity: 1, x: 0, clearProps: 'opacity,transform' });
        } else {
          gsap.set(card, { opacity: 0, x: side === 'right' ? 30 : -30 });
        }
      });
      annotations.forEach((obj) => createAnnotation(obj.phrase, obj));
    }, { passive: true });
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
      // Text items are anything with [data-slide] inside .text-col — supports
      // both the old flat <p data-slide="0"> and a real nested <ol>/<ul>:
      //   <ol><li data-slide="1">…<ul><li data-slide="1.1">…</li></ul></li></ol>
      // Items with [data-skip] (or [data-disabled]) are still rendered but
      // are NOT activatable: excluded from the scroll/hover/click rotation
      // and the figure column simply keeps the previous step's figure.
      const textParagraphs = $$('.text-col [data-slide]', scrolly)
        .filter((p) => !p.hasAttribute('data-skip') && !p.hasAttribute('data-disabled'));
      // Mark all skipped items so CSS can dim them visually.
      $$('.text-col [data-slide][data-skip], .text-col [data-slide][data-disabled]', scrolly)
        .forEach((p) => p.classList.add('is-skip'));
      const NUM_SLIDES     = textParagraphs.length;          // scroll length tracks text-step count
      if (!NUM_SLIDES || !figureSlides.length) return;

      // Map each figure slide's id → its DOM index. If a figure-slide has no
      // [data-slide], fall back to its sequential numeric position so legacy
      // (index-based) markup keeps working.
      const figIdToIdx = new Map();
      figureSlides.forEach((el, i) => {
        const id = (el.dataset.slide != null && el.dataset.slide !== '')
          ? String(el.dataset.slide) : String(i);
        if (!figIdToIdx.has(id)) figIdToIdx.set(id, i);
      });

      // Resolve a text item's slide id → figure index.
      // If exact match misses, walk up the dotted hierarchy ("1.2.3" → "1.2" → "1"),
      // so child nodes inherit their parent's figure when none of their own is given.
      function resolveFigIdx(slideId) {
        let id = String(slideId);
        while (id.length) {
          if (figIdToIdx.has(id)) return figIdToIdx.get(id);
          const dot = id.lastIndexOf('.');
          if (dot < 0) return -1;
          id = id.slice(0, dot);
        }
        return -1;
      }

      // Pre-resolve each text step → figure index. Unresolved ones inherit
      // the previous resolved figure (sticky), so a stray text node never
      // blanks the figure column.
      let lastSeen = -1;
      const stepFigIdx = textParagraphs.map((p) => {
        const r = resolveFigIdx(p.dataset.slide);
        if (r >= 0) lastSeen = r;
        return lastSeen >= 0 ? lastSeen : 0;
      });

      // Auto-generate progress dots inside each .figure-frame if missing.
      // One dot per *text step* (so progress reflects narration length).
      const frame = $('.figure-frame', scrolly);
      let progress = $('.figure-progress', scrolly);
      if (frame && !progress) {
        progress = document.createElement('div');
        progress.className = 'figure-progress';
        for (let i = 0; i < NUM_SLIDES; i++) progress.appendChild(document.createElement('span'));
        frame.appendChild(progress);
      }
      const progressDots = progress ? Array.from(progress.children) : [];

      // Collect each phrase inside paragraphs. They simply toggle .is-active
      // (CSS background highlight) when their step is the active one.
      const phrasePlain = []; // [{ el, stepIdx }]
      textParagraphs.forEach((p, stepIdx) => {
        $$('.annot', p).forEach((el) => {
          if (el.dataset.color) el.style.setProperty('--annot-bg', el.dataset.color);
          phrasePlain.push({ el, stepIdx });
        });
      });
      const phraseAnnots = []; // legacy, kept empty so existing refs are no-ops
      // (no rough-notation to draw)
      if (phraseAnnots.length) {
        ScrollTrigger.create({
          trigger: scrolly, start: 'top 80%',
          onEnter:     () => phraseAnnots.forEach(({ ann }) => ann.show()),
          onLeaveBack: () => phraseAnnots.forEach(({ ann }) => ann.hide()),
        });
      }
      // Pre-resolve each text item's chain of <li>s (itself + ancestors)
      // inside the text-col, so we can mark them "open" when this step (or one
      // of its descendants) is active. Used by the collapse/expand behaviour.
      const stepAncestors = textParagraphs.map((p) => {
        const out = [];
        let n = p;
        const root = $('.text-col', scrolly);
        while (n && n !== root) {
          if (n.tagName === 'LI') out.push(n);
          n = n.parentElement;
        }
        return out;
      });
      const allListItems = Array.from(new Set(stepAncestors.flat()));

      function setActiveSlide(stepIdx) {
        const figIdx = stepFigIdx[stepIdx];
        figureSlides.forEach((s, i)  => s.classList.toggle('is-active', i === figIdx));
        progressDots.forEach((d, i)  => d.classList.toggle('is-active', i === stepIdx));
        textParagraphs.forEach((p, i) => p.classList.toggle('is-active-para', i === stepIdx));
        phraseAnnots.forEach(({ el, stepIdx: si }) => el.classList.toggle('is-active', si === stepIdx));
        phrasePlain.forEach(({ el, stepIdx: si })  => el.classList.toggle('is-active', si === stepIdx));
        // Open only the ancestor <li>s of the active step; collapse all others.
        const openSet = new Set(stepAncestors[stepIdx] || []);
        allListItems.forEach((li) => li.classList.toggle('is-open', openSet.has(li)));
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
          let hoveredEl = null;
          const onMove = (e) => {
            lastMouseMoveAt = performance.now();
            hoveredEl = e.target;
          };
          window.addEventListener('mousemove', onMove);

          const handlers = [];
          const skipHandlers = [];
          textParagraphs.forEach((p, i) => {
            let retryTimer = 0;
            const targetY = () =>
              trig.start + ((i + 0.5) / NUM_SLIDES) * (trig.end - trig.start);
            const tryJump = () => {
              if (p.classList.contains('is-active-para')) return true;
              if (scrollingNow) return false;
              if (performance.now() - lastMouseMoveAt > 120) return false;
              // Don't activate the parent if the cursor is actually over a
              // deeper [data-slide] descendant.
              if (hoveredEl && hoveredEl.closest &&
                  hoveredEl.closest('[data-slide]') !== p) return false;
              if (lenis) lenis.scrollTo(targetY(), { duration: 0.25 });
              else window.scrollTo({ top: targetY(), behavior: 'smooth' });
              return true;
            };
            const enter = () => {
              if (tryJump()) return;
              // Hover was blocked (e.g. wheel scroll still settling). Keep
              // retrying while the user keeps hovering — the active step will
              // catch up shortly after they pause.
              clearInterval(retryTimer);
              retryTimer = setInterval(() => {
                if (tryJump()) clearInterval(retryTimer);
              }, 120);
            };
            const leave = () => { clearInterval(retryTimer); retryTimer = 0; };
            // Click: activate immediately + jump scroll position to match.
            // Stop propagation so a nested <li>'s click doesn't also fire on
            // its ancestor [data-slide] (which would re-activate the parent).
            const click = (ev) => {
              const closest = ev.target.closest('[data-slide]');
              if (closest !== p) return; // a deeper item handled it
              ev.preventDefault();
              ev.stopPropagation();
              clearInterval(retryTimer);
              setActiveSlide(i);
              const y = targetY();
              if (lenis) lenis.scrollTo(y, { duration: 0.15, immediate: false });
              else window.scrollTo({ top: y, behavior: 'auto' });
            };
            p.addEventListener('mouseenter', enter);
            p.addEventListener('mouseleave', leave);
            p.addEventListener('click', click);
            handlers.push([p, enter, leave, click]);
          });

          // Skip-item hover redirect: when a scrolly-list item with
          // [data-skip] or [data-disabled] is hovered, redirect to its
          // first non-skipped child slide so the parent acts as a
          // hoverable section header on desktop.
          const skipItems = $$('.text-col [data-slide][data-skip], .text-col [data-slide][data-disabled]', scrolly);
          skipItems.forEach((el) => {
            const firstChild = el.querySelector('[data-slide]:not([data-skip]):not([data-disabled])');
            if (!firstChild) return;
            const childId = firstChild.dataset.slide;
            if (!childId) return;
            const childIdx = textParagraphs.findIndex((p) => p.dataset.slide === childId);
            if (childIdx < 0) return;
            const childTargetY = () =>
              trig.start + ((childIdx + 0.5) / NUM_SLIDES) * (trig.end - trig.start);
            let skipRetryTimer = 0;
            const trySkipJump = () => {
              if (firstChild.classList.contains('is-active-para')) return true;
              if (scrollingNow) return false;
              if (performance.now() - lastMouseMoveAt > 120) return false;
              if (hoveredEl && hoveredEl.closest &&
                  hoveredEl.closest('[data-slide]') !== el) return false;
              if (lenis) lenis.scrollTo(childTargetY(), { duration: 0.25 });
              else window.scrollTo({ top: childTargetY(), behavior: 'smooth' });
              return true;
            };
            const onSkipEnter = () => {
              if (trySkipJump()) return;
              clearInterval(skipRetryTimer);
              skipRetryTimer = setInterval(() => {
                if (trySkipJump()) clearInterval(skipRetryTimer);
              }, 120);
            };
            const onSkipLeave = () => { clearInterval(skipRetryTimer); skipRetryTimer = 0; };
            el.addEventListener('mouseenter', onSkipEnter);
            el.addEventListener('mouseleave', onSkipLeave);
            skipHandlers.push([el, onSkipEnter, onSkipLeave]);
          });

          return () => {
            window.removeEventListener('mousemove', onMove);
            handlers.forEach(([p, enter, leave, click]) => {
              p.removeEventListener('mouseenter', enter);
              p.removeEventListener('mouseleave', leave);
              p.removeEventListener('click', click);
            });
            skipHandlers.forEach(([el, enter, leave]) => {
              el.removeEventListener('mouseenter', enter);
              el.removeEventListener('mouseleave', leave);
            });
          };
        }

        if (isMobile) {
          // Mobile layout: text + inline-figure per step.
          // Each [data-slide] item gets its content and resolved figure clone
          // wrapped together inside a `.scrolly-mobile-text` block.
          // The original `.figure-col` is hidden.
          // On breakpoint change back to desktop, the cleanup function
          // unwraps everything and removes clones.

          const figureCol = $('.figure-col', scrolly);
          const allSlides = $$('.text-col [data-slide]', scrolly);
          const cloneRefs   = [];   // figure clones to remove on cleanup
          const wrapRefs    = [];   // text wrappers to unwrap on cleanup
          const listForce   = [];   // nested lists we forced open

          if (figureCol) figureCol.dataset.scrollyMobileHidden = '1';

          allSlides.forEach((p) => {
            // Idempotency guard: if already restructured, skip.
            if (p.querySelector(':scope > .scrolly-mobile-text')) return;

            const figIdx = resolveFigIdx(p.dataset.slide);
            const isSkip = p.hasAttribute('data-skip') || p.hasAttribute('data-disabled');

            const nestedList = p.querySelector(':scope > ol, :scope > ul');
            const textWrap = document.createElement('div');
            textWrap.className = 'scrolly-mobile-text';
            if (isSkip) textWrap.classList.add('scrolly-mobile-skip');

            // Move every non-list child into the wrap.
            const moved = [];
            Array.from(p.childNodes).forEach((node) => {
              if (node === nestedList) return;
              moved.push(node);
              textWrap.appendChild(node);
            });

            // Insert figure clone INSIDE the text wrap (skipped section-headers don't get one).
            if (!isSkip && figIdx >= 0 && figureSlides[figIdx]) {
              const figClone = figureSlides[figIdx].cloneNode(true);
              figClone.classList.remove('figure-slide');
              figClone.classList.remove('is-active');
              figClone.classList.add('scrolly-mobile-figure');
              // Strip `.reveal` from clones — its hidden state is tied to the
              // original's ScrollTrigger, which doesn't apply here.
              const stripReveal = (el) => {
                if (el.classList.contains('reveal')) {
                  el.classList.remove('reveal');
                  el.style.opacity = '1';
                  el.style.transform = 'none';
                }
              };
              stripReveal(figClone);
              $$('.reveal', figClone).forEach(stripReveal);
              textWrap.appendChild(figClone);
              cloneRefs.push(figClone);
            }

            if (nestedList) p.insertBefore(textWrap, nestedList);
            else p.appendChild(textWrap);
            wrapRefs.push({ wrap: textWrap, parent: p, moved });
          });

          // Force-open any nested lists that the desktop collapse rule would hide.
          $$('.text-col li > ol, .text-col li > ul', scrolly).forEach((list) => {
            if (list.dataset.scrollyMobileForceOpen) return;
            list.dataset.scrollyMobileForceOpen = '1';
            listForce.push(list);
          });

          // Sync .is-active-para (for highlighting) as user scrolls.
          textParagraphs.forEach((p, i) => {
            ScrollTrigger.create({
              trigger: p,
              start: 'top 50%',
              end:   'bottom 50%',
              onEnter:     () => setActiveSlide(i),
              onEnterBack: () => setActiveSlide(i),
            });
          });
          setActiveSlide(0);

          return () => {
            cloneRefs.forEach((el) => el.remove());
            wrapRefs.forEach(({ wrap, parent }) => {
              while (wrap.firstChild) parent.insertBefore(wrap.firstChild, wrap);
              wrap.remove();
            });
            listForce.forEach((list) => { delete list.dataset.scrollyMobileForceOpen; });
            if (figureCol) delete figureCol.dataset.scrollyMobileHidden;
          };
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
        onEnterBack: () => {
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
    const holders = $$('.svg-holder[data-svg-src]');
    if (!holders.length) return;
    let loadedCount = 0;

    holders.forEach((holder) => {
      const url     = holder.dataset.svgSrc;
      const stagger = num(holder.dataset.svgStagger, 0.025);
      const fade    = num(holder.dataset.svgFade,    0.6);
      const draw    = num(holder.dataset.svgDraw,    1.2);
      const start   = holder.dataset.svgStart || 'top 75%';

      fetch(url).then((r) => r.text()).then((text) => {
        holder.innerHTML = text;
        const svg = holder.querySelector('svg');
        if (!svg) { loadedCount++; return; }
        animate(svg);
        loadedCount++;
        // All SVGs loaded — refresh ScrollTrigger positions. SVG injection
        // changes holder heights, and holders inside pinned containers
        // (e.g. scrolly) need recalculation after the pin is engaged.
        if (loadedCount === holders.length) {
          requestAnimationFrame(() => ScrollTrigger.refresh());
        }
      }).catch((err) => {
        loadedCount++;
        console.warn('[paper.js] external SVG fetch failed (' + err.message +
          '). Falling back to <img>. Run a local server (e.g. `python3 -m http.server`) for the animated version.');
        holder.innerHTML =
          '<img src="' + url + '" style="width:100%;height:auto;display:block;" alt="">' +
          '<p style="text-align:center;color:#9ca3af;font-size:12px;margin-top:8px;">' +
          '(static fallback — animation requires HTTP, not <code>file://</code>.)</p>';
        if (loadedCount === holders.length) {
          requestAnimationFrame(() => ScrollTrigger.refresh());
        }
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

        // Fade+draw children in immediately.  The holder's .reveal ScrollTrigger
        // (created by initReveal) already correctly gates show/hide on scroll,
        // including when the page loads mid-page.  A separate ScrollTrigger on
        // the same element inside a pinned container (scrolly) can get wildly
        // wrong positions when created after the pin is engaged.
        gsap.fromTo(children, { opacity: 0 }, { opacity: 1, duration: fade, stagger, ease: 'power2.out' });
        if (drawables.length) {
          gsap.to(drawables.map((d) => d.p), { strokeDashoffset: 0, duration: draw, stagger, ease: 'power2.out' });
        }
      }
    });
  }

  /* ============================================================
     SVG Fullscreen expand (mobile).  Markup:
       <div class="svg-holder" data-svg-src="…" data-fullscreen></div>
     On mobile (≤720px), a small "⛶ Expand" button appears after the
     SVG.  Tapping it opens the SVG in a landscape-oriented fullscreen
     overlay so wide diagrams are readable.
     ============================================================ */
  function initSvgFullscreen() {
    const isMobile = () => window.matchMedia('(max-width: 720px)').matches;
    const holders = $$('.svg-holder[data-fullscreen]');
    if (!holders.length) return;

    holders.forEach((holder) => {
      // Inject expand button after holder.
      let btn = holder.nextElementSibling;
      if (!btn || !btn.classList.contains('svg-expand-btn')) {
        btn = document.createElement('button');
        btn.className = 'svg-expand-btn';
        btn.type = 'button';
        btn.textContent = '⛶ Expand';
        holder.insertAdjacentElement('afterend', btn);
      }

      // Build fullscreen overlay once, lazily.
      let overlay = null;
      let content = null;
      let closeBtn = null;

      function ensureOverlay() {
        if (overlay) return;
        overlay = document.createElement('div');
        overlay.className = 'svg-fs-overlay';
        content = document.createElement('div');
        content.className = 'svg-fs-content';
        closeBtn = document.createElement('button');
        closeBtn.className = 'svg-fs-close';
        closeBtn.type = 'button';
        closeBtn.innerHTML = '&times;';
        closeBtn.setAttribute('aria-label', 'Close');
        overlay.appendChild(closeBtn);
        overlay.appendChild(content);
        document.body.appendChild(overlay);

        closeBtn.addEventListener('click', close);
        overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
        document.addEventListener('keydown', onKey);
      }

      function onKey(e) {
        if (e.key === 'Escape' && overlay && overlay.classList.contains('is-open')) close();
      }

      function open() {
        ensureOverlay();
        // Clone the current SVG into the overlay.
        const svg = holder.querySelector('svg');
        if (!svg) return;
        content.innerHTML = '';
        const clone = svg.cloneNode(true);
        clone.removeAttribute('width');
        clone.removeAttribute('height');
        content.appendChild(clone);
        // Determine landscape mode: if viewport is portrait (height > width)
        // and the SVG's natural aspect ratio is wider than tall, rotate.
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const vb = clone.viewBox && clone.viewBox.baseVal
          ? clone.viewBox.baseVal
          : null;
        const svgW = (vb && vb.width)  || vw;
        const svgH = (vb && vb.height) || vh;
        const wideSVG = svgW > svgH;
        if (vh > vw && wideSVG) {
          overlay.classList.add('is-landscape');
        } else {
          overlay.classList.remove('is-landscape');
        }
        overlay.classList.add('is-open');
        document.body.style.overflow = 'hidden';
      }

      function close() {
        if (!overlay) return;
        overlay.classList.remove('is-open');
        overlay.classList.remove('is-landscape');
        document.body.style.overflow = '';
      }

      btn.addEventListener('click', open);
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
      // Support both inline JSON (<script type="application/json">)
      // and external JSON loaded via data-src.
      const src = root.dataset.src;
      const cfgScript = $('script[type="application/json"]', root);
      if (cfgScript) {
        let configs = [];
        try { configs = JSON.parse(cfgScript.textContent); } catch (e) { console.error('[paper.js] explorer JSON parse failed', e); return; }
        if (Array.isArray(configs) && configs.length) buildExplorer(root, configs);
      } else if (src) {
        fetch(src)
          .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
          .then((configs) => { if (Array.isArray(configs) && configs.length) buildExplorer(root, configs); })
          .catch((e) => console.error('[paper.js] explorer fetch failed for ' + src, e));
      }
    });

    function buildExplorer(root, configs) {
      // Apply grid-gap from HTML attribute (e.g. data-grid-gap="6").
      // Sets CSS custom properties that .detail-grid / .detail-row read.
      if (root.dataset.gridGap != null) {
        const g = parseFloat(root.dataset.gridGap);
        if (!isNaN(g) && g >= 0) {
          root.style.setProperty('--explorer-cell-gap', g + 'px');
          root.style.setProperty('--explorer-row-gap',  (g * 2) + 'px');
        }
      }
      // Apply border-radius from HTML attribute (e.g. data-grid-radius="8").
      if (root.dataset.gridRadius != null) {
        const r = parseFloat(root.dataset.gridRadius);
        if (!isNaN(r) && r >= 0) {
          root.style.setProperty('--explorer-radius', r + 'px');
        }
      }

      // Build skeleton.
      root.innerHTML = `
        <div class="thumb-strip embla" data-lenis-prevent>
          <div class="embla__container"></div>
          <button class="thumb-nav prev" aria-label="prev">‹</button>
          <button class="thumb-nav next" aria-label="next">›</button>
        </div>
        <div class="detail">
          <div class="detail-header">
            <div class="detail-title">—</div>
            <div class="detail-toolbar">
              <button class="anim-toggle" type="button" aria-pressed="false" title="Toggle animation mode">▶ Animate</button>
            </div>
          </div>
          <div class="col-labels-host"></div>
          <div class="detail-grid"></div>
        </div>
      `;

      const stripContainer = $('.thumb-strip .embla__container', root);
      const detailTitle    = $('.detail-title', root);
      const detailGrid     = $('.detail-grid',  root);
      const colLabelsEl    = $('.col-labels-host', root);
      const animToggleBtn  = $('.anim-toggle', root);
      let activeIdx = -1;
      let explorerAnimToggle = false;

      configs.forEach((cfg, i) => {
        const slide = document.createElement('div');
        slide.className = 'embla__slide';
        if (cfg.thumb.src) {
          slide.innerHTML = `<div class="thumb" data-thumb-idx="${i}"><img src="${cfg.thumb.src}" alt=""></div>`;
        } else {
          slide.innerHTML = `<div class="thumb" data-thumb-idx="${i}" style="background:${cfg.thumb.color};">${cfg.thumb.label || ''}</div>`;
        }
        stripContainer.appendChild(slide);
      });

      const stripEl = $('.thumb-strip', root);
      const embla   = EmblaCarousel(stripEl, { dragFree: true, containScroll: 'trimSnaps' });
      $('.thumb-nav.prev', root).addEventListener('click', () => embla.scrollPrev());
      $('.thumb-nav.next', root).addEventListener('click', () => embla.scrollNext());

      // Mouse-wheel → horizontal scroll on the thumb strip
      stripEl.addEventListener('wheel', (e) => {
        if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) return; // native horizontal
        e.preventDefault();
        if (e.deltaY > 0) embla.scrollNext(); else embla.scrollPrev();
      }, { passive: false });

      let activeIntervals = [];
      function clearDetail() {
        activeIntervals.forEach(clearInterval);
        activeIntervals = [];
        detailGrid.innerHTML  = '';
        colLabelsEl.innerHTML = '';
        colLabelsEl.className = 'col-labels-host';
      }

      function applyMaxSize(media, img, cell) {
        if (cell.maxWidth) img.style.maxWidth = cell.maxWidth;
        if (cell.maxHeight) img.style.maxHeight = cell.maxHeight;
        if (cell.maxWidth || cell.maxHeight) {
          img.style.width = 'auto';
          img.style.height = 'auto';
          img.style.objectFit = 'contain';
          // Shrink the media container to fit the constrained image
          media.style.aspectRatio = 'auto';
          media.style.flex = 'none';
          media.style.alignSelf = 'center';
          // Mark the media so renderDetail can shrink the parent cell
          media.dataset.constrained = 'true';
        }
      }

      function buildCellMedia(cell) {
        const isWeChat = /MicroMessenger/i.test(navigator.userAgent || '');

        // Helper: show fallback text when a media element fails to load.
        function showMediaError(container, kind) {
          container.innerHTML = '';
          const msg = document.createElement('div');
          msg.style.cssText = 'display:flex;align-items:center;justify-content:center;width:100%;height:100%;padding:16px;text-align:center;font-size:13px;color:var(--muted);background:var(--bg-soft);';
          let text = kind === 'video' ? 'Video not rendered' : 'Image not rendered';
          if (isWeChat && kind === 'video') {
            text = 'WeChat cannot play this video, please open in browser';
          }
          msg.innerHTML = text;
          container.appendChild(msg);
        }

        const media = document.createElement('div');
        media.className = 'media';
        if (cell.kind === 'color') {
          media.style.background = cell.color;
          media.textContent = cell.label || '';
        } else if (cell.kind === 'image') {
          const img = document.createElement('img');
          img.src = cell.src; img.alt = cell.label || '';
          media.classList.add('is-loading');
          img.addEventListener('load',  () => media.classList.remove('is-loading'), { once: true });
          img.addEventListener('error', () => { media.classList.remove('is-loading'); showMediaError(media, 'image'); }, { once: true });
          media.appendChild(img);
          applyMaxSize(media, img, cell);
        } else if (cell.kind === 'video') {
          const v = document.createElement('video');
          v.autoplay = true; v.loop = true; v.muted = true; v.playsInline = true;
          v.setAttribute('playsinline', '');
          v.setAttribute('webkit-playsinline', '');
          v.setAttribute('x5-video-player-type', 'h5');
          v.setAttribute('x5-video-player-fullscreen', 'true');
          v.setAttribute('preload', 'auto');
          v.setAttribute('disableRemotePlayback', '');
          media.classList.add('is-loading');
          const hideLoader = () => { media.classList.remove('is-loading'); };
          // Set src AFTER x5 attrs so X5 browser (Android WeChat) picks them up.
          v.src = cell.src;
          media.appendChild(v);
          // WeChat can't play inline video — show error immediately.
          if (isWeChat) {
            hideLoader();
            showMediaError(media, 'video');
          } else {
            let errorShown = false;
            const fail = () => { if (!errorShown) { errorShown = true; hideLoader(); showMediaError(media, 'video'); } };
            v.addEventListener('error', fail, { once: true });
            v.addEventListener('loadedmetadata', hideLoader, { once: true });
            const tryPlay = () => { v.play().catch(fail); };
            if (v.readyState >= 1) { hideLoader(); tryPlay(); } else { v.addEventListener('loadedmetadata', tryPlay, { once: true }); }
            // Timeout fallback — if video still hasn't started after 4s, show error.
            setTimeout(() => { if (!errorShown && v.paused && v.currentTime === 0) fail(); }, 4000);
          }
          applyMaxSize(media, v, cell);
        } else if (cell.kind === 'sequence') {
          // Stack all frames as layered, pre-loaded children of `media` and
          // cycle by toggling opacity. Avoids the white-flash that happens
          // when swapping a single <img>'s src each tick (the browser briefly
          // shows the old image at unknown decode state). All frames are
          // decoded once up front, so cycling is just a CSS toggle.
          if (!cell.frames || !cell.frames.length) return media;
          media.classList.add('media-sequence');
          media.textContent = '';

          const layers = cell.frames.map((fr, i) => {
            let layer;
            if (fr.src) {
              layer = document.createElement('img');
              layer.src     = fr.src;
              layer.alt     = '';
              layer.decoding = 'async';
              layer.loading = i === 0 ? 'eager' : 'lazy';
              layer.addEventListener('error', () => showMediaError(media, 'image'), { once: true });
            } else {
              layer = document.createElement('div');
              layer.style.background = fr.color || '#000';
            }
            layer.classList.add('media-frame');
            if (i !== 0) layer.classList.add('is-hidden');
            media.appendChild(layer);
            return layer;
          });

          // Honor maxWidth / maxHeight if the cell sets them — only the
          // first layer drives the size; subsequent layers fill that box.
          if (cell.maxWidth || cell.maxHeight) {
            applyMaxSize(media, layers[0], cell);
            for (let i = 1; i < layers.length; i++) {
              if (cell.maxWidth)  layers[i].style.maxWidth  = cell.maxWidth;
              if (cell.maxHeight) layers[i].style.maxHeight = cell.maxHeight;
            }
          }

          let f = 0;

          // Time-step label overlay (if frameLabels provided)
          let labelEl = null;
          if (cell.frameLabels && cell.frameLabels.length) {
            labelEl = document.createElement('div');
            labelEl.className = 'sequence-label';
            labelEl.style.cssText = 'position:absolute;bottom:4px;left:0;right:0;text-align:center;font-size:11px;color:#fff;background:rgba(0,0,0,0.55);padding:2px 6px;pointer-events:none;z-index:2;';
            labelEl.textContent = cell.frameLabels[0] || '';
            media.appendChild(labelEl);
          }

          const tick = () => {
            const next = (f + 1) % layers.length;
            layers[f].classList.add('is-hidden');
            layers[next].classList.remove('is-hidden');
            if (labelEl && cell.frameLabels) {
              labelEl.textContent = cell.frameLabels[next] || '';
            }
            f = next;
          };
          activeIntervals.push(setInterval(tick, cell.interval || 400));
        }
        return media;
      }

      // Animation mode: turn a column-major grid (rows = time-step, cols = method)
      // into an animated GIF layout. Two layouts are supported:
      //   merge: "columns"  → transpose: each column becomes its own row, the row's
      //                       single cell cycles through that column's frames.
      //   merge: "rows"     → collapse non-skip rows into a single row of GIF cells.
      // The input row(s) listed in `animation.skipRows` (default [0]) are preserved
      // as-is so static reference content (input frames, action chunks) stays visible.
      function synthesizeAnimatedConfig(cfg, merge, animCfg) {
        const skipRows = animCfg.skipRows || [0];
        const interval = animCfg.interval || 600;

        const animRows = cfg.rows.map((row, idx) => ({ row, idx }))
          .filter(({ idx }) => !skipRows.includes(idx));
        if (!animRows.length) return cfg;

        const firstAnimRow = animRows[0].row;
        const colCount     = firstAnimRow.cells ? firstAnimRow.cells.length : 0;
        if (!colCount) return cfg;

        const colLabels = (firstAnimRow.colLabels && firstAnimRow.colLabels.length === colCount)
          ? firstAnimRow.colLabels
          : (cfg.colLabels && cfg.colLabels.length === colCount ? cfg.colLabels : null);

        // Convert <br> to a space so multi-line column labels collapse cleanly
        // into a single-line row-label. Other HTML (e.g. <strong>, emoji) is
        // preserved — the row-label is rendered via innerHTML.
        const flattenLabel = (s) => (s || '').replace(/<br\s*\/?>/gi, ' ');

        const buildColumnSequenceCell = (c) => {
          const frames = [];
          const frameLabels = [];
          animRows.forEach(({ row, idx }) => {
            const cell = row.cells && row.cells[c];
            if (!cell) return;
            if (cell.kind === 'image' && cell.src)        frames.push({ src: cell.src });
            else if (cell.kind === 'color' && cell.color) frames.push({ color: cell.color });
            // 'video' and 'sequence' kinds aren't fanned out — they animate themselves.
            const rawLabel = (cfg.rowLabels && cfg.rowLabels[idx]) || '';
            frameLabels.push(rawLabel.replace(/<br\s*\/?>/gi, ' ').trim());
          });
          if (!frames.length) return null;
          const sample = animRows[0].row.cells[c] || {};
          const out = { kind: 'sequence', frames, interval, frameLabels };
          if (sample.maxWidth)  out.maxWidth  = sample.maxWidth;
          if (sample.maxHeight) out.maxHeight = sample.maxHeight;
          return out;
        };

        if (merge === 'columns') {
          const newRows = [];
          const newRowLabels = [];
          // Preserve skipped rows (e.g. input row) at the top.
          skipRows.forEach((sIdx) => {
            if (cfg.rows[sIdx]) {
              newRows.push(cfg.rows[sIdx]);
              newRowLabels.push((cfg.rowLabels && cfg.rowLabels[sIdx]) || '');
            }
          });
          // Then one row per column.
          for (let c = 0; c < colCount; c++) {
            const seqCell = buildColumnSequenceCell(c);
            if (!seqCell) continue;
            newRows.push({ cells: [seqCell] });
            newRowLabels.push(colLabels ? flattenLabel(colLabels[c]) : '');
          }
          return Object.assign({}, cfg, {
            rows: newRows,
            rowLabels: newRowLabels,
            colLabels: undefined,
          });
        }

        if (merge === 'rows') {
          const seqCells = [];
          for (let c = 0; c < colCount; c++) {
            const seqCell = buildColumnSequenceCell(c);
            if (seqCell) {
              // Embed column label as a per-cell caption (desktop only; mobile uses 'columns').
              if (colLabels && colLabels[c]) {
                seqCell.caption = flattenLabel(colLabels[c]);
              }
              seqCells.push(seqCell);
            }
          }
          const newRows = [];
          const newRowLabels = [];
          skipRows.forEach((sIdx) => {
            if (cfg.rows[sIdx]) {
              newRows.push(cfg.rows[sIdx]);
              newRowLabels.push((cfg.rowLabels && cfg.rowLabels[sIdx]) || '');
            }
          });
          if (seqCells.length) {
            newRows.push({ cells: seqCells });
            newRowLabels.push(animCfg.mergedRowLabel || 'Predictions');
          }
          return Object.assign({}, cfg, {
            rows: newRows,
            rowLabels: newRowLabels,
            colLabels: undefined,
          });
        }

        return cfg;
      }

      function renderDetail(srcCfg) {
        clearDetail();
        // Determine effective animation flags.
        const isMobile      = window.matchMedia('(max-width: 720px)').matches;
        const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        const animCfg = srcCfg.animation || {};
        const gifOn   = !reducedMotion && (isMobile || animCfg.gif === true || explorerAnimToggle);
        let   merge   = false;
        if (gifOn) {
          // On mobile we always merge (default columns) regardless of toggle.
          // On desktop we respect the toggle and the JSON config, but default
          // to 'rows' so all methods stay in a single compact row.
          if (isMobile)                     merge = animCfg.merge || 'columns';
          else if (animCfg.merge)           merge = animCfg.merge;
          else if (explorerAnimToggle)      merge = 'rows';
        }
        const cfg = (gifOn && merge) ? synthesizeAnimatedConfig(srcCfg, merge, animCfg) : srcCfg;
        // Mark the explorer root so CSS can adjust spacing/labels for narrow rows.
        root.dataset.animMode = (gifOn && merge) ? merge : 'static';

        detailTitle.innerHTML = cfg.title || '';

        if (cfg.colLabels && cfg.colLabels.length) {
          colLabelsEl.className = 'col-labels';
          cfg.colLabels.forEach((label) => {
            const s = document.createElement('span'); s.innerHTML = label; colLabelsEl.appendChild(s);
          });
        }

        const cellNodes   = [];
        const rowsAsCells = [];
        cfg.rows.forEach((row, ri) => {
          const rowEl = document.createElement('div'); rowEl.className = 'detail-row';
          // Opt-in: rows with `stackOnMobile: true` switch their cells from
          // horizontal to vertical layout on ≤720px (CSS reads this attr).
          if (row.stackOnMobile) rowEl.dataset.stackMobile = '1';
          if (cfg.rowLabels) {
            const lab = document.createElement('div'); lab.className = 'row-label';
            lab.innerHTML = cfg.rowLabels[ri] || '';
            rowEl.appendChild(lab);
          }
          // Wrapper for column labels + cells
          const rowBody = document.createElement('div'); rowBody.className = 'row-body';
          // Per-row column labels (optional)
          if (row.colLabels && row.colLabels.length) {
            const cl = document.createElement('div'); cl.className = 'row-col-labels';
            row.colLabels.forEach((label) => {
              const s = document.createElement('span'); s.innerHTML = label; cl.appendChild(s);
            });
            rowBody.appendChild(cl);
          }
          const cellsWrap = document.createElement('div'); cellsWrap.className = 'cells';
          // Center-align cells in the row (default: true)
          if (cfg.centerCells !== false) cellsWrap.style.justifyContent = 'center';
          const rowCells = [];
          row.cells.forEach((cell) => {
            const cellEl = document.createElement('div'); cellEl.className = 'detail-cell';
            const mediaEl = buildCellMedia(cell);
            cellEl.appendChild(mediaEl);
            // If the media was constrained by maxWidth/maxHeight, shrink the cell too
            if (mediaEl.dataset.constrained === 'true') cellEl.style.flex = 'none';
            if (cell.caption) {
              const cap = document.createElement('div'); cap.className = 'caption';
              cap.innerHTML = cell.caption;
              cellEl.appendChild(cap);
            }
            cellsWrap.appendChild(cellEl);
            cellNodes.push(cellEl);
            rowCells.push(cellEl);
          });
          rowBody.appendChild(cellsWrap);
          rowEl.appendChild(rowBody);
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
        if (idx === activeIdx) return;
        activeIdx = idx;
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

      // Toolbar: animation toggle (desktop only — mobile force-on via CSS hide).
      if (animToggleBtn) {
        animToggleBtn.addEventListener('click', () => {
          explorerAnimToggle = !explorerAnimToggle;
          animToggleBtn.setAttribute('aria-pressed', String(explorerAnimToggle));
          animToggleBtn.textContent = explorerAnimToggle ? '⏸ Static' : '▶ Animate';
          renderDetail(configs[activeIdx]);
        });
      }

      // Re-render when the breakpoint changes (mobile↔desktop) so the
      // animation-mode forced state re-evaluates.
      let lastIsMobile = window.matchMedia('(max-width: 720px)').matches;
      window.addEventListener('resize', () => {
        const nowIsMobile = window.matchMedia('(max-width: 720px)').matches;
        if (nowIsMobile !== lastIsMobile) {
          lastIsMobile = nowIsMobile;
          renderDetail(configs[activeIdx]);
        }
      }, { passive: true });

      activate(0);
    }
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

      // Apply highlight.js syntax highlighting to the <pre> inside.
      if (typeof hljs !== 'undefined') {
        const pre = block.querySelector('pre');
        if (pre) {
          // If there's no <code> child, wrap the text in one.
          let code = pre.querySelector('code');
          if (!code) {
            code = document.createElement('code');
            code.textContent = pre.textContent;
            pre.textContent = '';
            pre.appendChild(code);
          }
          // Set language from data-lang if available
          if (block.dataset.lang && block.dataset.lang !== 'BibTeX') {
            code.classList.add('language-' + block.dataset.lang.toLowerCase());
          }
          try { hljs.highlightElement(code); } catch (e) { /* ignore */ }
        }
      }
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
     SVG long-press on mobile — tap shows hint tooltip,
     long-press opens the SVG in a fullscreen overlay.
     ============================================================ */
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
    // Refresh after scrolly pinning — pin-spacer changes the layout, so
    // ScrollTrigger positions created before the pin need recalculation.
    ScrollTrigger.refresh();
    initEmblas();
    initSvgFlows();
    initExternalSvgs();
    initExplorers();
    initCodeBlocks();
    initCompareSliders();
    initVideoPlayers();
    initSvgFullscreen();
    if (typeof initMarkdown === 'function') initMarkdown();
    if (typeof initMarkdownTables === 'function') initMarkdownTables();
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
    initCodeBlocks, initCompareSliders, initVideoPlayers, initSvgFullscreen,
    initMarkdown: typeof initMarkdown === 'function' ? initMarkdown : null,
    initMarkdownTables: typeof initMarkdownTables === 'function' ? initMarkdownTables : null,
  };
})();
