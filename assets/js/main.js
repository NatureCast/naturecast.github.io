/* =========================================================
   NatureCast — Main JavaScript
   Neural Network Canvas Animation + UI Interactions
   ========================================================= */

'use strict';

/* ---- Mobile nav toggle ---- */
(function () {
  const toggle = document.querySelector('.nav-toggle');
  const nav    = document.querySelector('.site-nav');
  if (!toggle || !nav) return;

  toggle.addEventListener('click', function () {
    const open = nav.classList.toggle('open');
    toggle.setAttribute('aria-expanded', open);
    // Animate hamburger lines
    const spans = toggle.querySelectorAll('span');
    if (open) {
      spans[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
      spans[1].style.opacity   = '0';
      spans[2].style.transform = 'rotate(-45deg) translate(5px, -5px)';
    } else {
      spans[0].style.transform = '';
      spans[1].style.opacity   = '';
      spans[2].style.transform = '';
    }
  });

  // Close on outside click
  document.addEventListener('click', function (e) {
    if (!toggle.contains(e.target) && !nav.contains(e.target)) {
      nav.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
      const spans = toggle.querySelectorAll('span');
      spans[0].style.transform = '';
      spans[1].style.opacity   = '';
      spans[2].style.transform = '';
    }
  });
}());

/* ---- Active nav link ---- */
(function () {
  const path  = window.location.pathname;
  const links = document.querySelectorAll('.site-nav__link');
  links.forEach(function (link) {
    const href = link.getAttribute('href');
    if (href === '/' && path === '/') {
      link.classList.add('active');
    } else if (href !== '/' && path.startsWith(href)) {
      link.classList.add('active');
    }
  });
}());

/* ---- Filter tabs (projects / blog) ---- */
(function () {
  const tabs = document.querySelectorAll('.filter-tab');
  if (!tabs.length) return;

  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      tabs.forEach(function (t) { t.classList.remove('active'); });
      tab.classList.add('active');

      const filter = tab.dataset.filter;
      const items  = document.querySelectorAll('[data-category]');
      items.forEach(function (item) {
        if (filter === 'all' || item.dataset.category === filter) {
          item.style.display = '';
        } else {
          item.style.display = 'none';
        }
      });
    });
  });
}());

/* ---- Smooth count-up for hero stats ---- */
(function () {
  const stats = document.querySelectorAll('[data-count]');
  if (!stats.length) return;

  function countUp(el) {
    const target = parseInt(el.dataset.count, 10);
    const suffix = el.dataset.suffix || '';
    const dur    = 1400;
    const start  = performance.now();

    function step(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / dur, 1);
      // Ease out
      const val = Math.floor(progress * target);
      el.textContent = val + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  const observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        countUp(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });

  stats.forEach(function (el) { observer.observe(el); });
}());

/* ---- Neural Network Canvas Animation ---- */
(function () {
  const canvas = document.getElementById('neural-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let W, H, nodes, animId;

  /* Node constructor */
  function Node(x, y) {
    this.x  = x;
    this.y  = y;
    this.vx = (Math.random() - 0.5) * 0.35;
    this.vy = (Math.random() - 0.5) * 0.35;
    this.r  = Math.random() * 2.5 + 1.5;
    this.pulse = Math.random() * Math.PI * 2;
    this.pulseSpeed = Math.random() * 0.025 + 0.01;
    // Assign to a "layer" for vertical gradient appearance
    this.layer = Math.floor(Math.random() * 5);
  }

  function resize() {
    W = canvas.width  = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  }

  function buildNodes() {
    const count = Math.floor((W * H) / 14000);
    nodes = [];
    for (let i = 0; i < count; i++) {
      nodes.push(new Node(Math.random() * W, Math.random() * H));
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Update & draw nodes
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      n.x += n.vx;
      n.y += n.vy;
      n.pulse += n.pulseSpeed;

      // Bounce
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;

      const pulseR = n.r + Math.sin(n.pulse) * 1.2;
      const alpha  = 0.55 + Math.sin(n.pulse) * 0.25;

      ctx.beginPath();
      ctx.arc(n.x, n.y, pulseR, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(126,207,160,${alpha})`;
      ctx.fill();
    }

    // Draw edges
    const maxDist = 120;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < maxDist) {
          const alpha = (1 - d / maxDist) * 0.4;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          // Gradient colour based on distance
          ctx.strokeStyle = `rgba(45,122,79,${alpha})`;
          ctx.lineWidth   = (1 - d / maxDist) * 1.5;
          ctx.stroke();
        }
      }
    }

    animId = requestAnimationFrame(draw);
  }

  function init() {
    resize();
    buildNodes();
    if (animId) cancelAnimationFrame(animId);
    draw();
  }

  init();

  let resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(init, 150);
  });
}());

/* ---- Scroll-reveal (simple fade-in) ---- */
(function () {
  const revealEls = document.querySelectorAll('[data-reveal]');
  if (!revealEls.length) return;

  const observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  revealEls.forEach(function (el) {
    el.style.opacity    = '0';
    el.style.transform  = 'translateY(18px)';
    el.style.transition = 'opacity 0.55s ease, transform 0.55s ease';
    observer.observe(el);
  });

  // Add revealed state styles via JS
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.revealed').forEach(function (el) {
      el.style.opacity   = '1';
      el.style.transform = 'translateY(0)';
    });
  });

  // Patch to actually apply styles on reveal
  const origFn = observer.constructor.prototype;
  // Simpler: add class and use CSS
})();

/* ---- Staggered reveal for grid items ---- */
document.addEventListener('DOMContentLoaded', function () {
  const style = document.createElement('style');
  style.textContent = `
    [data-reveal] { opacity: 0; transform: translateY(20px); transition: opacity .55s ease, transform .55s ease; }
    [data-reveal].is-visible { opacity: 1; transform: translateY(0); }
  `;
  document.head.appendChild(style);

  const els = document.querySelectorAll('[data-reveal]');
  const obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry, i) {
      if (entry.isIntersecting) {
        setTimeout(function () {
          entry.target.classList.add('is-visible');
        }, (entry.target.dataset.delay || 0) * 1);
        obs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  els.forEach(function (el, i) {
    el.dataset.delay = i * 80;
    obs.observe(el);
  });
});
