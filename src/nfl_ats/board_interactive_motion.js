/* Motion and lighting over the currently published board. */
(() => {
  'use strict';
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const inspector = $('.inspector-col');
  if (!inspector) return;
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const palettes = {
    ARI: '#e69c9c', ATL: '#e5a39e', BAL: '#b7a8eb', BUF: '#85b9ee',
    CAR: '#7dcced', CHI: '#e9b789', CIN: '#edb083', CLE: '#e2ab85',
    DAL: '#a9c6e7', DEN: '#efb089', DET: '#8ec9eb', GB: '#c8d99b',
    HOU: '#90bbd7', IND: '#a0bfee', JAX: '#84cec1', KC: '#f0aaaa',
    LA: '#a1c8ee', LAC: '#99d9ef', LV: '#d0d7dc', MIA: '#80ddcd',
    MIN: '#c5a9ed', NE: '#a9bbd9', NO: '#ded2a1', NYG: '#a4bee8',
    NYJ: '#93c9ae', PHI: '#8bc5b5', PIT: '#ebcf87', SEA: '#b2d7a0',
    SF: '#e8b6a0', TB: '#e7a9a3', TEN: '#a2c9e7', WAS: '#e1b898'
  };
  const tint = team => palettes[team] || '#bacbc4';
  const ns = 'http://www.w3.org/2000/svg';
  let animations = [];
  let lastGame = $('.dive-panel:not([hidden])')?.id;
  let selectionSequence = 0;

  function stopAnimations() {
    animations.forEach(animation => animation.cancel());
    animations = [];
  }
  function animate(el, frames, options) {
    if (!el || reduceMotion.matches) return;
    const animation = el.animate(frames, { duration: 240, easing: 'cubic-bezier(.2,.75,.2,1)', ...options });
    animations.push(animation);
    const release = () => { animations = animations.filter(item => item !== animation); };
    animation.finished.then(release, release);
  }
  function paintTheme(panel) {
    if (!panel) return;
    const [, , away, home] = panel.id.split('_');
    const field = $('.merged-field', panel);
    inspector.style.setProperty('--away-light', tint(away));
    inspector.style.setProperty('--home-light', tint(home));
    inspector.style.setProperty('--team-light', tint(field.dataset.offenseTeam));
    // Keep two legible marker categories, even when the teams share similar colors.
    inspector.style.setProperty('--opponent-light', '#e4bd7e');
    $('.dive-head', panel).dataset.homeTeam = home;
  }

  function decorateField(panel, animateChange = false) {
    const field = $('.merged-field', panel);
    const svg = $('svg', field);
    if (svg.dataset.stadiumReady === 'true') return;
    svg.dataset.stadiumReady = 'true';
    const key = panel.id.replaceAll('_', '-') + '-lit';
    const defs = document.createElementNS(ns, 'defs');
    defs.innerHTML = `
      <linearGradient id="${key}-turf" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#2d5135"/><stop offset=".52" stop-color="#23452e"/><stop offset="1" stop-color="#193820"/>
      </linearGradient>
      <radialGradient id="${key}-light" cx="50%" cy="0%" r="90%">
        <stop offset="0" stop-color="#d3e8b4" stop-opacity=".15"/><stop offset=".65" stop-color="#bcd9ab" stop-opacity=".025"/><stop offset="1" stop-color="#071a0c" stop-opacity=".24"/>
      </radialGradient>
      <pattern id="${key}-grass" width="7" height="7" patternUnits="userSpaceOnUse">
        <path d="M1 1v2M5 4v2M3 6v1" stroke="#abc79b" stroke-opacity=".07" stroke-width=".7"/>
      </pattern>
      <radialGradient id="${key}-offense" cx="35%" cy="20%" r="85%">
        <stop offset="0" stop-color="#527d68"/><stop offset=".48" stop-color="#284e3c"/><stop offset="1" stop-color="#163528"/>
      </radialGradient>
      <radialGradient id="${key}-defense" cx="35%" cy="20%" r="85%">
        <stop offset="0" stop-color="#8b7753"/><stop offset=".48" stop-color="#50462e"/><stop offset="1" stop-color="#312b1e"/>
      </radialGradient>`;
    svg.prepend(defs);
    const ground = $('rect', svg);
    ground.setAttribute('fill', `url(#${key}-turf)`);
    const surface = document.createElementNS(ns, 'g');
    surface.setAttribute('pointer-events', 'none');
    surface.innerHTML = `<rect width="720" height="440" fill="url(#${key}-grass)"/>
      <rect width="720" height="440" fill="url(#${key}-light)"/>
      <path d="M20 0V440M700 0V440" stroke="#d4e6c0" stroke-opacity=".37" stroke-width="2"/>`;
    for (let y = 10; y < 440; y += 11) {
      const hash = document.createElementNS(ns, 'path');
      hash.setAttribute('d', `M241 ${y}h5M474 ${y}h5`);
      hash.setAttribute('stroke', '#d4e6c0');
      hash.setAttribute('stroke-opacity', '.26');
      surface.append(hash);
    }
    const firstPlayer = $('.formation-player', svg);
    svg.insertBefore(surface, firstPlayer);
    $$('.formation-player', svg).forEach(player => {
      const disc = $('circle', player);
      const isDefense = player.classList.contains('defender');
      disc.style.fill = `url(#${key}-${isDefense ? 'defense' : 'offense'})`;
      const ring = document.createElementNS(ns, 'circle');
      ring.classList.add('selection-ring');
      ring.setAttribute('cx', disc.getAttribute('cx'));
      ring.setAttribute('cy', disc.getAttribute('cy'));
      ring.setAttribute('r', '24');
      ring.setAttribute('aria-hidden', 'true');
      player.prepend(ring);
    });
    // Clicking or keyboard-selecting a marker keeps its original data handler.
    if (!svg.dataset.stadiumListener) {
      svg.dataset.stadiumListener = 'true';
      svg.addEventListener('click', event => {
        if (!event.target.closest('.formation-player')) return;
        svg.closest('.stadium-stage').classList.add('is-engaged');
        animate($('.merged-player-detail', field), [{ opacity: .55, transform: 'translateY(3px)' }, { opacity: 1, transform: 'none' }], { duration: 160 });
      });
    }
    if (animateChange) {
      animate(svg, [{ opacity: .35 }, { opacity: 1 }], { duration: 220 });
      animate($('.merged-player-detail', field), [{ opacity: .5 }, { opacity: 1 }], { duration: 200 });
    }
    if (!panel.hidden) paintTheme(panel);
  }

  $$('.dive-panel').forEach(panel => {
    const field = $('.merged-field', panel);
    const svg = $('svg', field);
    const stage = document.createElement('div');
    stage.className = 'stadium-stage';
    stage.tabIndex = 0;
    stage.setAttribute('role', 'group');
    stage.setAttribute('aria-label', 'Football field. On a narrow screen, scroll sideways to see both sidelines. Select a player for details.');
    svg.before(stage);
    stage.append(svg);
    const toolbar = document.createElement('div');
    toolbar.className = 'stadium-toolbar';
    toolbar.innerHTML = '<span class="view-caption">THE FIELD / SELECT EITHER SIDE</span><div class="stadium-tools" role="group" aria-label="Choose field presentation"><button type="button" data-field-view="stadium" aria-pressed="true">Stadium</button><button type="button" data-field-view="flat" aria-pressed="false">Flat</button></div>';
    stage.before(toolbar);
    if (innerWidth <= 500) $('.view-caption', toolbar).textContent = 'SWIPE FIELD · TAP A PLAYER';
    $$('.stadium-tools button', toolbar).forEach(button => {
      button.addEventListener('click', () => {
        const flat = button.dataset.fieldView === 'flat';
        stage.classList.toggle('is-flat', flat);
        $$('.stadium-tools button', toolbar).forEach(b => b.setAttribute('aria-pressed', String(b === button)));
      });
    });
    decorateField(panel);
    if (innerWidth <= 500 && !panel.hidden) stage.scrollLeft = (stage.scrollWidth - stage.clientWidth) / 2;
    // The existing possession control replaces the SVG's children synchronously.
    $$('.possession-buttons button', panel).forEach(button => button.addEventListener('click', () => {
      delete svg.dataset.stadiumReady;
      stage.classList.remove('is-engaged');
      decorateField(panel, true);
    }));
  });

  const originalSelect = window.atsSelectGame;
  function selectGame(gameId, scrollOnMobile = false) {
    const panel = document.getElementById(gameId);
    if (!panel?.classList.contains('dive-panel')) return;
    const changed = gameId !== lastGame;
    stopAnimations();
    originalSelect(gameId);
    if (innerWidth <= 500 && changed) {
      const stage = $('.stadium-stage', panel);
      stage.scrollLeft = (stage.scrollWidth - stage.clientWidth) / 2;
    }
    paintTheme(panel);
    lastGame = gameId;
    inspector.dataset.selectionSequence = String(++selectionSequence);
    inspector.dataset.selectedGame = gameId;
    inspector.dispatchEvent(new CustomEvent('ball:gamechange', { detail: { gameId }, bubbles: true }));
    if (changed) {
      animate($('.refined-matchup', panel), [{ opacity: 0, transform: 'translateY(7px)' }, { opacity: 1, transform: 'none' }], { duration: 220 });
      animate($('.original-pick', panel), [{ opacity: .25, transform: 'translateY(5px)' }, { opacity: 1, transform: 'none' }], { duration: 230, delay: 15, fill: 'backwards' });
      animate($('.stadium-stage', panel), [{ opacity: .25, transform: 'translateY(8px)' }, { opacity: 1, transform: 'none' }], { duration: 260, delay: 20, fill: 'backwards' });
      animate($('.merged-player-detail', panel), [{ opacity: .25 }, { opacity: 1 }], { duration: 240, delay: 35, fill: 'backwards' });
    }
    if (scrollOnMobile && innerWidth < 951) inspector.scrollIntoView({ behavior: reduceMotion.matches ? 'auto' : 'smooth', block: 'start' });
  }
  window.atsSelectGame = gameId => selectGame(gameId, false);
  $$('table.board tr.game').forEach(row => {
    row.addEventListener('click', event => {
      if (event.target.closest('[data-board-action]')) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      selectGame(row.dataset.gameId, true);
    }, true);
  });
  reduceMotion.addEventListener('change', () => { if (reduceMotion.matches) stopAnimations(); });
  paintTheme(document.getElementById(lastGame));
  inspector.dataset.selectedGame = lastGame;
})();
