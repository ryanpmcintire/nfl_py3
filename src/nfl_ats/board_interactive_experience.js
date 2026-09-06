/* Interactive published board. Score scenarios never write to forecasts or result ledgers. */
(() => {
  'use strict';
  const data = window.BALL_CARD;
  if (!data) return;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const games = new Map(data.games.map(g => [g.id, Object.freeze({ ...g })]));
  const scenarios = new Map();
  const signed = n => `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n)}`;
  const pickText = g => `${g.pick} ${signed(g.spread)}`;
  const scoresValid = s => s && ['away', 'home'].every(k => Number.isInteger(s[k]) && s[k] >= 0 && s[k] <= 99);
  const element = (tag, className, text) => { const el = document.createElement(tag); if (className) el.className = className; if (text !== undefined) el.textContent = text; return el; };
  function grade(game, scores) {
    if (!scoresValid(scores)) return { outcome: 'INVALID', margin: null, adjusted: null };
    const margin = game.pick === game.home ? scores.home - scores.away : scores.away - scores.home;
    const adjusted = margin + game.spread;
    return { outcome: adjusted > 0 ? 'COVER' : adjusted < 0 ? 'LOSS' : 'PUSH', margin, adjusted };
  }
  function defaultScores(game) { return scenarios.get(game.id) || { away: 21, home: 24 }; }
  function receipt(game, scores = null, kind = 'recorded') {
    const card = element('article', 'ball-receipt');
    const result = scores ? grade(game, scores) : { outcome: 'PENDING' };
    if (scores && result.outcome === 'INVALID') throw new Error('A receipt requires valid final scores.');
    card.dataset.gameId = game.id;
    card.dataset.outcome = result.outcome;
    card.dataset.source = scores && kind === 'demo' ? 'demo' : 'saved';
    if (scores && kind === 'demo') card.append(element('div', 'ball-receipt-demo', 'ILLUSTRATIVE RESULT · NOT A RECORDED GAME'));
    card.append(element('div', 'ball-receipt-label', `WEEK ${String(data.week).padStart(2, '0')} / ${game.kickoff}`));
    card.append(element('h3', '', `${game.away} at ${game.home}`));
    card.append(element('p', 'ball-receipt-original', `Original pick: ${pickText(game)} · Decision score: ${game.score}`));
    const stamp = element('div', 'ball-receipt-outcome', result.outcome === 'PENDING' ? 'AWAITING A FINAL' : result.outcome);
    stamp.dataset.outcome = result.outcome;
    card.append(stamp);
    if (scores) {
      card.append(element('div', 'ball-receipt-final', `${game.away} ${scores.away} — ${game.home} ${scores.home}`));
      card.append(element('p', 'ball-receipt-original', result.outcome === 'PUSH' ? 'Exactly on the spread. Neither a cover nor a loss.' : `${result.outcome === 'COVER' ? 'Covers' : 'Misses'} by ${Math.abs(result.adjusted)} point${Math.abs(result.adjusted) === 1 ? '' : 's'}.`));
    } else card.append(element('p', 'ball-receipt-original', 'The original pick stays here. A result appears only when a final score is available.'));
    const reason = element('div', 'ball-receipt-explanation');
    reason.append(element('b', '', 'THE ORIGINAL REASONING'), document.createTextNode(game.explanation));
    card.append(reason);
    card.append(element('p', 'ball-receipt-provenance', kind === 'demo' && scores ? 'This is a score-scenario preview. It is excluded from all recorded results.' : 'The published pick and reasoning are shown above. Score scenarios never change recorded results.'));
    return card;
  }
  function createDialog(className, title, subtitle) {
    const dialog = element('dialog', `ball-dialog ${className}`);
    const head = element('div', 'ball-dialog-top');
    const copy = element('div');
    const h = element('h2', '', title); h.id = className + '-title';
    copy.append(h, element('p', '', subtitle));
    const close = element('button', 'ball-button', 'Close ×'); close.type = 'button'; close.setAttribute('aria-label', `Close ${title.toLowerCase()}`);
    head.append(copy, close); dialog.append(head); dialog.setAttribute('aria-labelledby', h.id);
    dialog.addEventListener('click', event => { if (event.target === dialog) { const b = dialog.getBoundingClientRect(); if (event.clientX < b.left || event.clientX > b.right || event.clientY < b.top || event.clientY > b.bottom) dialog.close(); } });
    close.addEventListener('click', () => dialog.close());
    document.body.append(dialog);
    return { dialog, head, close };
  }
  const receiptUI = createDialog('ball-receipt-dialog', 'The receipt', 'The original pick, its reasoning, and its result.');
  const receiptBody = element('div', 'ball-receipt-body'); receiptUI.dialog.append(receiptBody);
  function openReceipt(game, demo = false) {
    receiptBody.replaceChildren(receipt(game, demo ? defaultScores(game) : game.final, demo ? 'demo' : 'recorded'));
    const actions = element('div', 'ball-receipt-actions');
    if (!game.final) { const preview = element('button', 'ball-button', demo ? 'Back to published card' : 'Preview with score scenario'); preview.type = 'button'; preview.addEventListener('click', () => openReceipt(game, !demo)); actions.append(preview); }
    receiptBody.append(actions);
    if (!receiptUI.dialog.open) receiptUI.dialog.showModal();
  }

  function historyReceipts() {
    const main = $('main');
    const section = element('section', 'ball-receipts-section'); section.id = 'game-receipts';
    const header = element('div', 'section-head'); header.append(element('h2', '', 'Current card game receipts'), element('span', 'sub', 'Original picks, including the losses.'));
    section.append(header, element('p', 'policy-note', 'Every saved pick has a place here. Final results and illustrative previews stay separate.'));
    const filters = element('div', 'ball-receipt-filters'), grid = element('div', 'ball-receipt-grid'), empty = element('p', 'ball-receipt-empty');
    empty.hidden = true; section.append(filters, grid, empty);
    for (const g of games.values()) { const c = receipt(g, g.final); const reason = $('.ball-receipt-explanation', c); const more = element('details'); more.append(element('summary', '', 'Original reasoning'), reason); c.append(more); grid.append(c); }
    for (const [label, outcome] of [['All picks', 'ALL'], ['Covers', 'COVER'], ['Pushes', 'PUSH'], ['Losses', 'LOSS'], ['Awaiting finals', 'PENDING']]) {
      const b = element('button', 'ball-button', label); b.type = 'button'; b.dataset.filter = outcome; b.setAttribute('aria-pressed', String(outcome === 'ALL'));
      b.addEventListener('click', () => { let shown = 0; $$('.ball-receipt', grid).forEach(c => { c.hidden = outcome !== 'ALL' && c.dataset.outcome !== outcome; if (!c.hidden) shown++; }); $$('button', filters).forEach(n => n.setAttribute('aria-pressed', String(n === b))); empty.hidden = shown > 0; empty.textContent = `No ${outcome.toLowerCase()} results are recorded in this published card.`; }); filters.append(b);
    }
    const before = $('#history-grading-h')?.closest('section'); if (before) before.before(section); else main.append(section);
    const nav = $('.reader-nav'); if (nav) { const link = element('a', '', 'Game receipts'); link.href = '#game-receipts'; nav.prepend(link); }
  }
  window.BallExperience = { grade, receipt, games };
  if (document.body.dataset.interactivePage === 'history') { historyReceipts(); return; }
  const inspector = $('.inspector-col');
  if (!inspector) return;

  // The chart uses the same Gaussian read as the saved spread slider, never the old empirical sweep.
  function modelProbability(widget, offset) {
    const d = widget.dataset, line = Number(d.cardLine) + offset;
    const z = (line - Number(d.center) - Number(d.mean)) / Number(d.std) / Math.SQRT2;
    const t = 1 / (1 + .3275911 * Math.abs(z));
    const erf = Math.sign(z) * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-z * z));
    const home = 1 - .5 * (1 + erf);
    return d.pickIsHome === '1' ? home : 1 - home;
  }
  function repairCurve(panel, game) {
    const widget = $('.ats-adjuster', panel), svg = $('svg.curve', panel);
    if (!widget || !svg) return;
    const slider = $('.adjuster-slider', widget), lo = Number(slider.min), hi = Number(slider.max);
    const values = [];
    for (let offset = lo; offset <= hi + 1e-8; offset += .5) values.push({ offset, probability: modelProbability(widget, offset) });
    const min = Math.max(0, Math.min(...values.map(v => v.probability)) - .035), max = Math.min(1, Math.max(...values.map(v => v.probability)) + .035);
    const x = offset => 20 + (offset - lo) / (hi - lo) * 240, y = p => 85 - (p - min) / (max - min) * 75;
    widget.dataset.yMin = String(min); widget.dataset.yMax = String(max);
    svg.replaceChildren();
    const add = (tag, attrs, text) => { const n = document.createElementNS('http://www.w3.org/2000/svg', tag); Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v)); if (text !== undefined) n.textContent = text; svg.append(n); return n; };
    add('line', { class: 'grid', x1: 20, x2: 260, y1: 85, y2: 85 });
    if (min <= .5 && max >= .5) { add('line', { class: 'ref', x1: 20, x2: 260, y1: y(.5), y2: y(.5) }); add('text', { x: 223, y: y(.5) - 3 }, '50%'); }
    add('path', { class: 'curve-path', d: values.map((v, i) => `${i ? 'L' : 'M'}${x(v.offset).toFixed(2)},${y(v.probability).toFixed(2)}`).join(' ') });
    const zero = modelProbability(widget, 0);
    add('circle', { class: 'marker', cx: x(0), cy: y(zero), r: 3.4 });
    add('text', { x: x(0) + 6, y: y(zero) - 5 }, `${pickText(game)} / ${(zero * 100).toFixed(1)}%`);
    const marker = add('circle', { class: 'adjuster-marker', cx: x(0), cy: y(zero), r: 4.5 });
    add('text', { x: 16, y: 97 }, signed(lo)); add('text', { x: 136, y: 97 }, '0'); add('text', { x: 250, y: 97 }, signed(hi));
    const chartHost = widget.parentElement;
    $$('p', chartHost).filter(p => p.textContent.includes("This chart's own swept line")).forEach(p => p.remove());
    const cap = $('.chart-cap', chartHost); if (cap) cap.textContent = `Model cover chance · ${game.pick}`;
    if (game.adjusted) { const note = element('p', 'ball-model-note', `Situational rules changed this pick. The card’s ${game.score} is a decision score. The original model estimates ${(zero * 100).toFixed(1)}% for ${pickText(game)}; this chart shows that model estimate.`); widget.before(note); }
    else if (Math.abs(zero * 100 - parseFloat(game.score)) > .06) { svg.hidden = true; widget.hidden = true; chartHost.append(element('p', 'ball-invalid-curve', 'This saved curve does not match the card. It needs to be regenerated before it can be shown.')); }
    const update = () => { const offset = Number(slider.value), probability = modelProbability(widget, offset); marker.setAttribute('cx', x(offset).toFixed(2)); marker.setAttribute('cy', y(probability).toFixed(2)); marker.style.opacity = '1'; $('.adjuster-pct', widget).textContent = `${(probability * 100).toFixed(1)}%`; };
    slider.addEventListener('input', update); update();
    svg.dataset.probabilityAtQuote = zero.toFixed(8);
  }

  const roomUI = createDialog('ball-room-dialog', 'Game room', 'Arrow keys: games · Escape: back to the board');
  const roomBody = element('div', 'ball-room-content'); roomUI.dialog.append(roomBody);
  const roomButtons = element('div', 'room-buttons'); roomUI.close.before(roomButtons); roomButtons.append(roomUI.close);
  for (const [text, dir] of [['← Previous', -1], ['Next →', 1]]) { const b = element('button', 'ball-button', text); b.type = 'button'; b.addEventListener('click', () => moveGame(dir)); roomButtons.insertBefore(b, roomUI.close); }
  let placeholder, returnFocus, oldX, oldY, oldOverflow;
  function selectedPanel() { return $('.dive-panel:not([hidden])', inspector); }
  function selectedGame() { return games.get(selectedPanel().id); }
  function syncRoom() { const panel = selectedPanel(); const isField = panel.dataset.ballTab === 'field' || !panel.dataset.ballTab; roomUI.dialog.classList.toggle('room-overview', isField); $('.ball-dialog-top h2', roomUI.dialog).textContent = `${selectedGame().away} at ${selectedGame().home} / Game room`; }
  function openRoom() {
    if (roomUI.dialog.open) return;
    returnFocus = document.activeElement; oldX = scrollX; oldY = scrollY; oldOverflow = document.body.style.overflow;
    placeholder = element('div', 'ball-room-placeholder', 'This matchup is open in the game room.'); placeholder.style.height = `${inspector.getBoundingClientRect().height}px`;
    inspector.before(placeholder); roomBody.append(inspector); document.body.style.overflow = 'hidden'; syncRoom(); roomUI.dialog.showModal(); roomUI.close.focus();
  }
  roomUI.dialog.addEventListener('close', () => { if (!placeholder) return; placeholder.replaceWith(inspector); placeholder = null; document.body.style.overflow = oldOverflow; returnFocus?.focus({ preventScroll: true }); window.scrollTo(oldX, oldY); });
  function tab(panel, key, focus = false) {
    const nodes = { field: $('.merged-field', panel), analysis: $('.dive-body', panel), lineups: $('.lineups-block', panel), score: $('.ball-score-lab', panel) };
    Object.entries(nodes).forEach(([k, n]) => n?.classList.toggle('merged-view-hidden', k !== key));
    $$('[data-ball-tab]', panel).forEach(b => { const on = b.dataset.ballTab === key; b.setAttribute('aria-selected', String(on)); b.tabIndex = on ? 0 : -1; if (on && focus) b.focus(); });
    panel.dataset.ballTab = key; if (roomUI.dialog.open) syncRoom();
  }
  function scoreLab(panel, game) {
    const lab = element('div', 'ball-score-lab merged-view-hidden'); lab.id = `${game.id}-score-lab`; lab.setAttribute('role', 'tabpanel');
    lab.append(element('p', 'lab-caption', 'HYPOTHETICAL FINAL / THE PUBLISHED PICK STAYS FIXED'), element('h3', '', 'How does this pick win?'), element('p', 'lab-intro', `Change the final score and see how ${pickText(game)} settles. This is a scenario, not a prediction.`));
    const board = element('div', 'ball-scoreboard');
    const inputs = {};
    for (const side of ['away', 'home']) { if (side === 'home') board.append(element('div', 'ball-score-dash', '—')); const team = element('div', 'ball-score-team'), label = element('label', '', game[side]); const input = element('input'); input.type = 'number'; input.inputMode = 'numeric'; input.min = '0'; input.max = '99'; input.step = '1'; input.value = String(defaultScores(game)[side]); input.id = `${game.id}-${side}-scenario-score`; label.htmlFor = input.id; const numbers = element('div', 'ball-score-number'); for (const [text, step] of [['−', -1], ['+', 1]]) { const b = element('button', '', text); b.type = 'button'; b.setAttribute('aria-label', `${step > 0 ? 'Increase' : 'Decrease'} ${game[side]} score`); b.addEventListener('click', () => { input.value = String(Math.max(0, Math.min(99, (Number.isFinite(input.valueAsNumber) ? Math.trunc(input.valueAsNumber) : 0) + step))); update(); }); if (step === -1) numbers.append(b, input); else numbers.append(b); } inputs[side] = input; team.append(label, numbers); board.append(team); }
    lab.append(board); const result = element('div', 'ball-score-result'); result.setAttribute('aria-live', 'polite'); result.append(element('strong'), element('p')); lab.append(result);
    const chart = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); chart.classList.add('ball-margin-chart'); chart.setAttribute('viewBox', '0 0 540 90'); chart.setAttribute('role', 'img'); lab.append(chart);
    const footer = element('div', 'ball-scenario-footer'); footer.append(element('p', '', 'Only these hypothetical scores change. The published card and recorded results do not.')); const preview = element('button', 'ball-button', 'Preview the receipt ↗'); preview.type = 'button'; preview.addEventListener('click', () => openReceipt(game, true)); footer.append(preview); lab.append(footer);
    function update() {
      const scores = { away: inputs.away.valueAsNumber, home: inputs.home.valueAsNumber }, valid = scoresValid(scores);
      Object.values(inputs).forEach(i => i.setAttribute('aria-invalid', String(!/^\d{1,2}$/.test(i.value) || !Number.isInteger(i.valueAsNumber) || i.valueAsNumber < 0 || i.valueAsNumber > 99)));
      const r = grade(game, scores); result.dataset.outcome = r.outcome; preview.disabled = !valid;
      if (!valid) { $('strong', result).textContent = 'CHECK THE SCORES'; $('p', result).textContent = 'Use whole-number scores from 0 to 99 for both teams.'; chart.replaceChildren(); return; }
      scenarios.set(game.id, scores); $('strong', result).textContent = r.outcome;
      const football = r.margin > 0 ? `${game.pick} wins by ${r.margin}` : r.margin < 0 ? `${game.pick} loses by ${-r.margin}` : 'The game finishes tied';
      $('p', result).textContent = `${football}. ${pickText(game)} ${r.outcome === 'PUSH' ? 'lands exactly on the spread.' : `${r.outcome === 'COVER' ? 'covers' : 'misses'} by ${Math.abs(r.adjusted)} point${Math.abs(r.adjusted) === 1 ? '' : 's'}.`}`;
      const range = Math.max(14, Math.ceil((Math.abs(game.spread) + 1) / 7) * 7, Math.ceil((Math.abs(r.margin) + 1) / 7) * 7), x = v => 30 + (v + range) / (2 * range) * 480, boundary = x(-game.spread);
      chart.replaceChildren(); const add = (tag, attrs, text) => { const n = document.createElementNS('http://www.w3.org/2000/svg', tag); Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v)); if (text !== undefined) n.textContent = text; chart.append(n); };
      add('rect', { x: 30, y: 25, width: boundary - 30, height: 13, fill: '#745046', rx: 2 }); add('rect', { x: boundary, y: 25, width: 510 - boundary, height: 13, fill: '#4d8b65', rx: 2 }); add('line', { x1: boundary, x2: boundary, y1: 18, y2: 48, stroke: '#efd49a', 'stroke-width': 2 }); add('circle', { class: 'ball-score-marker', cx: x(r.margin), cy: 31, r: 7, fill: '#f5e4ba', stroke: '#17231a', 'stroke-width': 2 }); add('text', { x: 30, y: 65, fill: '#c4d0ba', 'font-size': 10 }, `Lose by ${range}`); add('text', { x: 270, y: 65, fill: '#c4d0ba', 'font-size': 10, 'text-anchor': 'middle' }, 'Tie game'); add('text', { x: 510, y: 65, fill: '#c4d0ba', 'font-size': 10, 'text-anchor': 'end' }, `Win by ${range}`); add('text', { x: boundary, y: 11, fill: '#ead096', 'font-size': 10, 'text-anchor': 'middle' }, 'Spread boundary'); chart.setAttribute('aria-label', `${football}; ${r.outcome.toLowerCase()} at ${pickText(game)}.`);
    }
    Object.values(inputs).forEach(input => input.addEventListener('input', update)); update(); return lab;
  }

  $$('.dive-panel').forEach(panel => {
    const game = games.get(panel.id); if (!game) return;
    repairCurve(panel, game);
    const coverLabel = $('.cover-read small', panel); if (coverLabel) coverLabel.textContent = 'decision score';
    $$('.game-sub', $('.dive-head', panel)).forEach(n => { [...n.childNodes].filter(c => c.nodeType === 3).forEach(t => { t.textContent = t.textContent.replace('cover prob', 'decision score'); }); });
    const actions = element('div', 'ball-actions');
    const expand = element('button', 'ball-button ball-expand primary', '↗ Game room'); expand.type = 'button'; expand.addEventListener('click', openRoom);
    const receiptButton = element('button', 'ball-button ball-receipt-open', 'Receipt'); receiptButton.type = 'button'; receiptButton.addEventListener('click', () => openReceipt(game)); actions.append(expand, receiptButton); $('.dive-head', panel).after(actions);
    const lab = scoreLab(panel, game), tabs = $('.merged-tabs', panel); tabs.after(lab);
    const scenarioButton = element('button', '', 'Score scenario'); scenarioButton.type = 'button'; scenarioButton.setAttribute('role', 'tab'); scenarioButton.id = `${game.id}-score-tab`; scenarioButton.setAttribute('aria-controls', lab.id); lab.setAttribute('aria-labelledby', scenarioButton.id); tabs.append(scenarioButton);
    const labels = [['field', 'Field'], ['analysis', 'Analysis'], ['lineups', 'Lineups'], ['score', 'Score scenario']];
    $$('button', tabs).forEach((b, i) => { const [key, label] = labels[i]; b.textContent = label; b.dataset.ballTab = key; b.addEventListener('click', e => { e.preventDefault(); e.stopImmediatePropagation(); tab(panel, key); }, true); b.addEventListener('keydown', e => { if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return; e.preventDefault(); e.stopImmediatePropagation(); const next = e.key === 'Home' ? 0 : e.key === 'End' ? 3 : (i + (e.key === 'ArrowRight' ? 1 : -1) + 4) % 4; tab(panel, labels[next][0], true); }, true); });
    $('.merged-scenario-link button', panel).addEventListener('click', e => { e.stopImmediatePropagation(); tab(panel, 'analysis'); $('.adjuster-slider', panel)?.focus(); }, true);
    tab(panel, 'field');
  });
  const probabilityHeading = $('table.board th:nth-child(4)'); if (probabilityHeading) { probabilityHeading.textContent = 'Score'; probabilityHeading.title = 'On adjusted picks this is a mirrored decision score, not a newly calibrated probability. The analysis chart labels the original model estimate separately.'; }
  const keyboardHint = element('p', 'ball-keyboard-hint'); keyboardHint.innerHTML = '<kbd>↑</kbd> <kbd>↓</kbd> switch games &nbsp; <kbd>Enter</kbd> game room &nbsp; <kbd>Esc</kbd> return'; $('.board-col .sort-toggle').after(keyboardHint);
  function moveGame(direction) { const order = $$('table.board tr.game').map(r => r.dataset.gameId), index = order.indexOf(selectedPanel().id); window.atsSelectGame(order[(index + direction + order.length) % order.length]); }
  document.addEventListener('keydown', event => {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
    if (receiptUI.dialog.open) return;
    const editable = event.target.closest('input,textarea,select,[contenteditable="true"],[role="tablist"],.stadium-stage'); if (editable) return;
    if (event.key === 'Escape' && roomUI.dialog.open) { event.preventDefault(); roomUI.dialog.close(); return; }
    if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Enter'].includes(event.key)) return;
    if (event.key === 'Enter') { const row = event.target.closest('table.board tr.game'); if (!roomUI.dialog.open && (row || !event.target.closest('button,a,summary'))) { event.preventDefault(); if (row) window.atsSelectGame(row.dataset.gameId); openRoom(); } return; }
    if (event.target.closest('button,a,summary') && !event.target.closest('table.board') && !roomUI.dialog.open) return;
    event.preventDefault(); moveGame(event.key === 'ArrowUp' || event.key === 'ArrowLeft' ? -1 : 1);
  });
  inspector.addEventListener('ball:gamechange', () => { if (roomUI.dialog.open) syncRoom(); });
  Object.assign(window.BallExperience, { openRoom, tab, openReceipt, modelProbability });
})();
