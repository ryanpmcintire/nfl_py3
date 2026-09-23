(function () {
  'use strict';
  const dataNode = document.getElementById('week-navigation-data');
  const browser = document.getElementById('week-browser');
  const main = document.getElementById('main-content');
  if (!dataNode || !browser || !main) return;
  const data = JSON.parse(dataNode.textContent);
  const select = document.getElementById('week-select');
  const current = document.getElementById('week-current');
  const readiness = document.getElementById('week-readiness');
  const status = document.getElementById('week-selection-status');
  const live = main.querySelector('.week-grid');
  if (!live) return;
  const slot = document.createElement('div');
  slot.id = 'week-card';
  live.before(slot);
  slot.append(live);
  const record = main.querySelector('.season-record-strip');
  (record || slot).before(browser);
  const weeklyNotes = Array.from(main.children).filter(node =>
    node.matches('.pool-line-note, .week-line, .assistant') ||
    ['What changed this week', 'Rival rules', 'Findings desk'].includes(node.querySelector('h2')?.textContent.trim())
  );
  const weekChip = record?.querySelector('.record-chip');
  const liveRecord = weekChip?.textContent;
  const archive = document.getElementById('week-archive');
  const panels = new Map();
  Array.from(archive?.children || []).forEach(panel => {
    if (panel.dataset.weekPanel) panels.set(panel.dataset.weekPanel, panel);
    panel.remove();
  });
  archive?.remove();
  const options = new Set(Array.from(select.options).map(option => option.value));
  const originalTitle = document.title;
  const weekLabel = key => {
    const [season, week] = key.split('-');
    return `${season} · Week ${week}`;
  };
  const selections = new Map();
  const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
  const selectedGameId = panel => panel?.querySelector('table.board tr.game.is-selected[data-game-id]')?.dataset.gameId || panel?.querySelector('.dive-panel:not([hidden])')?.dataset.gameId;
  const chooseGame = (key, panel) => {
    if (!panel) return null;
    const ids = new Set(Array.from(panel.querySelectorAll('table.board tr.game[data-game-id]'), row => row.dataset.gameId));
    const remembered = selections.get(key);
    if (ids.has(remembered)) return remembered;
    const selected = selectedGameId(panel);
    if (ids.has(selected)) return selected;
    return panel.querySelector('table.board tr.game.is-best[data-game-id]')?.dataset.gameId || panel.querySelector('table.board tr.game[data-game-id]')?.dataset.gameId;
  };
  const pending = document.createElement('div');
  pending.className = 'week-grid week-pending-card';
  pending.innerHTML = `<section class="board-col"><div class="section-head"><h2>Week ${data.current.week} / The complete card</h2><span class="sub">Current week</span></div><div class="week-pending-message"><span class="match-label">WEEK ${data.current.week}</span><h3>Picks are on the way</h3><p>${escapeHTML(data.readiness.message)}</p>${data.published ? '<button type="button" class="ball-button" data-latest-card>View the latest published card</button>' : ''}</div></section>`;
  pending.querySelector('[data-latest-card]')?.addEventListener('click', () => show(data.published.key, true));
  const requested = () => {
    const url = new URL(window.location.href);
    const key = `${url.searchParams.get('season')}-${url.searchParams.get('week')}`;
    return options.has(key) ? key : data.current.key;
  };
  const show = (requestedKey, updateURL = false) => {
    const key = options.has(requestedKey) ? requestedKey : data.current.key;
    const isCurrent = key === data.current.key;
    const hasLiveCard = data.published?.key === key;
    const outgoing = slot.querySelector('.week-grid[data-week-panel]');
    const outgoingId = selectedGameId(outgoing);
    if (outgoing?.dataset.weekPanel && outgoingId) selections.set(outgoing.dataset.weekPanel, outgoingId);
    window.BallExperience?.closeRoom();
    document.querySelectorAll('dialog[open]').forEach(dialog => dialog.close());
    document.body.dataset.weekLiveVisible = String(hasLiveCard);
    select.value = key;
    current.disabled = isCurrent;
    const panel = hasLiveCard ? live : panels.get(key);
    slot.replaceChildren(panel || pending);
    const gameId = chooseGame(key, panel);
    if (gameId && window.atsSelectGame) window.atsSelectGame(gameId);
    weeklyNotes.forEach(node => { node.hidden = !hasLiveCard; });
    readiness.hidden = true;
    const week = data.weeks.find(item => item.key === key);
    if (weekChip) {
      if (hasLiveCard) weekChip.textContent = liveRecord.replace('This week:', `Week ${key.split('-')[1]}:`);
      else if (week) {
        const wins = week.games.filter(game => game.status === 'Won').length;
        const losses = week.games.filter(game => game.status === 'Lost').length;
        const pushes = week.games.filter(game => game.status === 'Push').length;
        const settled = wins + losses + pushes;
        weekChip.textContent = `Week ${week.week}: ${settled ? `${wins}-${losses}${pushes ? `-${pushes}` : ''}` : 'In progress'}`;
      } else weekChip.textContent = `Week ${data.current.week}: Picks pending`;
    }
    status.textContent = `${weekLabel(key)} · ${isCurrent ? 'Current week' : 'Published picks'}`;
    const kicker = main.querySelector('.merged-kicker');
    if (kicker) kicker.textContent = `WEEK ${key.split('-')[1]} / THE CARD`;
    const headerWeek = document.querySelector('.fixed-brand-label');
    if (headerWeek) headerWeek.textContent = `NFL / WEEK ${key.split('-')[1]}`;
    document.title = `${weekLabel(key)} · ${originalTitle}`;
    if (updateURL) {
      const url = new URL(window.location.href);
      url.searchParams.delete('season');
      url.searchParams.delete('week');
      url.hash = '';
      if (!isCurrent) {
        const [season, weekNumber] = key.split('-');
        url.searchParams.set('season', season);
        url.searchParams.set('week', weekNumber);
      }
      window.history.pushState(null, '', url);
    }
  };
  select.addEventListener('change', () => show(select.value, true));
  current.addEventListener('click', () => show(data.current.key, true));
  window.addEventListener('popstate', () => show(requested()));
  show(requested());
})();
