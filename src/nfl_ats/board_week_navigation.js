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
  const panels = new Map(Array.from(document.getElementById('week-archive').content.children).map(panel => [panel.dataset.weekPanel, panel]));
  const options = new Set(Array.from(select.options).map(option => option.value));
  const originalTitle = document.title;
  const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
  const weekLabel = key => {
    const [season, week] = key.split('-');
    return `${season} · Week ${week}`;
  };
  const selectArchivedGame = (panel, week, index, focus = false) => {
    const game = week.games[index];
    if (!game) return;
    panel.dataset.selectedIndex = String(index);
    panel.querySelectorAll('[data-archive-index]').forEach(row => {
      const selected = Number(row.dataset.archiveIndex) === index;
      row.classList.toggle('is-selected', selected);
      if (selected) row.setAttribute('aria-current', 'true');
      else row.removeAttribute('aria-current');
      if (selected && focus) row.focus();
    });
    panel.querySelector('.week-archive-detail').innerHTML = `
      <div class="dive"><div class="dive-head"><div class="refined-matchup">
        <div><div class="match-label">THE MATCHUP</div><div class="teams"><span class="away">${escapeHTML(game.awayTeam)}</span><i>at</i><span class="home">${escapeHTML(game.homeTeam)}</span></div></div>
        <div class="cover-read"><strong>${escapeHTML(game.confidence)}</strong><small>published cover chance</small></div>
      </div><div class="original-pick"><div><div class="game-id">${game.bestPick ? '★ ' : ''}${escapeHTML(game.pickTeam)} ${escapeHTML(game.pickLine)}</div><div class="game-sub">Week ${week.week} · ${game.bestPick ? 'Best Pick of the week' : 'Published pick'}</div></div></div></div>
      <div class="week-saved-result"><span class="match-label">${game.status === 'Pending' ? 'RESULT' : 'FINAL'}</span><strong>${escapeHTML(game.score || 'Awaiting the final score')}</strong><span>${escapeHTML(game.status)}</span></div>
      <p class="week-saved-note">This is the pick and pool line saved with this week’s card.</p>
      <div class="ball-actions"><button type="button" class="ball-button" data-week-step="-1">← Previous game</button><button type="button" class="ball-button" data-week-step="1">Next game →</button></div></div>`;
  };
  panels.forEach((panel, key) => {
    const week = data.weeks.find(item => item.key === key);
    panel.addEventListener('click', event => {
      const row = event.target.closest('[data-archive-index]');
      if (row) selectArchivedGame(panel, week, Number(row.dataset.archiveIndex));
      const step = event.target.closest('[data-week-step]');
      if (step) {
        const direction = Number(step.dataset.weekStep);
        const index = (Number(panel.dataset.selectedIndex) + direction + week.games.length) % week.games.length;
        selectArchivedGame(panel, week, index);
        panel.querySelector(`[data-week-step="${direction}"]`).focus();
      }
    });
    panel.addEventListener('keydown', event => {
      const row = event.target.closest('[data-archive-index]');
      if (!row || !['ArrowDown', 'ArrowUp', 'Enter', ' '].includes(event.key)) return;
      event.preventDefault();
      const direction = event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0;
      const index = (Number(row.dataset.archiveIndex) + direction + week.games.length) % week.games.length;
      selectArchivedGame(panel, week, index, true);
    });
    selectArchivedGame(panel, week, Math.max(0, week.games.findIndex(game => game.bestPick)));
  });
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
    document.querySelectorAll('dialog[open]').forEach(dialog => dialog.close());
    document.body.dataset.weekLiveVisible = String(hasLiveCard);
    select.value = key;
    current.disabled = isCurrent;
    slot.replaceChildren(hasLiveCard ? live : panels.get(key) || pending);
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
