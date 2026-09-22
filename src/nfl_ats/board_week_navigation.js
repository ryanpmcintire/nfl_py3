(() => {
  const dataNode = document.getElementById('week-navigation-data');
  if (!dataNode) return;
  const data = JSON.parse(dataNode.textContent);
  const browser = document.getElementById('week-browser');
  const select = document.getElementById('week-select');
  const current = document.getElementById('week-current');
  const readiness = document.getElementById('week-readiness');
  const status = document.getElementById('week-selection-status');
  const archive = document.getElementById('week-archive');
  const main = document.getElementById('main-content');
  let live = document.getElementById('week-live');
  if (!live) {
    live = document.createElement('div');
    live.id = 'week-live';
    live.dataset.weekKey = data.published?.key || '';
    [...main.children].filter(node => node !== browser && node !== archive && node !== dataNode).forEach(node => live.append(node));
    browser.after(live);
  }
  const options = new Set([...select.options].map(option => option.value));
  const title = document.title;
  const weekLabel = key => {
    const [season, week] = key.split('-');
    return `${season} · Week ${week}`;
  };
  const requested = () => {
    const query = new URL(window.location.href).searchParams;
    const key = `${query.get('season')}-${query.get('week')}`;
    return options.has(key) ? key : data.current.key;
  };
  const show = (key, updateUrl = false) => {
    if (!options.has(key)) key = data.current.key;
    const isCurrent = key === data.current.key;
    const hasLiveCard = isCurrent && data.published?.key === key;
    document.body.dataset.weekLiveVisible = String(hasLiveCard);
    select.value = key;
    current.disabled = isCurrent;
    live.hidden = !hasLiveCard;
    archive.hidden = isCurrent;
    archive.querySelectorAll('[data-week-panel]').forEach(panel => {
      panel.hidden = panel.dataset.weekPanel !== key;
    });
    readiness.hidden = !isCurrent || hasLiveCard;
    const selectedWeek = data.weeks.find(week => week.key === key);
    status.textContent = isCurrent
      ? `${weekLabel(key)} · Current week${hasLiveCard ? '' : ' · Picks not available yet'}`
      : `${weekLabel(key)} · Published picks · ${selectedWeek?.games.length || 0} games`;
    const tag = document.querySelector('.fixed-brand-label');
    if (tag) tag.textContent = `NFL / WEEK ${key.split('-')[1]}`;
    document.title = `${weekLabel(key)} · ${isCurrent ? title : 'Published picks'}`;
    if (updateUrl) {
      const url = new URL(window.location.href);
      url.searchParams.delete('season');
      url.searchParams.delete('week');
      if (!isCurrent) {
        const [season, week] = key.split('-');
        url.searchParams.set('season', season);
        url.searchParams.set('week', week);
      }
      window.history.pushState(null, '', url);
    }
  };
  select.addEventListener('change', () => show(select.value, true));
  current.addEventListener('click', () => show(data.current.key, true));
  window.addEventListener('popstate', () => show(requested()));
  show(requested());
})();
