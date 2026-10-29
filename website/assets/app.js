(() => {
  'use strict';
  const data = window.RIDEWAY_DATA;
  const body = document.querySelector('#leaderboard-body');
  const search = document.querySelector('#model-search');
  const sort = document.querySelector('#ranking-sort');
  const get = (row, path) => path.split('.').reduce((value, key) => value?.[key], row);
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let sortKey = 'eu';
  let ascending = false;
  const maxima = {};
  for (const scope of ['full58', 'custom14']) {
    for (const metric of ['avg', 'pass_at_3', 'pass_3']) {
      const key = `${scope}.${metric}`;
      maxima[key] = Math.max(...data.models.map(r => get(r, key)));
    }
  }
  const ranked = (a, b) => {
    const av = get(a, sortKey), bv = get(b, sortKey);
    const result = sortKey === 'model' ? av.localeCompare(bv) : av - bv;
    return (ascending ? result : -result) || a.rank - b.rank;
  };
  function render() {
    const query = search.value.trim().toLocaleLowerCase();
    const rows = data.models.filter(r => `${r.model} ${r.provider} ${r.key}`.toLocaleLowerCase().includes(query)).sort(ranked);
    const positions = new Map([...data.models].sort(ranked).map((row, i) => [row.key, i + 1]));
    body.innerHTML = rows.map(row => {
      const position = positions.get(row.key);
      const rank = position <= 3 ? `<span class="top-rank">${position}</span>` : position;
      const cells = ['full58'].flatMap(scope => ['avg', 'pass_at_3', 'pass_3'].map(metric => {
        const key = `${scope}.${metric}`, val = get(row, key);
        return `<td class="${val === maxima[key] ? 'best' : ''}">${val.toFixed(1)}</td>`;
      })).join('');
      return `<tr><td class="rank-col">${rank}</td><th scope="row" class="model-col">${esc(row.model)}<small>${esc(row.provider)}</small></th>${cells}<td class="eu-cell">${row.eu.toFixed(3)}</td><td class="ci-cell">[${row.ci90.map(v => v.toFixed(3)).join(', ')}]</td></tr>`;
    }).join('');
    document.querySelector('#model-count').textContent = `${rows.length} ${rows.length === 1 ? 'model' : 'models'}`;
    document.querySelector('#empty-state').hidden = rows.length > 0;
    document.querySelectorAll('[data-sort]').forEach(button => {
      const th = button.closest('th');
      th.removeAttribute('aria-sort');
      button.querySelector('.sort-arrow')?.remove();
      if (button.dataset.sort === sortKey) {
        th.setAttribute('aria-sort', ascending ? 'ascending' : 'descending');
        const arrow = document.createElement('span');
        arrow.className = 'sort-arrow'; arrow.textContent = ascending ? ' ↑' : ' ↓';
        button.appendChild(arrow);
      }
    });
  }
  search.addEventListener('input', render);
  sort.addEventListener('change', () => {sortKey = sort.value; ascending = sortKey === 'model'; render();});
  document.querySelectorAll('[data-sort]').forEach(button => button.addEventListener('click', () => {
    const next = button.dataset.sort;
    ascending = sortKey === next ? !ascending : false;
    sortKey = next;
    if (![...sort.options].some(o => o.value === next)) {
      const option = document.createElement('option');
      option.value = next; option.textContent = button.textContent.replace(/[↑↓]/g, '').trim() + ' · held-out';
      sort.appendChild(option);
    }
    sort.value = next;
    render();
  }));
  document.querySelector('#copy-citation').addEventListener('click', async event => {
    const text = document.querySelector('#bibtex').textContent;
    const status = document.querySelector('#copy-status');
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const field = document.createElement('textarea');
        field.value = text; field.style.cssText = 'position:fixed;top:-9999px';
        document.body.appendChild(field); field.select();
        const copied = document.execCommand('copy'); field.remove();
        if (!copied) throw new Error('Clipboard unavailable');
      }
      event.target.textContent = 'Copied ✓';
      status.textContent = 'Citation copied to clipboard.';
      setTimeout(() => {event.target.textContent = 'Copy BibTeX'; status.textContent = '';}, 2500);
    } catch {
      const range = document.createRange(); range.selectNodeContents(document.querySelector('#bibtex'));
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
    }
  });
  if ('IntersectionObserver' in window) {
    const sections = document.querySelectorAll('section[id]');
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) {
        document.querySelectorAll('.navlinks a').forEach(a => a.classList.toggle('active', a.hash === `#${entry.target.id}`));
      }
    }), {rootMargin: '-12% 0px -68% 0px'});
    sections.forEach(section => observer.observe(section));
  }
  render();
})();
