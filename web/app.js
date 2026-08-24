'use strict';

const $ = (s, r = document) => r.querySelector(s);
const api = async (u, o) => {
  const r = await fetch(u, o);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
/* Most phones write the maker into BOTH Make and Model ("OnePlus" / "OnePlus 13"),
   so naive concatenation reads "OnePlus OnePlus 13". */
const camera = (p) => {
  const make = (p.camera_make || '').trim(), model = (p.camera_model || '').trim();
  if (!model) return make;
  return model.toLowerCase().startsWith(make.toLowerCase()) ? model : `${make} ${model}`.trim();
};
const when = (t) => t ? new Date(t * 1000).toLocaleDateString(undefined,
  { year: 'numeric', month: 'short', day: 'numeric' }) : '';

const state = {
  results: [], q: '', offset: 0, loading: false, done: false, cursor: -1,
  group: localStorage.getItem('pa.group') || 'month',
  sort: localStorage.getItem('pa.sort') || 'newest',
};

/* Search runs three engines and fuses them. Naming them "fts"/"vec"/"filter" is
   how the code thinks; these are how a person thinks about the same thing. */
const MATCH = {
  fts:    { cls: 'lex', label: 'contain your words',
            tip: 'Your words appear in the caption, tags, or text in the photo' },
  vec:    { cls: 'sem', label: 'look like it',
            tip: 'The photo looks like what you described, even without those words' },
  filter: { cls: 'fil', label: 'match your filters',
            tip: 'Matched a person, date, tag or camera you asked for' },
};

function matchSummary(results) {
  const counts = {};
  for (const r of results)
    for (const s of (r.sources || [])) counts[s] = (counts[s] || 0) + 1;
  // Counts come from the loaded page, so say "of these" rather than implying
  // they describe every match.
  const parts = Object.entries(counts)
    .filter(([k]) => MATCH[k])
    .sort((a, b) => b[1] - a[1])
    .map(([k, n]) => `<span class="why" title="${esc(MATCH[k].tip)}">` +
                     `<i class="dot ${MATCH[k].cls}"></i>${n} ${MATCH[k].label}</span>`);
  return parts.length ? ` <span class="whys">${parts.join('')}</span>` : '';
}

/* ------------------------------------------------------------------ views */
function show(view) {
  for (const id of ['grid', 'people', 'places', 'dupes', 'library', 'settings'])
    $(`#${id}-view`).hidden = id !== view;
  document.querySelectorAll('.tabs button').forEach(b =>
    b.classList.toggle('on', b.dataset.view === view));
  if (view === 'people') loadPeople();
  if (view === 'library') loadLibrary();
  if (view === 'places') loadPlaces();
  if (view === 'dupes') loadDupes();
  if (view === 'settings') loadSettings();
  // Indexing progress is only drawn on Library, so stop asking for it elsewhere.
  if (view !== 'library') stopRunPoll();
}
document.querySelectorAll('.tabs button').forEach(b =>
  b.addEventListener('click', () => show(b.dataset.view)));

/* ------------------------------------------------------------ organisation */
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'];

/* Returns [sortKey, heading, subheading]. sortKey keeps groups in the same order
   the server returned rows in, so switching sort reorders groups too. */
function groupOf(p) {
  if (state.group === 'folder') {
    const dir = (p.rel_path || '').replace(/\\/g, '/').split('/').slice(0, -1).join('/');
    return [dir || '\u0000', dir || 'Top level', ''];
  }
  if (!p.taken_at) return ['0000', 'Date unknown', ''];
  const d = new Date(p.taken_at * 1000);
  const y = d.getFullYear(), m = d.getMonth(), day = d.getDate();
  const pad = (n) => String(n).padStart(2, '0');
  if (state.group === 'year') return [`${y}`, `${y}`, ''];
  if (state.group === 'day') {
    const today = new Date(), yest = new Date(Date.now() - 864e5);
    const same = (a, b) => a.toDateString() === b.toDateString();
    const name = same(d, today) ? 'Today' : same(d, yest) ? 'Yesterday'
      : `${day} ${MONTHS[m]} ${y}`;
    return [`${y}${pad(m + 1)}${pad(day)}`, name, ''];
  }
  return [`${y}${pad(m + 1)}`, `${MONTHS[m]} ${y}`, ''];
}

/* Collects photos into groups by key rather than assuming the list already
   arrives in group order. Search results are ranked by relevance, so their dates
   are scattered -- walking the list and starting a new section whenever the key
   changes produced one photo per group and the same month heading several times.

   Group order follows the first appearance of each key, which keeps the most
   relevant group first when searching and stays chronological when browsing. */
function render() {
  const host = $('#groups');
  host.innerHTML = '';
  if (state.group === 'none') {
    const sheet = document.createElement('div');
    sheet.className = 'sheet';
    state.results.forEach((p, i) => sheet.appendChild(cell(p, i)));
    host.appendChild(sheet);
    return;
  }

  const groups = new Map();
  state.results.forEach((p, i) => {
    const [key, heading, sub] = groupOf(p);
    if (!groups.has(key)) groups.set(key, { heading, sub, items: [] });
    groups.get(key).items.push(i);
  });

  // Browsing is already in the server's order, so sort chronologically to match
  // the chosen direction. Searching keeps relevance order (best group first).
  const entries = [...groups.entries()];
  if (!state.q && state.group !== 'folder') {
    const dir = state.sort === 'oldest' ? 1 : -1;
    entries.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0) * dir);
  }

  const frag = document.createDocumentFragment();
  for (const [, g] of entries) {
    const wrap = document.createElement('section');
    wrap.className = 'group';
    const n = g.items.length;
    wrap.innerHTML = `<div class="grouphd">
        <h3 class="${state.group === 'folder' ? 'path' : ''}">${esc(g.heading)}</h3>
        <span>${n} photo${n === 1 ? '' : 's'}</span>${g.sub ? `<span>${esc(g.sub)}</span>` : ''}
      </div><div class="sheet"></div>`;
    const sheet = wrap.querySelector('.sheet');
    g.items.forEach(i => sheet.appendChild(cell(state.results[i], i)));
    frag.appendChild(wrap);
  }
  host.appendChild(frag);
}

/* ------------------------------------------------------------------- grid */
function cell(p, i) {
  const el = document.createElement('button');
  el.className = 'cell';
  el.dataset.i = i;
  el.setAttribute('aria-label', p.caption || p.filename || 'photo');
  const found = (p.sources || []).filter(s => MATCH[s]);
  const prov = found.map(s => `<i class="${MATCH[s].cls}"></i>`).join('');
  const why = found.map(s => MATCH[s].tip).join('. ');
  el.innerHTML =
    (prov ? `<div class="prov" title="${esc(why)}">${prov}</div>` : '') +
    `<img loading="lazy" src="/api/thumb/${p.blake3}/grid" alt="">` +
    `<div class="meta">${esc((p.caption || p.filename || '').slice(0, 90))}</div>`;
  el.querySelector('img').addEventListener('error', e => {
    e.target.remove();
    el.classList.add('nothumb');
    el.insertAdjacentHTML('beforeend', `<span>${esc(p.filename || 'no thumbnail')}</span>`);
  });
  el.addEventListener('click', () => openLightbox(i));
  return el;
}

async function load(reset) {
  if (state.loading || (state.done && !reset)) return;
  state.loading = true;
  if (reset) { state.offset = 0; state.done = false; state.results = []; $('#groups').innerHTML = ''; }
  $('#end').textContent = 'loading…';
  try {
    const d = await api(`/api/search?q=${encodeURIComponent(state.q)}` +
      `&limit=80&offset=${state.offset}&sort=${encodeURIComponent(state.sort)}`);
    state.results.push(...d.results);
    render();
    state.offset += d.results.length;
    state.done = d.results.length < 80;

    const n = d.total ?? state.results.length;
    const count = d.capped ? `${n}+` : `${n}`;
    $('#resultline').innerHTML = state.q
      ? `${count} result${n === 1 ? '' : 's'} for "${esc(state.q)}"${matchSummary(state.results)}`
      : `${state.results.length} photos, newest first`;
    $('#end').textContent = state.results.length === 0
      ? (state.q ? 'Nothing matched. Try fewer words, or a person or tag.'
                 : 'No photos indexed yet. Run: pa root add <folder>')
      : (state.done ? '— end of results —' : '');
  } catch (e) {
    $('#end').textContent = `Search failed: ${e.message}`;
  } finally { state.loading = false; }
}

let timer;
$('#q').addEventListener('input', e => {
  clearTimeout(timer);
  const v = e.target.value;
  timer = setTimeout(() => { state.q = v; load(true); }, 220);
});
$('#searchform').addEventListener('submit', e => {
  e.preventDefault(); clearTimeout(timer); state.q = $('#q').value; load(true);
});
addEventListener('scroll', () => {
  if (innerHeight + scrollY > document.body.offsetHeight - 900) load(false);
});

/* --------------------------------------------------------------- lightbox */
async function openLightbox(i) {
  const p = state.results[i];
  if (!p) return;
  state.cursor = i;
  $('#lightbox').hidden = false;
  $('#lb-img').src = `/api/thumb/${p.blake3}/view`;
  $('#lb-img').alt = p.caption || p.filename || '';
  $('#lb-rail').innerHTML = '<div class="lb-sec">loading</div>';
  try {
    const d = await api(`/api/photos/${p.id}`);
    const a = d.annotation || {}, ph = d.photo;
    const row = (k, v) => v ? `<div class="row"><span>${esc(k)}</span><span>${esc(v)}</span></div>` : '';
    const named = d.faces.filter(f => f.person_name).map(f => f.person_name);
    const edited = a.model === 'manual';
    $('#lb-rail').innerHTML =
      `<h3 class="editable" id="cap" contenteditable="plaintext-only"
           data-empty="Describe this photo…">${esc(a.caption || '')}</h3>
       <div class="editrow">
         <span class="src">${edited ? 'edited by you' : a.model ? 'described by ' +
           esc(String(a.model).split('/').pop()) : 'not described yet'}</span>
         ${edited ? '<button id="revert">use the model\'s version</button>' : ''}
       </div>` +
      `<div class="lb-sec">tags</div><div class="tagedit" id="tagedit">${d.tags.map(t =>
        `<span class="chip ${t.source === 'manual' ? 'manual' : ''}" data-tag="${esc(t.name)}"
               data-source="${esc(t.source)}">${esc(t.name)}` +
        (t.source === 'manual' ? '<button aria-label="Remove tag">&times;</button>' : '') +
        '</span>').join('')}<input id="newtag" placeholder="+ add tag" aria-label="Add a tag"></div>` +
      (d.faces.length ? `<div class="lb-sec">faces</div><div class="facerow-lb">${d.faces.map(f =>
        `<div class="facelb" data-face="${f.id}">
           <img src="/api/face/${f.id}" alt="">
           <b>${esc(f.person_name || 'unnamed')}</b>
           ${f.person_name ? '<button class="x" title="Not this person">&times;</button>' : ''}
         </div>`).join('')}</div>` : '') +
      `<div class="lb-sec">details</div>` +
      row('taken', when(ph.taken_at)) + row('scene', a.scene) + row('setting', a.setting) +
      row('size', ph.width && ph.height ? `${ph.width} × ${ph.height}` : '') +
      row('camera', camera(ph)) +
      row('lens', ph.lens) + row('iso', ph.iso) +
      row('gps', ph.gps_lat ? `${ph.gps_lat.toFixed(5)}, ${ph.gps_lon.toFixed(5)}` : '') +
      `<div class="lb-sec">files</div>` +
      d.files.map(f => `<div class="row"><span>${esc(f.drive)}${f.online ? '' : ' (offline)'}</span>` +
        `<span>${esc(f.rel_path)}</span></div>`).join('') +
      `<div class="lb-sec">text in image</div>
       <div class="ocr editable" id="ocr" contenteditable="plaintext-only"
            data-empty="No text found. Click to type it in…">${esc(a.ocr_text || '')}</div>`;
    wireTagEditor(p.id, d);
    wireAnnotationEditor(p.id);
  } catch (e) {
    $('#lb-rail').innerHTML = `<div class="lb-sec">could not load details: ${esc(e.message)}</div>`;
  }
}

/* Caption and transcribed text save on blur, not on every keystroke: this is
   prose, and a request per character would be both wasteful and lossy if one
   failed mid-sentence. Escape abandons the edit. */
function wireAnnotationEditor(photoId) {
  /* The first edit turns a model description into yours, so the row has to gain
     a way back. Building it here rather than only on next open means the escape
     hatch exists the moment you might want it. */
  const markEdited = (id) => {
    const row = $('.editrow');
    if (!row) return;
    row.querySelector('.src').textContent = 'edited by you';
    if (row.querySelector('#revert')) return;
    const btn = document.createElement('button');
    btn.id = 'revert';
    btn.textContent = "use the model's version";
    btn.addEventListener('click', () => revert(id));
    row.appendChild(btn);
  };

  const revert = async (id) => {
    if (!confirm("Discard your edits and go back to the model's description?")) return;
    await api(`/api/photos/${id}/annotation`, { method: 'DELETE' });
    openLightbox(state.cursor);
  };

  const save = async (field, el) => {
    const value = el.textContent.trim();
    if (value === el.dataset.saved) return;
    el.classList.add('saving');
    try {
      await api(`/api/photos/${photoId}/annotation`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [field]: value })
      });
      el.dataset.saved = value;
      el.classList.remove('saving');
      el.classList.add('saved');
      setTimeout(() => el.classList.remove('saved'), 1200);
      markEdited(photoId);
    } catch (e) {
      el.classList.remove('saving');
      alert(`Could not save: ${e.message}`);
    }
  };

  for (const [id, field] of [['cap', 'caption'], ['ocr', 'ocr_text']]) {
    const el = $(`#${id}`);
    if (!el) continue;
    el.dataset.saved = el.textContent.trim();
    el.addEventListener('blur', () => save(field, el));
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { el.textContent = el.dataset.saved; el.blur(); }
      // Enter commits the caption; the transcription may legitimately be
      // several lines, so it keeps normal newline behaviour.
      if (e.key === 'Enter' && id === 'cap') { e.preventDefault(); el.blur(); }
    });
  }

  const existing = $('#revert');
  if (existing) existing.addEventListener('click', () => revert(photoId));
}

/* Manual tags are stored separately from the model's, so saving replaces only
   the user's own set and never destroys what the captioner produced. */
function wireTagEditor(photoId, d) {
  const box = $('#tagedit');
  if (!box) return;
  const manual = () => [...box.querySelectorAll('.chip[data-source="manual"]')]
    .map(c => c.dataset.tag);
  const save = (tags) => api(`/api/photos/${photoId}/tags`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tags })
  });

  box.querySelectorAll('.chip button').forEach(btn =>
    btn.addEventListener('click', async () => {
      const chip = btn.closest('.chip');
      const next = manual().filter(t => t !== chip.dataset.tag);
      chip.remove();
      await save(next);
    }));

  const input = $('#newtag');
  input.addEventListener('keydown', async (e) => {
    if (e.key !== 'Enter') return;
    e.preventDefault();
    const name = input.value.trim().toLowerCase();
    if (!name || manual().includes(name)) { input.value = ''; return; }
    const chip = document.createElement('span');
    chip.className = 'chip manual';
    chip.dataset.tag = name;
    chip.dataset.source = 'manual';
    chip.innerHTML = `${esc(name)}<button aria-label="Remove tag">&times;</button>`;
    chip.querySelector('button').addEventListener('click', async () => {
      chip.remove();
      await save(manual());
    });
    box.insertBefore(chip, input);
    input.value = '';
    await save(manual());
  });

  $('#lb-rail').querySelectorAll('.facelb .x').forEach(btn =>
    btn.addEventListener('click', async () => {
      const card = btn.closest('.facelb');
      await api(`/api/faces/${card.dataset.face}/detach`, { method: 'POST' });
      card.querySelector('b').textContent = 'unnamed';
      btn.remove();
    }));
}
const closeLb = () => { $('#lightbox').hidden = true; $('#lb-img').src = ''; };
const step = (d) => {
  const n = state.cursor + d;
  if (n >= 0 && n < state.results.length) openLightbox(n);
};
$('.lb-close').addEventListener('click', closeLb);
$('.lb-nav.prev').addEventListener('click', () => step(-1));
$('.lb-nav.next').addEventListener('click', () => step(1));

addEventListener('keydown', e => {
  if (!$('#picker').hidden && e.key === 'Escape') { closePicker(); return; }
  if (!$('#lightbox').hidden) {
    if (e.key === 'Escape') closeLb();
    if (e.key === 'ArrowLeft') step(-1);
    if (e.key === 'ArrowRight') step(1);
    return;
  }
  if (e.key === '/' && document.activeElement !== $('#q')) { e.preventDefault(); $('#q').focus(); }
});

/* ----------------------------------------------------------------- people */
async function loadPeople() {
  const [pe, cl] = await Promise.all([api('/api/people'), api('/api/clusters')]);
  $('#named').innerHTML = pe.people.filter(p => p.name).map(p =>
    `<div class="person" data-id="${p.id}" data-name="${esc(p.name)}">
       <button class="crop go">${p.cover_face ? `<img src="/api/face/${p.cover_face}" alt="">` : ''}</button>
       <b>${esc(p.name)}</b><span>${p.n} photo${p.n === 1 ? '' : 's'}</span>
       <button class="edit">rename</button>
     </div>`).join('') || '<p style="color:var(--faint)">Nobody named yet.</p>';

  $('#named').querySelectorAll('.person .go').forEach(b => b.addEventListener('click', () => {
    const card = b.closest('.person');
    $('#q').value = `person:"${card.dataset.name}"`;
    state.q = $('#q').value; show('grid'); load(true);
  }));
  $('#named').querySelectorAll('.person .edit').forEach(b => b.addEventListener('click', async () => {
    const card = b.closest('.person');
    const name = prompt(`Rename ${card.dataset.name} to:`, card.dataset.name);
    if (name === null) return;
    if (!name.trim()) {
      if (!confirm(`Un-name ${card.dataset.name}? Their faces go back to the naming queue.`)) return;
      await api(`/api/people/${card.dataset.id}`, { method: 'DELETE' });
    } else {
      await api(`/api/people/${card.dataset.id}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim() })
      }).catch(e => alert(e.message));
    }
    loadPeople();
  }));

  $('#unnamed-h').hidden = cl.clusters.length === 0;
  $('#clusters').innerHTML = cl.clusters.map(c =>
    `<div class="cluster" data-id="${c.cluster_id}">
       <label class="pick"><input type="checkbox"> same person as…</label>
       <div class="n">${c.count} face${c.count === 1 ? '' : 's'}</div>
       <div class="faces">${c.faces.slice(0, 5).map(f =>
         `<div><img src="/api/face/${f.id}" alt=""></div>`).join('')}</div>
       <form><input placeholder="Who is this?" aria-label="Name this person"><button>Name</button></form>
     </div>`).join('');
  wireMerge();
  $('#clusters').querySelectorAll('.cluster form').forEach(f =>
    f.addEventListener('submit', async e => {
      e.preventDefault();
      const box = f.closest('.cluster');
      const name = f.querySelector('input').value.trim();
      if (!name) return;
      await api(`/api/clusters/${box.dataset.id}/name`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      loadPeople();  // keeps focus flowing to the next unnamed cluster
    }));
}

/* Clustering splits one person across several groups whenever lighting or angle
   varies, so merging is the most common correction on this screen. */
function wireMerge() {
  const boxes = [...document.querySelectorAll('.cluster .pick input')];
  const bar = document.createElement('div');
  bar.className = 'mergebar';
  bar.hidden = true;
  bar.innerHTML = `<span id="mergecount"></span>
    <input id="mergename" placeholder="Name them (optional)" aria-label="Name for merged person">
    <button id="mergego">Merge</button>
    <button class="ghost" id="mergecancel">Cancel</button>`;
  $('#clusters').before(bar);

  const picked = () => boxes.filter(b => b.checked)
    .map(b => +b.closest('.cluster').dataset.id);
  const refresh = () => {
    const n = picked().length;
    bar.hidden = n < 1;
    $('#mergecount').textContent = `${n} group${n === 1 ? '' : 's'} selected`;
    boxes.forEach(b => b.closest('.cluster').classList.toggle('picked', b.checked));
  };
  boxes.forEach(b => b.addEventListener('change', refresh));

  $('#mergecancel').addEventListener('click', () => {
    boxes.forEach(b => { b.checked = false; });
    refresh();
  });
  $('#mergego').addEventListener('click', async () => {
    const ids = picked();
    if (ids.length < 2) { alert('Pick at least two groups to merge.'); return; }
    await api('/api/clusters/merge', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cluster_ids: ids, name: $('#mergename').value.trim() || null })
    });
    loadPeople();
  });
}

/* ------------------------------------------------------------- duplicates */
const bytes = (n) => {
  if (!n) return '—';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${i ? n.toFixed(1) : n} ${u[i]}`;
};

async function loadDupes() {
  const near = $('#near-toggle').checked;
  $('#dupes').innerHTML = '<div class="empty">Looking…</div>';
  const d = await api(`/api/duplicates?near=${near}`);
  const parts = [];
  if (d.exact.length) {
    parts.push(`<h3 class="sub">Identical files — ${bytes(d.wasted_bytes)} recoverable</h3>`);
    parts.push(d.exact.map(r => `<div class="dupe">
        <div class="hd"><b>${r.n} copies</b><span>${bytes(r.bytes)} each</span></div>
        <div class="shots"><img src="/api/thumb/${r.blake3}/grid" alt=""></div>
        <div class="locs">${esc(r.locations || '')}</div>
      </div>`).join(''));
  }
  if (near) {
    parts.push(`<h3 class="sub">Visually similar</h3>`);
    parts.push(d.near.length ? d.near.map(g => `<div class="dupe">
        <div class="hd"><b>${g.photos.length} similar photos</b>
          <span>${bytes(g.photos.reduce((a, p) => a + (p.bytes || 0), 0))} total</span></div>
        <div class="shots">${g.photos.slice(0, 8).map(p =>
          `<img src="/api/thumb/${p.blake3}/grid" alt="${esc(p.filename || '')}">`).join('')}</div>
        <div class="locs">${g.photos.map(p => esc(p.filename || '')).join('\n')}</div>
      </div>`).join('') : '<div class="empty">No near-identical photos found.</div>');
  }
  if (!d.exact.length && !near) parts.push('<div class="empty">No photo is stored twice.</div>');
  $('#dupes').innerHTML = parts.join('');
}
$('#near-toggle').addEventListener('change', loadDupes);

/* ----------------------------------------------------------------- places */
let map;
async function loadPlaces() {
  const d = await api('/api/map');
  if (!d.points.length) {
    $('#map').hidden = true;
    $('#map-empty').hidden = false;
    $('#map-empty').textContent =
      'No photo in the library recorded a location. Phones only write GPS when location is enabled for the camera.';
    return;
  }
  $('#map').hidden = false;
  $('#map-empty').hidden = true;
  if (!window.L) {
    await new Promise((ok, fail) => {
      const css = document.createElement('link');
      css.rel = 'stylesheet';
      css.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
      document.head.appendChild(css);
      const js = document.createElement('script');
      js.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
      js.onload = ok; js.onerror = fail;
      document.head.appendChild(js);
    }).catch(() => null);
  }
  if (!window.L) {
    $('#map').hidden = true;
    $('#map-empty').hidden = false;
    $('#map-empty').textContent = 'The map needs an internet connection to load its tiles.';
    return;
  }
  if (!map) {
    map = L.map('map');
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      { attribution: '© OpenStreetMap', maxZoom: 19 }).addTo(map);
  }
  const group = L.featureGroup(d.points.map(p =>
    L.marker([p.gps_lat, p.gps_lon]).bindPopup(
      `<img src="/api/thumb/${p.blake3}/grid" alt="">${esc(p.caption || '')}`))).addTo(map);
  map.fitBounds(group.getBounds().pad(0.2));
}

/* ---------------------------------------------------------------- library */
let scanPoll;

async function loadLibrary() {
  const [s, rr] = await Promise.all([api('/api/stats'), api('/api/roots')]);
  const cap = s.photos ? Math.round(s.captioned / s.photos * 100) : 0;
  $('#stats').innerHTML = [
    ['photos', s.photos], ['described', `${cap}%`], ['faces', s.faces],
    ['people', s.people], ['tags', s.tags], ['offline files', s.offline],
  ].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');

  $('#roots').innerHTML = rr.roots.map(r => `
    <div class="root" data-id="${r.id}">
      <div>
        <b>${esc(r.label || r.rel_path || '/')}</b>
        <div class="p">${esc(r.full_path || `${r.volume_label} / ${r.rel_path || ''}`)}</div>
      </div>
      <div class="acts">
        <span class="prog" data-prog></span>
        <span class="pill ${r.online ? 'on' : 'off'}">${
          r.online ? `${r.photos} photos` : 'drive not connected'}</span>
        ${r.online ? '<button class="ghostbtn" data-act="scan">Rescan</button>' : ''}
        <button class="ghostbtn danger" data-act="remove">Remove</button>
      </div>
    </div>`).join('') ||
    `<p style="color:var(--faint)">No folders yet. Add one to start indexing.</p>`;

  $('#roots').querySelectorAll('[data-act="scan"]').forEach(b =>
    b.addEventListener('click', async () => {
      await api(`/api/roots/${b.closest('.root').dataset.id}/scan`, { method: 'POST' })
        .catch(e => alert(e.message));
      watchScans();
    }));

  $('#roots').querySelectorAll('[data-act="remove"]').forEach(b =>
    b.addEventListener('click', async () => {
      const card = b.closest('.root');
      const name = card.querySelector('b').textContent;
      if (!confirm(`Remove "${name}" from the library?\n\n` +
                   `Nothing on disk is deleted. Photos that exist in another ` +
                   `indexed folder keep their other copies.`)) return;
      const r = await api(`/api/roots/${card.dataset.id}`, { method: 'DELETE' })
        .catch(e => { alert(e.message); return null; });
      if (r) loadLibrary();
    }));

  watchScans();
  loadPipeline();
}

/* Scans run on a server thread; poll only while one is actually going. */
async function watchScans() {
  clearInterval(scanPoll);
  const tick = async () => {
    let d;
    try { d = await api('/api/scans'); } catch { return; }
    const active = Object.entries(d.scanning);
    // A finished scan hands over to the indexer (see auto_process), so pick the
    // pipeline poll up where the scan poll left off rather than going quiet.
    if (!active.length) { clearInterval(scanPoll); loadLibraryCounts(); loadPipeline(); return; }
    for (const [id, st] of active) {
      const el = $(`#roots .root[data-id="${id}"] [data-prog]`);
      if (!el) continue;
      el.textContent = st.state === 'scanning' ? `scanning… ${st.seen} seen, ${st.new} new`
        : st.state === 'failed' ? `failed: ${st.error || ''}`.slice(0, 60)
        : `done — ${st.new} new`;
    }
  };
  await tick();
  scanPoll = setInterval(tick, 1200);
}

async function loadLibraryCounts() {
  if ($('#library-view').hidden) return;
  const [s, rr] = await Promise.all([api('/api/stats'), api('/api/roots')]);
  const cap = s.photos ? Math.round(s.captioned / s.photos * 100) : 0;
  $('#stats').innerHTML = [
    ['photos', s.photos], ['described', `${cap}%`], ['faces', s.faces],
    ['people', s.people], ['tags', s.tags], ['offline files', s.offline],
  ].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');
  rr.roots.forEach(r => {
    const pill = $(`#roots .root[data-id="${r.id}"] .pill`);
    if (pill && r.online) pill.textContent = `${r.photos} photos`;
  });
}

/* ------------------------------------------------------------ folder picker */
const picker = { path: null, selected: null };

async function openPicker(path) {
  $('#picker').hidden = false;
  await showDir(path || null);
}
function closePicker() {
  $('#picker').hidden = true;
  picker.selected = null;
  $('#picked').textContent = '—';
  $('#picker-add').disabled = true;
  $('#rootlabel').value = '';
}

async function showDir(path) {
  let d;
  try {
    d = await api(`/api/browse${path ? `?path=${encodeURIComponent(path)}` : ''}`);
  } catch (e) {
    $('#dirs').innerHTML = `<li class="empty-note">${esc(e.message)}</li>`;
    return;
  }
  picker.path = d.path;
  select(d.path, d.photos_here);

  $('#shortcuts').innerHTML = d.shortcuts.map(s =>
    `<button data-path="${esc(s.path)}">${esc(s.label)}</button>`).join('');
  $('#shortcuts').querySelectorAll('button').forEach(b =>
    b.addEventListener('click', () => showDir(b.dataset.path)));

  const parts = d.path.split('/').filter(Boolean);
  $('#crumbs').innerHTML =
    `<button data-path="/">/</button>` +
    parts.map((seg, i) =>
      ` <button data-path="${esc('/' + parts.slice(0, i + 1).join('/'))}">${esc(seg)}</button>`)
      .join(' /');
  $('#crumbs').querySelectorAll('button').forEach(b =>
    b.addEventListener('click', () => showDir(b.dataset.path)));

  $('#dirs').innerHTML = d.entries.length
    ? d.entries.map(e => `
        <li data-path="${esc(e.path)}" data-photos="${e.photos}">
          <span class="nm">${esc(e.name)}</span>
          <span class="ct ${e.photos ? 'has' : ''}">${
            e.photos ? `${e.photos}${e.photos > 400 ? '+' : ''} photos` : ''}</span>
          ${e.children ? '<span class="go">&rsaquo;</span>' : ''}
        </li>`).join('')
    : `<li class="empty-note">No sub-folders${
        d.photos_here ? ` — but ${d.photos_here} photos are right here.` : '.'}</li>`;

  $('#dirs').querySelectorAll('li[data-path]').forEach(li => {
    // One click selects this folder; the chevron descends into it. Selecting a
    // folder you are looking at is the common case and should not need a
    // separate "use this one" step.
    li.addEventListener('click', (ev) => {
      if (ev.target.classList.contains('go')) { showDir(li.dataset.path); return; }
      $('#dirs').querySelectorAll('li').forEach(x => x.classList.remove('on'));
      li.classList.add('on');
      select(li.dataset.path, +li.dataset.photos);
    });
    li.addEventListener('dblclick', () => showDir(li.dataset.path));
  });
}

function select(path, photos) {
  picker.selected = path;
  $('#picked').textContent = path;
  $('#picker-add').disabled = false;
  $('#picker-add').textContent = photos
    ? `Add to library (${photos}${photos > 400 ? '+' : ''} photos here)`
    : 'Add to library';
}

$('#addroot').addEventListener('click', () => openPicker(picker.path));
$('#picker-close').addEventListener('click', closePicker);
$('#picker').addEventListener('click', (e) => { if (e.target.id === 'picker') closePicker(); });
$('#picker-add').addEventListener('click', async () => {
  if (!picker.selected) return;
  $('#picker-add').disabled = true;
  const r = await api('/api/roots', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      path: picker.selected, label: $('#rootlabel').value.trim() || null, scan: true })
  }).catch(e => { alert(e.message); $('#picker-add').disabled = false; return null; });
  if (!r) return;
  closePicker();
  await loadLibrary();
});

/* ---------------------------------------------------------------- indexing */
/* A scan only queues work; these are the controls that drain that queue, so
   nothing about running this library needs a terminal. */

const STAGE_INFO = {
  thumbs:  { name: 'Thumbnails',
             blurb: 'Cached previews. Every later stage reads these, so it goes first.' },
  embed:   { name: 'Meaning',
             blurb: 'Vectors that let "mountains" find a mountain nobody tagged.' },
  faces:   { name: 'Faces',
             blurb: 'Finds faces. Grouping them into people is the step after.' },
  caption: { name: 'Captions & text',
             blurb: 'Describes each photo and reads any text in it. The slow one.' },
  cluster: { name: 'Group people',
             blurb: 'Sorts detected faces into people and matches new ones to names.' },
};

let runPoll = null;
const stopRunPoll = () => { clearInterval(runPoll); runPoll = null; };

async function loadPipeline() {
  let d;
  try { d = await api('/api/process'); } catch { return; }
  renderPipeline(d);
  // Poll only while something is running: an idle library should not wake the
  // server every second forever.
  if (d.running && !runPoll) runPoll = setInterval(pollRun, 900);
  if (!d.running) stopRunPoll();
}

async function pollRun() {
  if ($('#library-view').hidden) return stopRunPoll();
  let d;
  try { d = await api('/api/process'); } catch { return; }
  renderPipeline(d);
  if (!d.running) {
    stopRunPoll();
    loadLibraryCounts();  // new thumbnails and captions move the counters
  }
}

function renderPipeline(d) {
  const running = d.running;
  $('#run-all').disabled = running;
  $('#run-all').textContent = running ? 'Indexing…' : 'Index everything';
  $('#run-cancel').hidden = !running;

  const p = d.progress || {};
  const bar = $('#runprog');
  bar.hidden = !running;
  if (running) {
    const stage = d.stage || '';
    const seen = (p.done || 0) + (p.failed || 0);
    const total = (d.queue?.[stage]?.pending || 0) + seen;
    // Clustering reads every face at once and has no per-photo counter, so an
    // honest indeterminate bar beats one that never moves.
    bar.classList.toggle('indet', stage === 'cluster');
    bar.firstElementChild.style.width =
      stage === 'cluster' ? '' : `${total ? Math.min(100, seen / total * 100) : 0}%`;
  }

  $('#runstate').textContent = running
    ? `${STAGE_INFO[d.stage]?.name || d.stage || 'starting'} — ${p.done || 0} done` +
      `${p.failed ? `, ${p.failed} failed` : ''}` +
      `${p.skipped ? `, ${p.skipped} waiting on a drive` : ''}`
    : '';

  $('#stages').innerHTML = (d.stages_all || []).map(stage => {
    const q = d.queue?.[stage] || {};
    const info = STAGE_INFO[stage] || { name: stage, blurb: '' };
    const done = q.done || 0, pending = q.pending || 0, failed = q.failed || 0;
    const res = d.results?.[stage];
    const counts = stage === 'cluster'
      ? (res ? `<span class="ok">${res.new_clusters} groups, ${res.anchored} matched</span>`
             : '<span class="idle">runs on demand</span>')
      : [pending ? `<span class="wait">${pending} waiting</span>` : '',
         done ? `<span class="ok">${done} done</span>` : '',
         failed ? `<span class="bad" data-retry="${stage}" ` +
                  `title="Put these back in the queue">${failed} failed</span>` : '',
        ].filter(Boolean).join('') || '<span class="idle">nothing queued</span>';
    return `
      <div class="stagecard${running && d.stage === stage ? ' now' : ''}">
        <div class="stagehd">
          <b>${info.name}</b>
          <button class="ghostbtn" data-run="${stage}"${running ? ' disabled' : ''}>Run</button>
        </div>
        <p>${info.blurb}</p>
        <div class="counts">${counts}</div>
      </div>`;
  }).join('');

  $('#stages').querySelectorAll('[data-run]').forEach(b =>
    b.addEventListener('click', () => startRun([b.dataset.run])));
  $('#stages').querySelectorAll('[data-retry]').forEach(el =>
    el.addEventListener('click', async () => {
      await api(`/api/process/retry-failed?stage=${el.dataset.retry}`, { method: 'POST' })
        .catch(e => alert(e.message));
      loadPipeline();
    }));

  const note = $('#runnote');
  note.className = 'note';
  if (d.error) { note.classList.add('bad'); note.textContent = d.error; }
  else if (d.cancelled) note.textContent = 'Stopped. Nothing was lost — press Run to carry on.';
  else if (!running && d.finished_at) note.textContent = summarise(d.results);
  else if (!running) note.textContent =
    'Scanning a folder finds photos. Indexing is what makes them searchable.';
  else note.textContent = '';
}

function summarise(results) {
  const parts = Object.entries(results || {}).map(([stage, r]) => {
    const name = STAGE_INFO[stage]?.name || stage;
    return stage === 'cluster'
      ? `${name}: ${r.new_clusters} groups`
      : `${name}: ${r.done} done${r.failed ? `, ${r.failed} failed` : ''}`;
  });
  return parts.length ? `Finished — ${parts.join(' · ')}` : 'Finished — nothing was waiting.';
}

async function startRun(stages) {
  const r = await api('/api/process', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stages: stages || null }),
  }).catch(e => { alert(e.message); return null; });
  if (r) loadPipeline();
}

$('#run-all').addEventListener('click', () => startRun(null));
$('#run-cancel').addEventListener('click', async () => {
  $('#run-cancel').disabled = true;
  await api('/api/process/cancel', { method: 'POST' }).catch(() => {});
  $('#run-cancel').disabled = false;
});

$('#groupby').value = state.group;
$('#sortby').value = state.sort;
$('#groupby').addEventListener('change', (e) => {
  state.group = e.target.value;
  try { localStorage.setItem('pa.group', state.group); } catch { /* private window */ }
  render();  // grouping is client-side: no refetch needed
});
$('#sortby').addEventListener('change', (e) => {
  state.sort = e.target.value;
  try { localStorage.setItem('pa.sort', state.sort); } catch { /* private window */ }
  load(true);  // ordering comes from the server
});

load(true);

/* ---------------------------------------------------------------- settings */
/* The config file is the source of truth; this screen is a typed editor for it.
   Fields are described rather than hand-written as markup, so a new setting is
   one line here and the labels stay in one place. */

const SETTINGS = [
  { key: 'caption', title: 'Captions, tags and OCR',
    blurb: 'The vision model that describes each photo and reads text in it. ' +
           'Runs wherever you point it — nothing is sent off this machine unless ' +
           'you set base_url to something that is.',
    fields: [
      { k: 'provider', label: 'Provider', type: 'select', from: 'caption',
        help: 'All of these speak the same OpenAI-shaped API; only the address and model differ.' },
      { k: 'base_url', label: 'Server address', type: 'text',
        help: 'Under WSL, 127.0.0.1 is the Linux side — not the Windows host.' },
      { k: 'model', label: 'Model', type: 'text' },
      { k: 'long_edge', label: 'Image size sent', type: 'number', unit: 'px',
        help: 'Longest edge the photo is scaled to. Bigger reads small text better and costs time.' },
      { k: 'max_tokens', label: 'Reply budget', type: 'number', unit: 'tokens',
        help: 'Text-heavy screenshots need room for a full OCR pass.' },
      { k: 'temperature', label: 'Temperature', type: 'number', step: 0.05 },
      { k: 'timeout_s', label: 'Timeout', type: 'number', unit: 'seconds' },
      { k: 'reasoning_effort', label: 'Thinking before answering', type: 'select',
        options: ['none', 'low', 'medium', 'high'],
        help: 'Models that think first cost roughly 4x here for no gain on captioning. ' +
              'Raise it only if captions start looking careless.' },
      { k: 'plugins', label: 'Plugin files', type: 'list',
        help: 'Extra .py files to import so their providers become selectable. One per line.' },
    ] },
  { key: 'embed', title: 'Meaning search',
    blurb: 'Turns photos and your typed words into vectors, which is what makes ' +
           '"golden hour" work on photos nobody tagged. Runs in this process.',
    fields: [
      { k: 'provider', label: 'Provider', type: 'select', from: 'image_embed' },
      { k: 'model', label: 'Model', type: 'text' },
      { k: 'device', label: 'Runs on', type: 'select', options: ['cuda', 'cpu'] },
      { k: 'batch_size', label: 'Batch size', type: 'number',
        help: 'Photos embedded per GPU pass. Lower this if you run out of memory.' },
      { k: 'dim', label: 'Vector size', type: 'number',
        help: 'Must match the model. Changing it invalidates every stored vector.' },
      { k: 'min_score', label: 'Match floor', type: 'number', step: 0.005,
        help: 'A real match scores about 0.12; unrelated content sits below 0.06. ' +
              'Lower to widen results, raise if search feels noisy.' },
      { k: 'rel_score', label: 'Share of best hit', type: 'number', step: 0.05,
        help: 'A hit must also be at least this fraction of the top result.' },
    ] },
  { key: 'face', title: 'Faces and people',
    blurb: 'Detects faces, then groups them so you can name each person once.',
    fields: [
      { k: 'provider', label: 'Provider', type: 'select', from: 'face' },
      { k: 'model', label: 'Model', type: 'text', help: 'buffalo_l is accurate, buffalo_s is faster.' },
      { k: 'device', label: 'Runs on', type: 'select', options: ['cuda', 'cpu'] },
      { k: 'det_size', label: 'Detector size', type: 'number', unit: 'px' },
      { k: 'min_det_score', label: 'Detection confidence', type: 'number', step: 0.05 },
      { k: 'min_face_px', label: 'Smallest face kept', type: 'number', unit: 'px',
        help: 'Faces below this cluster badly and mostly add noise.' },
      { k: 'cluster_eps', label: 'Same-person distance', type: 'number', step: 0.01,
        help: 'Lower splits one person into more groups; higher starts merging strangers.' },
      { k: 'min_cluster_size', label: 'Smallest group', type: 'number',
        help: 'How many faces it takes before a group is worth offering you to name.' },
    ] },
  { key: 'thumbs', title: 'Thumbnails',
    blurb: 'Cached previews. These are what keeps the library browsable with the ' +
           'external drives unplugged. Changing sizes needs the stage re-run.',
    fields: [
      { k: 'grid_px', label: 'Grid size', type: 'number', unit: 'px' },
      { k: 'view_px', label: 'Full-view size', type: 'number', unit: 'px',
        help: 'Also the image the AI stages actually look at.' },
      { k: 'quality', label: 'Quality', type: 'number' },
      { k: 'format', label: 'Format', type: 'select', options: ['WEBP', 'JPEG', 'PNG'] },
    ] },
  { key: 'sidecar', title: 'XMP sidecars',
    blurb: 'Where exported metadata is written for Lightroom and digiKam.',
    fields: [
      { k: 'location', label: 'Write them', type: 'select', options: ['app', 'beside'],
        labels: { app: 'Inside the app folder (your photo folders untouched)',
                  beside: 'Beside each photo (what Lightroom reads directly)' } },
    ] },
  { key: 'server', title: 'Server and indexing',
    blurb: 'How the web app listens, and what happens on its own after a scan.',
    fields: [
      { k: 'host', label: 'Bind address', type: 'text',
        help: '127.0.0.1 keeps it to this machine. There is no login, so do not ' +
              'expose it on a shared network.' },
      { k: 'port', label: 'Port', type: 'number' },
      { k: 'scan_workers', label: 'Scan threads', type: 'number' },
      { k: 'auto_process', label: 'After a scan, run', type: 'stages',
        help: 'A scan only queues work. Without this, nothing indexes until you ' +
              'press a button. Thumbnails alone is the safe default: no GPU, no ' +
              'model server, and it is the stage whose absence shows on every tile.' },
    ] },
];

let cfgData = null;  // the last /api/config response, for dirty-checking

async function loadSettings() {
  let d;
  try { d = await api('/api/config'); } catch (e) {
    $('#settings-form').innerHTML = `<p class="note bad">${esc(e.message)}</p>`;
    return;
  }
  cfgData = d;
  renderSettings(d);
  renderPaths(d.paths);
  loadHealth();
}

function fieldHTML(section, f, value, d) {
  const id = `set-${section}-${f.k}`;
  const path = `${section}.${f.k}`;
  const flag = d.restart_required[path]
    ? `<span class="flag warn" title="${esc(d.restart_required[path])}">needs restart</span>`
    : d.reindex_required[path]
    ? `<span class="flag warn" title="Everything already indexed was computed with the old value">needs re-index</span>`
    : '';
  let input;
  if (f.type === 'select') {
    const opts = f.options || d.providers[f.from] || [];
    input = `<select id="${id}">${opts.map(o =>
      `<option value="${esc(o)}"${String(value) === String(o) ? ' selected' : ''}>${
        esc(f.labels?.[o] || o)}</option>`).join('')}</select>`;
  } else if (f.type === 'number') {
    input = `<input id="${id}" type="number" step="${f.step || 1}" value="${esc(value)}">`;
  } else if (f.type === 'list') {
    input = `<textarea id="${id}" rows="2" spellcheck="false">${
      esc((value || []).join('\n'))}</textarea>`;
  } else if (f.type === 'stages') {
    input = `<div class="stagepick">${d.stages.map(s =>
      `<label class="toggle"><input type="checkbox" data-stage="${s}"${
        (value || []).includes(s) ? ' checked' : ''}> ${
        esc(STAGE_INFO[s]?.name || s)}</label>`).join('')}</div>`;
  } else {
    input = `<input id="${id}" type="text" value="${esc(value)}" spellcheck="false">`;
  }
  return `
    <div class="field" data-section="${section}" data-key="${f.k}" data-type="${f.type}">
      <label for="${id}"><span class="fl">${esc(f.label)}${flag}</span>
        ${f.help ? `<span class="fh">${esc(f.help)}</span>` : ''}</label>
      <div class="fi">${input}${f.unit ? `<span class="unit">${esc(f.unit)}</span>` : ''}</div>
    </div>`;
}

function renderSettings(d) {
  $('#settings-form').innerHTML = SETTINGS.map(sec => `
    <h3 class="sub">${esc(sec.title)}</h3>
    <div class="panel">
      <p class="secblurb">${esc(sec.blurb)}</p>
      ${sec.fields.map(f => fieldHTML(sec.key, f, d.config[sec.key][f.k], d)).join('')}
      <div class="secfoot">
        <button class="ghostbtn" data-reset="${sec.key}">Reset this section to defaults</button>
      </div>
    </div>`).join('') + `
    <div class="savebar" id="savebar">
      <span class="note" id="savenote">Stored in ${esc(d.path)}</span>
      <button class="ghostbtn" id="settings-revert">Discard changes</button>
      <button class="primary" id="settings-save" disabled>Save</button>
    </div>`;

  $('#settings-form').addEventListener('input', markDirty);
  $('#settings-form').addEventListener('change', markDirty);
  $('#settings-save').addEventListener('click', saveSettings);
  $('#settings-revert').addEventListener('click', () => renderSettings(cfgData));
  $('#settings-form').querySelectorAll('[data-reset]').forEach(b =>
    b.addEventListener('click', () => {
      const key = b.dataset.reset;
      // Defaults come from the server so this screen never has to keep its own
      // copy of them, which would drift the moment a default changed.
      cfgData = { ...cfgData, config: { ...cfgData.config, [key]: cfgData.defaults[key] } };
      renderSettings(cfgData);
      markDirty();
    }));
}

function readField(el) {
  const type = el.dataset.type;
  if (type === 'stages')
    return [...el.querySelectorAll('[data-stage]')].filter(c => c.checked)
      .map(c => c.dataset.stage);
  const input = el.querySelector('input, select, textarea');
  if (type === 'number') return Number(input.value);
  if (type === 'list') return input.value.split('\n').map(s => s.trim()).filter(Boolean);
  return input.value;
}

function collectSettings() {
  const out = {};
  $('#settings-form').querySelectorAll('.field').forEach(el => {
    (out[el.dataset.section] ||= {})[el.dataset.key] = readField(el);
  });
  return out;
}

function markDirty() {
  const now = collectSettings();
  const changed = [];
  for (const [sec, vals] of Object.entries(now))
    for (const [k, v] of Object.entries(vals))
      if (JSON.stringify(v) !== JSON.stringify(cfgData.config[sec][k])) changed.push(`${sec}.${k}`);

  $('#settings-save').disabled = !changed.length;
  const note = $('#savenote');
  const restart = changed.filter(c => cfgData.restart_required[c]);
  const reindex = [...new Set(changed.map(c => cfgData.reindex_required[c]).filter(Boolean))];
  if (!changed.length) { note.textContent = `Stored in ${cfgData.path}`; return; }
  note.textContent = `${changed.length} change${changed.length > 1 ? 's' : ''}` +
    (restart.length ? ` · ${restart.join(', ')} take effect after a restart` : '') +
    (reindex.length ? ` · re-run ${reindex.join(' and ')} to apply to existing photos` : '');
}

async function saveSettings() {
  const btn = $('#settings-save');
  btn.disabled = true;
  btn.textContent = 'Saving…';
  try {
    const d = await api('/api/config', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectSettings()),
    });
    cfgData = d;
    renderSettings(d);
    renderPaths(d.paths);
    $('#savenote').textContent = 'Saved.';
    loadHealth();  // a new base_url or model is exactly when this matters
  } catch (e) {
    alert(e.message);
    btn.disabled = false;
  } finally {
    btn.textContent = 'Save';
  }
}

function renderPaths(paths) {
  $('#paths').innerHTML = Object.entries(paths).map(([k, v]) =>
    `<div class="row"><span>${esc(k)}</span><code>${esc(v)}</code></div>`).join('');
}

async function loadHealth() {
  $('#health').innerHTML = '<p class="note">checking…</p>';
  let d;
  try { d = await api('/api/config/check'); } catch (e) {
    $('#health').innerHTML = `<p class="note bad">${esc(e.message)}</p>`;
    return;
  }
  $('#health').innerHTML = d.checks.map(c => `
    <div class="hrow ${c.level}">
      <span class="hbadge">${c.level === 'ok' ? 'OK' : c.level === 'warn' ? 'WARN' : 'FAIL'}</span>
      <div>
        <b>${esc(c.label)}</b>
        <div class="hdetail">${esc(c.detail)}</div>
        ${c.message ? `<div class="hmsg">${esc(c.message)}</div>` : ''}
        ${c.hint ? `<div class="hhint">${esc(c.hint)}</div>` : ''}
      </div>
    </div>`).join('');
}

$('#recheck').addEventListener('click', loadHealth);

/* ------------------------------------------------------------ maintenance */
const TOOLS = {
  'sidecar-export': () => ['/api/sidecar/export', {
    beside: $('#sc-beside').checked, overwrite: $('#sc-overwrite').checked }],
  'sidecar-import': () => ['/api/sidecar/import', {}],
  'prune': () => ['/api/prune', { keep_missing: $('#pr-keep').checked }],
  'rebuild-thumbs': () => ['/api/process/reset', { stage: 'thumbs', rebuild: true }],
  'retry': () => ['/api/process/retry-failed', {}],
};

const TOOL_SAID = {
  'sidecar-export': r => `Wrote ${r.written} sidecars to ${r.location}. ` +
    `${r.skipped} already existed, ${r.offline} are on a disconnected drive.`,
  'sidecar-import': r => `Read ${r.found} sidecars, imported ${r.tags} keywords.`,
  'prune': r => r.dropped ? `Dropped ${r.dropped} photos with no file left anywhere.`
                          : 'Nothing to prune.',
  'rebuild-thumbs': r => `Cleared ${r.removed} cached files and queued ${r.requeued} ` +
    `photos. Starting now — watch it on the Library tab.`,
  'retry': r => r.requeued ? `Put ${r.requeued} failed jobs back in the queue.`
                           : 'No failed jobs to retry.',
};

document.querySelectorAll('[data-tool]').forEach(b =>
  b.addEventListener('click', async () => {
    const tool = b.dataset.tool;
    if (tool === 'rebuild-thumbs' && !confirm(
      'Delete every cached thumbnail and make them again?\n\n' +
      'Your photos are not touched. Anything on a drive that is currently ' +
      'unplugged loses its preview until that drive is back.')) return;
    if (tool === 'prune' && !confirm(
      'Drop photos that have no readable file left anywhere?\n\n' +
      'Nothing on disk is deleted. Photos on a disconnected drive are offline, ' +
      'not missing, and are kept.')) return;
    const [url, body] = TOOLS[tool]();
    const note = $('#toolnote');
    note.className = 'note';
    note.textContent = 'working…';
    b.disabled = true;
    try {
      const r = await api(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      note.textContent = TOOL_SAID[tool](r);
      if (r.errors?.length) note.textContent += ` First error: ${r.errors[0]}`;
      // Requeueing only fills the queue; something still has to drain it.
      if (tool === 'rebuild-thumbs' && r.requeued) await startRun(['thumbs']);
    } catch (e) {
      note.classList.add('bad');
      note.textContent = e.message;
    } finally {
      b.disabled = false;
    }
  }));
