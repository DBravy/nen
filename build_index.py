#!/usr/bin/env python
"""Build the browsable pages for the J-lens readouts.

Reads readouts/index.jsonl (written by collect_readouts.py) and emits:
  * readouts/index.html  -- sidebar of every readout (searchable, filterable by
    kind and run-label, grouped by prompt so variants sit together) beside an
    embedded viewer. Built for small laptop screens: collapsible/resizable
    sidebar that becomes an overlay drawer on narrow windows.
  * readouts/attn.html   -- attention "lookback" viewer: for readouts collected
    with --attention, pick a key token and see which later tokens attend back to
    it, and at which layer (head-averaged summary).

Variant convention: name related prompts `base__variant` (e.g. `gen_secret_sport`,
`gen_secret_sport__cot_high`); the index clusters everything sharing a `base`.

Usage:
  python build_index.py                 # reads/writes ./readouts next to this script
  python build_index.py --out ~/readouts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "readouts"),
                   help="readouts dir containing index.jsonl and pages/ "
                        "(default: ./readouts next to this script)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out).expanduser()
    index_path = out / "index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"no index.jsonl under {out}/ -- run collect_readouts.py first")

    records = []
    with index_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("html"):
                continue  # page rendering was skipped for this prompt
            records.append({
                "pid": r["pid"],
                "kind": r.get("kind", "raw"),
                "payload": r.get("payload", ""),
                "html": r["html"],
                "n_tokens": r.get("n_tokens"),
                "n_layers": len(r.get("layers", []) or []),
                "roundtrip_ok": r.get("roundtrip_ok", True),
                "has_attn": bool(r.get("attn_bin")),
                # provenance (blank on records written before these fields existed)
                "run_label": r.get("run_label") or "",
                "reasoning": r.get("reasoning") or "",
                "model": r.get("model") or "",
                "lens_file": r.get("lens_file") or "",
                "ts": r.get("ts") or "",
            })

    data_json = json.dumps(records, ensure_ascii=False)
    html = TEMPLATE.replace("__DATA__", data_json).replace("__COUNT__", str(len(records)))
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "attn.html").write_text(ATTN_TEMPLATE, encoding="utf-8")
    n_attn = sum(r["has_attn"] for r in records)
    print(f"[done] wrote {out/'index.html'} with {len(records)} readouts "
          f"({n_attn} with attention)")
    print(f"       serve it with:  cd {out} && python -m http.server 8123")
    print("       then browse to  http://localhost:8123/")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Readouts &mdash; index</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --line:#2a2f3a;
          --fg:#e6e8ec; --muted:#9aa3b2; --accent:#6ea8fe; --accent2:#2b3550;
          --sbw:300px; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; }
  body { font:13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         color:var(--fg); background:var(--bg); display:flex; height:100vh; overflow:hidden; }
  #sidebar { width:var(--sbw); flex:0 0 auto; background:var(--panel);
             border-right:1px solid var(--line); display:flex; flex-direction:column; min-width:0; }
  body.collapsed #sidebar, body.collapsed #resizer { display:none; }
  #resizer { width:6px; flex:0 0 auto; cursor:col-resize; background:transparent; }
  #resizer:hover, #resizer.drag { background:var(--accent); }
  #head { padding:11px 12px 8px; border-bottom:1px solid var(--line); }
  #head h1 { margin:0 0 2px; font-size:14px; font-weight:650; letter-spacing:.2px; }
  #head .sub { color:var(--muted); font-size:11px; }
  #controls { padding:9px 10px; border-bottom:1px solid var(--line); display:flex; flex-direction:column; gap:7px; }
  #search { width:100%; padding:7px 9px; border-radius:8px; border:1px solid var(--line);
            background:var(--panel2); color:var(--fg); font-size:13px; outline:none; }
  #search:focus { border-color:var(--accent); }
  .filters { display:flex; gap:5px; flex-wrap:wrap; align-items:center; }
  .filters .lbl { color:var(--muted); font-size:10px; margin-right:2px; }
  .chip { padding:2px 9px; border-radius:999px; border:1px solid var(--line);
          background:var(--panel2); color:var(--muted); cursor:pointer; font-size:11px; user-select:none; }
  .chip.active { background:var(--accent2); color:var(--fg); border-color:var(--accent); }
  #list { overflow-y:auto; flex:1 1 auto; padding:5px; }
  .group { display:flex; align-items:center; gap:8px; padding:9px 8px 3px; color:var(--muted);
           font-size:10px; text-transform:uppercase; letter-spacing:.6px; }
  .group::after { content:""; flex:1 1 auto; height:1px; background:var(--line); }
  .item { padding:8px 9px; border-radius:8px; cursor:pointer; border:1px solid transparent; margin-bottom:2px; }
  .item.grouped { margin-left:8px; border-left:2px solid var(--line); border-radius:0 8px 8px 0; }
  .item:hover { background:var(--panel2); }
  .item.active { background:var(--accent2); border-color:var(--accent); }
  .item .row1 { display:flex; align-items:center; gap:6px; margin-bottom:2px; flex-wrap:wrap; }
  .item .pid { font-weight:600; font-size:12px; }
  .badge { font-size:9px; padding:1px 6px; border-radius:6px; background:#243; color:#7fd6a3; text-transform:uppercase; letter-spacing:.4px; }
  .badge.chat { background:#2a2440; color:#c3a6ff; }
  .badge.chat_gen { background:#40282a; color:#ff9db0; }
  .badge.raw_gen { background:#2a3a24; color:#a9e08a; }
  .badge.warn { background:#4a3a1e; color:#ffcf7a; }
  .badge.attn { background:#1e3340; color:#7fc7e0; }
  .tag { font-size:9px; padding:1px 6px; border-radius:6px; background:var(--panel2);
         border:1px solid var(--line); color:var(--muted); }
  .item .meta { color:var(--muted); font-size:10px; }
  .item .prompt { color:var(--muted); font-size:11px; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #main { flex:1 1 auto; display:flex; flex-direction:column; min-width:0; }
  #bar { padding:6px 10px; border-bottom:1px solid var(--line); background:var(--panel);
         display:flex; align-items:center; gap:9px; min-height:42px; }
  .iconbtn { flex:0 0 auto; width:30px; height:30px; display:flex; align-items:center; justify-content:center;
             border:1px solid var(--line); background:var(--panel2); color:var(--fg); border-radius:7px;
             cursor:pointer; font-size:15px; line-height:1; user-select:none; }
  .iconbtn:hover { border-color:var(--accent); }
  #barleft { display:flex; flex-direction:column; gap:1px; min-width:0; flex:0 1 auto; max-width:40%; }
  #barTitle { font-weight:650; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #barMeta { color:var(--muted); font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #barPrompt { color:var(--muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1 1 auto; }
  .barlink { color:var(--accent); text-decoration:none; font-size:11px; border:1px solid var(--line);
             padding:5px 9px; border-radius:7px; white-space:nowrap; flex:0 0 auto; }
  .barlink:hover { border-color:var(--accent); }
  #frameWrap { flex:1 1 auto; position:relative; background:#fff; }
  iframe { border:0; width:100%; height:100%; background:#fff; }
  #empty { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
           color:var(--muted); background:var(--bg); text-align:center; padding:30px; }
  .kbd { font-size:10px; color:var(--muted); padding:6px 10px; border-top:1px solid var(--line); }
  .kbd b { color:var(--fg); font-weight:600; }
  @media (max-width:1024px) {
    #sidebar { position:absolute; top:0; left:0; height:100%; z-index:30;
               box-shadow:0 0 40px rgba(0,0,0,.6); }
    #resizer { display:none; }
    #barPrompt { display:none; }
  }
</style>
</head><body>
  <div id="sidebar">
    <div id="head">
      <h1>J-lens readouts</h1>
      <div class="sub"><span id="count">__COUNT__</span> readouts &middot; gpt-oss-20b</div>
    </div>
    <div id="controls">
      <input id="search" type="search" placeholder="Search prompt or id&hellip;" autocomplete="off">
      <div class="filters" id="kindFilters">
        <span class="lbl">kind</span>
        <span class="chip active" data-kind="all">all</span>
        <span class="chip" data-kind="raw">raw</span>
        <span class="chip" data-kind="raw_gen">raw_gen</span>
        <span class="chip" data-kind="chat">chat</span>
        <span class="chip" data-kind="chat_gen">chat_gen</span>
      </div>
      <div class="filters" id="runFilters" style="display:none"><span class="lbl">run</span></div>
    </div>
    <div id="list"></div>
    <div class="kbd"><b>b</b> toggle list &middot; <b>&uarr;/&darr;</b> or <b>j/k</b> move &middot; <b>/</b> search</div>
  </div>
  <div id="resizer" title="Drag to resize"></div>
  <div id="main">
    <div id="bar">
      <div class="iconbtn" id="navToggle" title="Toggle list (b)">&#9776;</div>
      <div id="barleft">
        <span id="barTitle">&mdash;</span>
        <span id="barMeta"></span>
      </div>
      <span id="barPrompt"></span>
      <a class="barlink" id="barAttn" href="#" style="display:none">attention &#8599;</a>
      <a class="barlink" id="barOpen" href="#" target="_blank" rel="noopener">open &#8599;</a>
    </div>
    <div id="frameWrap">
      <iframe id="frame" title="readout"></iframe>
      <div id="empty">Select a readout on the left to view it here.</div>
    </div>
  </div>
<script>
const RECORDS = __DATA__;
RECORDS.forEach((r, i) => { r._i = i; });
const listEl = document.getElementById('list');
const searchEl = document.getElementById('search');
const frame = document.getElementById('frame');
const empty = document.getElementById('empty');
const barTitle = document.getElementById('barTitle');
const barMeta = document.getElementById('barMeta');
const barPrompt = document.getElementById('barPrompt');
const barOpen = document.getElementById('barOpen');
const barAttn = document.getElementById('barAttn');
let kind = 'all';
let runLabel = 'all';
let query = '';
let activePid = null;

const baseOf = pid => pid.split('__')[0];
const variantOf = pid => { const i = pid.indexOf('__'); return i < 0 ? '' : pid.slice(i + 2); };
const isOverlay = () => window.matchMedia('(max-width:1024px)').matches;

/* ---- collapsible / resizable sidebar (tuned for small laptop screens) ---- */
function setCollapsed(v) {
  document.body.classList.toggle('collapsed', v);
  try { localStorage.setItem('nav-collapsed', v ? '1' : '0'); } catch (e) {}
}
function toggleNav() { setCollapsed(!document.body.classList.contains('collapsed')); }
document.getElementById('navToggle').addEventListener('click', toggleNav);

(function initNav() {
  let stored = null;
  try { stored = localStorage.getItem('nav-collapsed'); } catch (e) {}
  if (stored === null) setCollapsed(isOverlay());
  else setCollapsed(stored === '1');
  let w = null;
  try { w = localStorage.getItem('sbw'); } catch (e) {}
  if (w) document.body.style.setProperty('--sbw', w + 'px');
})();

(function initResizer() {
  const rez = document.getElementById('resizer');
  let dragging = false;
  const onMove = e => {
    if (!dragging) return;
    const w = Math.max(220, Math.min(560, e.clientX));
    document.body.style.setProperty('--sbw', w + 'px');
  };
  const stop = () => {
    if (!dragging) return;
    dragging = false;
    rez.classList.remove('drag');
    document.body.style.userSelect = '';
    const w = parseInt(getComputedStyle(document.body).getPropertyValue('--sbw'));
    try { localStorage.setItem('sbw', w); } catch (e) {}
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseup', stop);
  };
  rez.addEventListener('mousedown', e => {
    dragging = true; rez.classList.add('drag'); document.body.style.userSelect = 'none';
    e.preventDefault();
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', stop);
  });
})();

const baseFirst = {};
RECORDS.forEach(r => { const b = baseOf(r.pid); if (baseFirst[b] === undefined) baseFirst[b] = r._i; });

const labels = [...new Set(RECORDS.map(r => r.run_label).filter(Boolean))].sort();
const hasUnlabeled = RECORDS.some(r => !r.run_label);
if (labels.length) {
  const row = document.getElementById('runFilters');
  row.style.display = 'flex';
  const chips = ['all', ...labels];
  if (hasUnlabeled) chips.push('unlabeled');
  for (const l of chips) {
    const c = document.createElement('span');
    c.className = 'chip' + (l === 'all' ? ' active' : '');
    c.dataset.run = l;
    c.textContent = l;
    c.addEventListener('click', () => {
      row.querySelectorAll('.chip').forEach(x => x.classList.remove('active'));
      c.classList.add('active');
      runLabel = l;
      render();
    });
    row.appendChild(c);
  }
}

function visible() {
  const rows = RECORDS.filter(r => {
    if (kind !== 'all' && r.kind !== kind) return false;
    if (runLabel === 'unlabeled' && r.run_label) return false;
    if (runLabel !== 'all' && runLabel !== 'unlabeled' && r.run_label !== runLabel) return false;
    if (query) {
      const hay = (r.pid + ' ' + r.payload).toLowerCase();
      if (!hay.includes(query)) return false;
    }
    return true;
  });
  rows.sort((a, b) => {
    const d = baseFirst[baseOf(a.pid)] - baseFirst[baseOf(b.pid)];
    return d !== 0 ? d : a._i - b._i;
  });
  return rows;
}

function render() {
  const rows = visible();
  const groupCount = {};
  rows.forEach(r => { const b = baseOf(r.pid); groupCount[b] = (groupCount[b] || 0) + 1; });
  listEl.innerHTML = '';
  if (!rows.length) {
    listEl.innerHTML = '<div style="padding:14px;color:var(--muted)">No matches.</div>';
    return;
  }
  let lastBase = null;
  for (const r of rows) {
    const base = baseOf(r.pid);
    const grouped = groupCount[base] > 1;
    if (grouped && base !== lastBase) {
      const h = document.createElement('div');
      h.className = 'group';
      h.textContent = base + ' · ' + groupCount[base];
      listEl.appendChild(h);
    }
    if (base !== lastBase) lastBase = base;

    const el = document.createElement('div');
    el.className = 'item' + (grouped ? ' grouped' : '') + (r.pid === activePid ? ' active' : '');
    el.dataset.pid = r.pid;
    const warn = r.roundtrip_ok ? '' : '<span class="badge warn" title="tokenizer roundtrip drifted; positions may be off by a token">drift</span>';
    const attn = r.has_attn ? '<span class="badge attn" title="has attention data">attn</span>' : '';
    const labelTag = r.run_label ? '<span class="tag"></span>' : '';
    const reasonTag = r.reasoning ? '<span class="tag">r=' + r.reasoning + '</span>' : '';
    el.innerHTML =
      '<div class="row1"><span class="pid"></span>' +
      '<span class="badge ' + r.kind + '">' + r.kind + '</span>' + warn + attn + labelTag + reasonTag + '</div>' +
      '<div class="meta">' + (r.n_tokens ?? '?') + ' tok &middot; ' + (r.n_layers ?? '?') + ' layers</div>' +
      '<div class="prompt"></div>';
    el.querySelector('.pid').textContent = grouped ? (variantOf(r.pid) || '(base)') : r.pid;
    if (r.run_label) el.querySelector('.tag').textContent = r.run_label;
    el.querySelector('.prompt').textContent = r.payload.replace(/\n/g, ' ↩ ');
    el.addEventListener('click', () => select(r.pid));
    listEl.appendChild(el);
  }
}

function select(pid) {
  const r = RECORDS.find(x => x.pid === pid);
  if (!r) return;
  activePid = pid;
  empty.style.display = 'none';
  frame.src = r.html;
  barTitle.textContent = r.pid;
  const bits = [r.kind];
  if (r.reasoning) bits.push('reasoning=' + r.reasoning);
  if (r.run_label) bits.push(r.run_label);
  if (r.model) bits.push(r.model);
  if (r.ts) bits.push(r.ts);
  barMeta.textContent = bits.join(' · ');
  barPrompt.textContent = r.payload.replace(/\n/g, ' ↩ ');
  barPrompt.title = r.payload;
  barOpen.href = r.html;
  if (r.has_attn) { barAttn.style.display = ''; barAttn.href = 'attn.html?pid=' + encodeURIComponent(pid); }
  else barAttn.style.display = 'none';
  render();
  const active = listEl.querySelector('.item.active');
  if (active) active.scrollIntoView({ block: 'nearest' });
  history.replaceState(null, '', '#' + encodeURIComponent(pid));
  if (isOverlay()) setCollapsed(true);
}

function move(delta) {
  const rows = visible();
  if (!rows.length) return;
  let i = rows.findIndex(r => r.pid === activePid);
  i = (i + delta + rows.length) % rows.length;
  if (i < 0) i = 0;
  select(rows[i].pid);
}

document.querySelectorAll('#kindFilters .chip').forEach(chip => {
  chip.addEventListener('click', () => {
    document.querySelectorAll('#kindFilters .chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    kind = chip.dataset.kind;
    render();
  });
});
searchEl.addEventListener('input', () => { query = searchEl.value.trim().toLowerCase(); render(); });

document.addEventListener('keydown', e => {
  if (e.target === searchEl) { if (e.key === 'Escape') searchEl.blur(); return; }
  if (e.key === '/') { e.preventDefault(); searchEl.focus(); return; }
  if (e.key === 'b') { e.preventDefault(); toggleNav(); return; }
  if (e.key === 'ArrowDown' || e.key === 'j') { e.preventDefault(); move(1); }
  if (e.key === 'ArrowUp'   || e.key === 'k') { e.preventDefault(); move(-1); }
});

render();
const hash = decodeURIComponent(location.hash.replace(/^#/, ''));
if (hash && RECORDS.some(r => r.pid === hash)) select(hash);
else if (RECORDS.length) select(RECORDS[0].pid);
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# Attention "lookback" viewer. Fully static: it reads index.jsonl and the
# attn/<pid>.u8 binaries at runtime (served over http.server), so it needs no
# per-build data injection.
# --------------------------------------------------------------------------- #
ATTN_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Attention lookback</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --line:#2a2f3a;
          --fg:#e6e8ec; --muted:#9aa3b2; --accent:#6ea8fe; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; }
  body { font:13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
         color:var(--fg); background:var(--bg); display:flex; flex-direction:column; height:100vh; overflow:hidden; }
  #bar { display:flex; align-items:center; gap:10px; padding:8px 12px; border-bottom:1px solid var(--line);
         background:var(--panel); flex-wrap:wrap; }
  #bar h1 { font-size:14px; margin:0 8px 0 0; font-weight:650; }
  a.back { color:var(--accent); text-decoration:none; font-size:12px; }
  select, button { background:var(--panel2); color:var(--fg); border:1px solid var(--line);
                   border-radius:7px; padding:5px 8px; font-size:12px; }
  button { cursor:pointer; }
  button.on { border-color:var(--accent); color:var(--fg); background:#243; }
  .seg { display:flex; gap:0; }
  .seg button { border-radius:0; }
  .seg button:first-child { border-radius:7px 0 0 7px; }
  .seg button:last-child { border-radius:0 7px 7px 0; border-left:0; }
  .lbl { color:var(--muted); font-size:11px; }
  #wrap { flex:1 1 auto; overflow:auto; padding:10px 12px; }
  #info { color:var(--muted); font-size:12px; margin-bottom:8px; }
  #info b { color:var(--fg); }
  .keytok { color:var(--accent); font-weight:600; }
  #canvasHolder { position:relative; display:inline-block; }
  canvas { display:block; image-rendering:pixelated; }
  #tip { position:fixed; pointer-events:none; background:#000; border:1px solid var(--line);
         border-radius:6px; padding:5px 8px; font-size:11px; z-index:50; display:none; max-width:340px; }
  #tip .w { color:var(--accent); font-weight:600; }
  #ylabels, #xnote { color:var(--muted); font-size:10px; }
  .row { display:flex; align-items:flex-start; gap:8px; }
  .empty { color:var(--muted); padding:40px; text-align:center; }
</style>
</head><body>
  <div id="bar">
    <h1>Attention lookback</h1>
    <a class="back" href="index.html">&larr; index</a>
    <span class="lbl">readout</span>
    <select id="pidSel"></select>
    <span class="lbl">summary</span>
    <div class="seg" id="whichSeg">
      <button data-w="0" class="on">mean</button>
      <button data-w="1">max</button>
    </div>
    <span class="lbl">view</span>
    <div class="seg" id="modeSeg">
      <button data-m="lookback" class="on">lookback</button>
      <button data-m="matrix">layer matrix</button>
    </div>
    <span id="layerCtl" style="display:none">
      <span class="lbl">layer</span>
      <input type="range" id="layer" min="0" max="0" value="0" style="vertical-align:middle">
      <span id="layerVal" class="lbl">0</span>
    </span>
    <span class="lbl">attend&nbsp;back&nbsp;to</span>
    <select id="keySel"></select>
  </div>
  <div id="wrap">
    <div id="info"></div>
    <div class="row">
      <div id="ylabels"></div>
      <div>
        <div id="canvasHolder"><canvas id="cv"></canvas></div>
        <div id="xnote"></div>
      </div>
    </div>
  </div>
  <div id="tip"></div>
<script>
const qs = new URLSearchParams(location.search);
let records = [], rec = null, data = null;
let W = 2, L = 0, T = 0;
let which = 0;            // 0 mean, 1 max
let mode = 'lookback';    // 'lookback' | 'matrix'
let keyPos = 0;           // key token index to attend back to
let matrixLayer = 0;

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const tip = document.getElementById('tip');
const pidSel = document.getElementById('pidSel');
const keySel = document.getElementById('keySel');
const info = document.getElementById('info');
const ylabels = document.getElementById('ylabels');
const xnote = document.getElementById('xnote');
const layerCtl = document.getElementById('layerCtl');
const layerInput = document.getElementById('layer');
const layerVal = document.getElementById('layerVal');

const at = (w, l, qy, k) => data[((w * L + l) * T + qy) * T + k];  // C-order [W,L,T,T]
const tok = i => (rec.token_strs && rec.token_strs[i] != null) ? rec.token_strs[i] : ('#' + i);
const shortTok = i => { let s = tok(i).replace(/\n/g, '\\n'); return s.length > 14 ? s.slice(0, 13) + '…' : s; };

function color(t) {  // t in [0,1] -> dark navy to bright cyan
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(10 + t * 60), g = Math.round(20 + t * 200), b = Math.round(35 + t * 210);
  return [r, g, b];
}

async function loadIndex() {
  let txt;
  try { txt = await fetch('index.jsonl').then(r => r.text()); }
  catch (e) { info.innerHTML = '<span class="empty">Could not load index.jsonl (serve the folder over http).</span>'; return; }
  records = txt.split('\n').filter(Boolean).map(JSON.parse).filter(r => r.attn_bin);
  if (!records.length) { info.innerHTML = '<span class="empty">No readouts have attention data yet. Re-run collect_readouts.py with <b>--attention</b>.</span>'; return; }
  pidSel.innerHTML = '';
  for (const r of records) {
    const o = document.createElement('option');
    o.value = r.pid; o.textContent = r.pid;
    pidSel.appendChild(o);
  }
  const want = qs.get('pid');
  const start = records.find(r => r.pid === want) ? want : records[0].pid;
  pidSel.value = start;
  await loadPid(start);
}

async function loadPid(pid) {
  rec = records.find(r => r.pid === pid);
  if (!rec) return;
  const sh = rec.attn_shape;            // [W, L, T, T]
  W = sh[0]; L = sh[1]; T = sh[2];
  const buf = await fetch(rec.attn_bin).then(r => r.arrayBuffer());
  data = new Uint8Array(buf);
  keyPos = Math.max(0, Math.floor(T * 0.4));  // a content token with tokens after it
  matrixLayer = Math.floor(L / 2);
  layerInput.max = String(L - 1);
  layerInput.value = String(matrixLayer);
  layerVal.textContent = matrixLayer;
  keySel.innerHTML = '';
  for (let i = 0; i < T; i++) {
    const o = document.createElement('option');
    o.value = String(i); o.textContent = i + ': ' + shortTok(i);
    keySel.appendChild(o);
  }
  keySel.value = String(keyPos);
  history.replaceState(null, '', 'attn.html?pid=' + encodeURIComponent(pid));
  draw();
}

function draw() {
  if (!data) return;
  if (mode === 'lookback') drawLookback(); else drawMatrix();
}

// Rows = layers, cols = query positions; cell = attention from query qy to fixed key.
function drawLookback() {
  layerCtl.style.display = 'none';
  const cw = Math.max(3, Math.min(16, Math.floor(1200 / T)));
  const ch = 16;
  cv.width = T * cw; cv.height = L * ch;
  const img = ctx.createImageData(cv.width, cv.height);
  // per-view max for contrast (only causal cells qy>=key can be nonzero)
  let vmax = 1;  // exclude the self-attention diagonal so it doesn't wash out lookback
  for (let l = 0; l < L; l++) for (let qy = keyPos + 1; qy < T; qy++) vmax = Math.max(vmax, at(which, l, qy, keyPos));
  for (let l = 0; l < L; l++) {
    for (let qy = 0; qy < T; qy++) {
      const v = qy >= keyPos ? at(which, l, qy, keyPos) / vmax : 0;
      const [r, g, b] = color(v);
      for (let dy = 0; dy < ch; dy++) for (let dx = 0; dx < cw; dx++) {
        const px = ((l * ch + dy) * cv.width + (qy * cw + dx)) * 4;
        img.data[px] = r; img.data[px+1] = g; img.data[px+2] = b; img.data[px+3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  cv.dataset.cw = cw; cv.dataset.ch = ch;
  ylabels.innerHTML = Array.from({length: L}, (_, l) =>
    '<div style="height:' + ch + 'px;line-height:' + ch + 'px">L' + l + '</div>').join('');
  info.innerHTML = 'Attention flowing <b>back into</b> key <span class="keytok">' + keyPos + ': ' +
    escapeHtml(tok(keyPos)) + '</span> &mdash; rows are layers (0 = first), columns are query tokens ' +
    '(later tokens to the right). Bright = that later token, at that layer, looked back here. ' +
    'Head-averaged (' + (which ? 'max' : 'mean') + ' over heads).';
  xnote.textContent = 'x-axis: query token 0…' + (T - 1) + ' (hover for token & weight). Only tokens at/after the key can attend to it.';
}

// Full [query x key] matrix at one layer.
function drawMatrix() {
  layerCtl.style.display = '';
  const cell = Math.max(2, Math.min(10, Math.floor(1200 / T)));
  cv.width = T * cell; cv.height = T * cell;
  const img = ctx.createImageData(cv.width, cv.height);
  let vmax = 1;  // exclude the diagonal (k==qy) from the contrast scale
  for (let qy = 0; qy < T; qy++) for (let k = 0; k < qy; k++) vmax = Math.max(vmax, at(which, matrixLayer, qy, k));
  for (let qy = 0; qy < T; qy++) {
    for (let k = 0; k < T; k++) {
      const v = k <= qy ? at(which, matrixLayer, qy, k) / vmax : 0;
      const [r, g, b] = color(v);
      for (let dy = 0; dy < cell; dy++) for (let dx = 0; dx < cell; dx++) {
        const px = ((qy * cell + dy) * cv.width + (k * cell + dx)) * 4;
        img.data[px] = r; img.data[px+1] = g; img.data[px+2] = b; img.data[px+3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  cv.dataset.cell = cell;
  ylabels.innerHTML = '';
  info.innerHTML = 'Full attention at <b>layer ' + matrixLayer + '</b> &mdash; rows are query tokens ' +
    '(who is attending), columns are key tokens (attended to). Click a column to pin it as the ' +
    'lookback key. Head-averaged (' + (which ? 'max' : 'mean') + ' over heads).';
  xnote.textContent = 'rows = query 0…' + (T - 1) + ' (top→down), cols = key 0…' + (T - 1) + '. Hover for details.';
}

function escapeHtml(s) { return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

cv.addEventListener('mousemove', e => {
  if (!data) return;
  const rect = cv.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  if (mode === 'lookback') {
    const cw = +cv.dataset.cw, ch = +cv.dataset.ch;
    const qy = Math.floor(x / cw), l = Math.floor(y / ch);
    if (qy < 0 || qy >= T || l < 0 || l >= L) { tip.style.display = 'none'; return; }
    const raw = qy >= keyPos ? at(which, l, qy, keyPos) : 0;
    const val = (raw / 255 * rec.attn_scale).toFixed(3);
    tip.innerHTML = 'L' + l + ' &middot; query <b>' + qy + ': ' + escapeHtml(tok(qy)) + '</b><br>' +
      '&rarr; key <b>' + keyPos + ': ' + escapeHtml(tok(keyPos)) + '</b><br>weight <span class="w">' + val + '</span>';
    showTip(e);
  } else {
    const cell = +cv.dataset.cell;
    const qy = Math.floor(y / cell), k = Math.floor(x / cell);
    if (qy < 0 || qy >= T || k < 0 || k >= T) { tip.style.display = 'none'; return; }
    const raw = k <= qy ? at(which, matrixLayer, qy, k) : 0;
    const val = (raw / 255 * rec.attn_scale).toFixed(3);
    tip.innerHTML = 'query <b>' + qy + ': ' + escapeHtml(tok(qy)) + '</b><br>&rarr; key <b>' + k + ': ' +
      escapeHtml(tok(k)) + '</b><br>L' + matrixLayer + ' weight <span class="w">' + val + '</span>';
    showTip(e);
  }
});
cv.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
cv.addEventListener('click', e => {
  if (mode !== 'matrix') return;
  const cell = +cv.dataset.cell;
  const k = Math.floor((e.clientX - cv.getBoundingClientRect().left) / cell);
  if (k >= 0 && k < T) { keyPos = k; keySel.value = String(k); setMode('lookback'); }
});
function showTip(e) {
  tip.style.display = 'block';
  tip.style.left = (e.clientX + 14) + 'px';
  tip.style.top = (e.clientY + 14) + 'px';
}

function setMode(m) {
  mode = m;
  document.querySelectorAll('#modeSeg button').forEach(b => b.classList.toggle('on', b.dataset.m === m));
  draw();
}
document.querySelectorAll('#whichSeg button').forEach(b => b.addEventListener('click', () => {
  which = +b.dataset.w;
  document.querySelectorAll('#whichSeg button').forEach(x => x.classList.toggle('on', x === b));
  draw();
}));
document.querySelectorAll('#modeSeg button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.m)));
layerInput.addEventListener('input', () => { matrixLayer = +layerInput.value; layerVal.textContent = matrixLayer; draw(); });
keySel.addEventListener('change', () => { keyPos = +keySel.value; if (mode !== 'lookback') setMode('lookback'); else draw(); });
pidSel.addEventListener('change', () => loadPid(pidSel.value));

loadIndex();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
