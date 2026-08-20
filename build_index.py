#!/usr/bin/env python
"""Build a single browsable index page for the J-lens readouts.

Reads readouts/index.jsonl (written by collect_readouts.py) and emits
readouts/index.html: a sidebar of every readout (searchable, filterable by
kind and by run-label, and grouped by prompt so variants sit together) beside
an embedded viewer, so you can flip through all pages from one browser tab.

Variant convention: name related prompts `base__variant` (e.g.
`gen_secret_sport`, `gen_secret_sport__cot_high`). The index clusters everything
sharing a `base` under one group header.

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
    print(f"[done] wrote {out/'index.html'} with {len(records)} readouts")
    print(f"       serve it with:  cd {out} && python -m http.server 8123")
    print("       then browse to  http://localhost:8123/")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Readouts &mdash; index</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --line:#2a2f3a;
          --fg:#e6e8ec; --muted:#9aa3b2; --accent:#6ea8fe; --accent2:#2b3550; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; }
  body { font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         color:var(--fg); background:var(--bg); display:flex; height:100vh; overflow:hidden; }
  #sidebar { width:352px; min-width:260px; max-width:560px; flex:0 0 auto;
             background:var(--panel); border-right:1px solid var(--line);
             display:flex; flex-direction:column; }
  #head { padding:14px 14px 10px; border-bottom:1px solid var(--line); }
  #head h1 { margin:0 0 2px; font-size:15px; font-weight:650; letter-spacing:.2px; }
  #head .sub { color:var(--muted); font-size:12px; }
  #controls { padding:10px 12px; border-bottom:1px solid var(--line); display:flex; flex-direction:column; gap:8px; }
  #search { width:100%; padding:8px 10px; border-radius:8px; border:1px solid var(--line);
            background:var(--panel2); color:var(--fg); font-size:13px; outline:none; }
  #search:focus { border-color:var(--accent); }
  .filters { display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
  .filters .lbl { color:var(--muted); font-size:11px; margin-right:2px; }
  .chip { padding:3px 10px; border-radius:999px; border:1px solid var(--line);
          background:var(--panel2); color:var(--muted); cursor:pointer; font-size:12px; user-select:none; }
  .chip.active { background:var(--accent2); color:var(--fg); border-color:var(--accent); }
  #list { overflow-y:auto; flex:1 1 auto; padding:6px; }
  .group { display:flex; align-items:center; gap:8px; padding:10px 8px 4px; color:var(--muted);
           font-size:11px; text-transform:uppercase; letter-spacing:.6px; }
  .group::after { content:""; flex:1 1 auto; height:1px; background:var(--line); }
  .item { padding:9px 10px; border-radius:8px; cursor:pointer; border:1px solid transparent; margin-bottom:2px; }
  .item.grouped { margin-left:8px; border-left:2px solid var(--line); border-radius:0 8px 8px 0; }
  .item:hover { background:var(--panel2); }
  .item.active { background:var(--accent2); border-color:var(--accent); }
  .item .row1 { display:flex; align-items:center; gap:7px; margin-bottom:3px; flex-wrap:wrap; }
  .item .pid { font-weight:600; font-size:13px; }
  .badge { font-size:10px; padding:1px 6px; border-radius:6px; background:#243; color:#7fd6a3; text-transform:uppercase; letter-spacing:.4px; }
  .badge.chat { background:#2a2440; color:#c3a6ff; }
  .badge.chat_gen { background:#40282a; color:#ff9db0; }
  .badge.warn { background:#4a3a1e; color:#ffcf7a; }
  .tag { font-size:10px; padding:1px 6px; border-radius:6px; background:var(--panel2);
         border:1px solid var(--line); color:var(--muted); }
  .item .meta { color:var(--muted); font-size:11px; }
  .item .prompt { color:var(--muted); font-size:12px; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #main { flex:1 1 auto; display:flex; flex-direction:column; min-width:0; }
  #bar { padding:8px 14px; border-bottom:1px solid var(--line); background:var(--panel);
         display:flex; align-items:center; gap:12px; min-height:46px; }
  #barleft { display:flex; flex-direction:column; gap:1px; min-width:0; flex:0 0 auto; max-width:38%; }
  #barTitle { font-weight:650; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #barMeta { color:var(--muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #barPrompt { color:var(--muted); font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1 1 auto; }
  #barOpen { color:var(--accent); text-decoration:none; font-size:12px; border:1px solid var(--line);
             padding:5px 10px; border-radius:7px; white-space:nowrap; flex:0 0 auto; }
  #barOpen:hover { border-color:var(--accent); }
  #frameWrap { flex:1 1 auto; position:relative; background:#fff; }
  iframe { border:0; width:100%; height:100%; background:#fff; }
  #empty { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
           color:var(--muted); background:var(--bg); text-align:center; padding:30px; }
  .kbd { font-size:11px; color:var(--muted); padding:6px 12px; border-top:1px solid var(--line); }
  .kbd b { color:var(--fg); font-weight:600; }
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
        <span class="chip" data-kind="chat">chat</span>
        <span class="chip" data-kind="chat_gen">chat_gen</span>
      </div>
      <div class="filters" id="runFilters" style="display:none"><span class="lbl">run</span></div>
    </div>
    <div id="list"></div>
    <div class="kbd"><b>&uarr;/&darr;</b> or <b>j/k</b> to move &middot; <b>/</b> to search</div>
  </div>
  <div id="main">
    <div id="bar">
      <div id="barleft">
        <span id="barTitle">&mdash;</span>
        <span id="barMeta"></span>
      </div>
      <span id="barPrompt"></span>
      <a id="barOpen" href="#" target="_blank" rel="noopener">open in new tab &#8599;</a>
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
let kind = 'all';
let runLabel = 'all';
let query = '';
let activePid = null;

const baseOf = pid => pid.split('__')[0];
const variantOf = pid => { const i = pid.indexOf('__'); return i < 0 ? '' : pid.slice(i + 2); };

// Order groups by first appearance in the dataset (preserves battery order),
// keep variants of a base adjacent and in dataset order.
const baseFirst = {};
RECORDS.forEach(r => { const b = baseOf(r.pid); if (baseFirst[b] === undefined) baseFirst[b] = r._i; });

// Build the run-label filter from whatever labels are present.
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
    const labelTag = r.run_label ? '<span class="tag"></span>' : '';
    const reasonTag = r.reasoning ? '<span class="tag">r=' + r.reasoning + '</span>' : '';
    el.innerHTML =
      '<div class="row1"><span class="pid"></span>' +
      '<span class="badge ' + r.kind + '">' + r.kind + '</span>' + warn + labelTag + reasonTag + '</div>' +
      '<div class="meta">' + (r.n_tokens ?? '?') + ' tok &middot; ' + (r.n_layers ?? '?') + ' layers</div>' +
      '<div class="prompt"></div>';
    // Under a group header, show just the variant suffix; otherwise the full pid.
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
  render();
  const active = listEl.querySelector('.item.active');
  if (active) active.scrollIntoView({ block: 'nearest' });
  history.replaceState(null, '', '#' + encodeURIComponent(pid));
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


if __name__ == "__main__":
    main()
