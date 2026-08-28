"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  datasets: [],
  view: null,       // current rollout view
  pauseIndex: null, // selected token index
};

async function jget(url) {
  const r = await fetch(url);
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}
async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

// -- edit rows ------------------------------------------------------------- //
const MODES = {
  suppress: { label: "suppress (×0)", mult: 0, bias: 0 },
  amplify: { label: "amplify ×4", mult: 4, bias: 0 },
  flip: { label: "flip (×-1)", mult: -1, bias: 0 },
  push_pos: { label: "push +b·SD", mult: 1, bias: null },
  push_neg: { label: "push -b·SD", mult: 1, bias: null },
  custom: { label: "custom", mult: 1, bias: 0 },
};

function addEdit() {
  const row = document.createElement("div");
  row.className = "edit";
  row.innerHTML = `
    <label>Layer<input class="e-layer" type="number" min="0" max="22" value="17" /></label>
    <label>SV<input class="e-sv" type="number" min="1" max="64" value="1" /></label>
    <label>Mode
      <select class="e-mode">
        ${Object.entries(MODES).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join("")}
      </select>
    </label>
    <button class="del" title="remove">✕</button>
    <label class="full" style="display:none">Multiplier<input class="e-mult" type="number" step="0.5" value="0" /></label>
    <label class="full" style="display:none">Bias<input class="e-bias" type="number" step="0.5" value="0" /></label>
    <label class="full e-sdwrap" style="display:none">b (SD units)<input class="e-b" type="number" step="0.5" value="4" /></label>
    <div class="sd-note"></div>
  `;
  const mode = row.querySelector(".e-mode");
  const multL = row.querySelector(".e-mult").parentElement;
  const biasL = row.querySelector(".e-bias").parentElement;
  const sdL = row.querySelector(".e-sdwrap");
  const note = row.querySelector(".sd-note");

  function refresh() {
    const m = mode.value;
    const isCustom = m === "custom";
    const isPush = m === "push_pos" || m === "push_neg";
    multL.style.display = isCustom ? "flex" : "none";
    biasL.style.display = isCustom ? "flex" : "none";
    sdL.style.display = isPush ? "flex" : "none";
    updateSdNote(row);
  }
  mode.addEventListener("change", refresh);
  row.querySelectorAll(".e-layer, .e-sv, .e-b").forEach((el) =>
    el.addEventListener("input", () => updateSdNote(row))
  );
  row.querySelector(".del").addEventListener("click", () => row.remove());
  $("edits").appendChild(row);
  refresh();
}

const STD = {}; // "basis:layer:sv" -> std (cached from /api/sv_meta)
async function svStd(basis, layer, sv) {
  const key = `${basis}:${layer}:${sv}`;
  if (key in STD) return STD[key];
  try {
    const j = await jget(`/api/sv_meta?basis=${basis}&layer=${layer}&sv=${sv}`);
    STD[key] = j.std || null;
  } catch { STD[key] = null; }
  return STD[key];
}

async function updateSdNote(row) {
  const mode = row.querySelector(".e-mode").value;
  const note = row.querySelector(".sd-note");
  if (mode !== "push_pos" && mode !== "push_neg") { note.textContent = ""; return; }
  const layer = +row.querySelector(".e-layer").value;
  const sv = +row.querySelector(".e-sv").value;
  const b = +row.querySelector(".e-b").value;
  const std = await svStd($("basis").value, layer, sv);
  if (std == null) { note.textContent = "std unavailable — bias applied in raw units"; return; }
  const bias = (mode === "push_neg" ? -1 : 1) * b * std;
  note.textContent = `std=${std.toFixed(3)} → bias=${bias.toFixed(3)}`;
}

// Server resolves SD-unit bias; we send bias_sd (signed) for push modes.
function readEdit(row) {
  const layer = +row.querySelector(".e-layer").value;
  const sv_rank = +row.querySelector(".e-sv").value;
  const mode = row.querySelector(".e-mode").value;
  if (mode === "custom") {
    return { layer, sv_rank, multiplier: +row.querySelector(".e-mult").value, bias: +row.querySelector(".e-bias").value };
  }
  if (mode === "push_pos" || mode === "push_neg") {
    const b = +row.querySelector(".e-b").value;
    return { layer, sv_rank, multiplier: 1, bias_sd: (mode === "push_neg" ? -1 : 1) * b };
  }
  return { layer, sv_rank, multiplier: MODES[mode].mult, bias: MODES[mode].bias };
}

// -- rollout + token stream ------------------------------------------------ //
async function loadDatasets() {
  const j = await jget("/api/datasets");
  state.datasets = j.datasets;
  $("dataset").innerHTML = j.datasets
    .map((d) => `<option value="${d.key}">${d.key} (${d.rollouts.length})</option>`)
    .join("");
  populateRollouts();
}

function populateRollouts() {
  const d = state.datasets.find((x) => x.key === $("dataset").value);
  $("rollout").innerHTML = d.rollouts
    .map((r) => `<option value="${r.index}">#${r.index} · ${r.prompt_id} · ${r.category || ""}</option>`)
    .join("");
  loadRollout();
}

async function loadRollout() {
  const dataset = $("dataset").value;
  const index = +$("rollout").value;
  $("status").textContent = "loading rollout…";
  try {
    const v = await jget(`/api/rollout?dataset=${encodeURIComponent(dataset)}&index=${index}`);
    state.view = v;
    state.pauseIndex = null;
    renderStream();
    const d = state.datasets.find((x) => x.key === dataset).rollouts[index];
    $("rollout-meta").textContent =
      `${v.prompt_id} · ${d.difficulty || "?"} · effort ${d.reasoning_effort} · ${v.tokens.length} resp tokens`;
    const warn = $("mismatch-warn");
    if (v.prompt_mismatch) {
      warn.hidden = false;
      warn.textContent = `⚠ prompt token mismatch: rebuilt ${v.prompt_len} vs stored ${v.expected_prompt_tokens}. Pause positions may be off.`;
    } else {
      warn.hidden = true;
    }
    $("status").textContent = "";
  } catch (e) {
    $("status").textContent = e.message;
    $("status").className = "status err";
  }
}

function renderStream() {
  const stream = $("stream");
  stream.innerHTML = "";
  state.view.tokens.forEach((t) => {
    const span = document.createElement("span");
    let cls = "tok";
    if (t.channel === "marker") cls += " marker";
    else if (t.channel === "final") cls += " final";
    if (state.pauseIndex != null) {
      if (t.index === state.pauseIndex) cls += " paused";
      else if (t.index > state.pauseIndex) cls += " after";
    }
    span.className = cls;
    span.textContent = t.text.replace(/\n/g, "↵\n");
    span.title = `#${t.index} · id ${t.token_id} · ${t.channel || "?"}`;
    span.addEventListener("click", () => {
      state.pauseIndex = t.index;
      renderStream();
      $("pause-info").textContent = `pause at token #${t.index}: ${JSON.stringify(t.text)}`;
      $("run").disabled = false;
    });
    stream.appendChild(span);
  });
}

// -- run ------------------------------------------------------------------- //
async function run() {
  if (state.pauseIndex == null) return;
  const edits = [...document.querySelectorAll(".edit")].map(readEdit);
  const body = {
    dataset: $("dataset").value,
    rollout_index: +$("rollout").value,
    pause_index: state.pauseIndex,
    basis: $("basis").value,
    edits,
    n_samples: +$("n_samples").value,
    temperature: +$("temperature").value,
    top_p: +$("top_p").value,
    seed: +$("seed").value,
    max_new_tokens: +$("max_new_tokens").value,
  };
  $("run").disabled = true;
  $("status").className = "status";
  $("status").textContent = "generating… (this runs the 20B model on the VM)";
  try {
    const res = await jpost("/api/intervene", body);
    renderResults(res);
    $("status").textContent = `done · prefix ${res.prefix_len} tokens · pause token ${JSON.stringify(res.pause_token)}`;
  } catch (e) {
    $("status").textContent = e.message;
    $("status").className = "status err";
  } finally {
    $("run").disabled = false;
  }
}

function renderResults(res) {
  const tel = Object.entries(res.telemetry || {})
    .map(([k, v]) => `${k}: pre ${v.mean_pre.toFixed(3)} → post ${v.mean_post.toFixed(3)}  (n=${v.n_forward_vector_instances})`)
    .join("\n");
  $("telemetry").textContent = tel || (res.edits.length ? "" : "no edits (control comparison)");

  const sample = (t, i) => `<div class="sample"><span class="idx">#${i}</span> ${escapeHtml(t)}</div>`;
  $("col-original").innerHTML = `<div class="sample">${escapeHtml(res.original_continuation)}</div>`;
  $("col-baseline").innerHTML = res.baseline.map(sample).join("");
  $("col-intervened").innerHTML = res.intervened.map(sample).join("");
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

// -- wire up --------------------------------------------------------------- //
$("dataset").addEventListener("change", populateRollouts);
$("rollout").addEventListener("change", loadRollout);
$("basis").addEventListener("change", () =>
  document.querySelectorAll(".edit").forEach(updateSdNote)
);
$("add-edit").addEventListener("click", addEdit);
$("run").addEventListener("click", run);

addEdit();
loadDatasets().catch((e) => {
  $("status").textContent = e.message;
  $("status").className = "status err";
});
