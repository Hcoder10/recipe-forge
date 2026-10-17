// recipe-forge dash: subscribe to /api/stream and render every panel from one snapshot. No dependencies.
"use strict";

const SRC = new URLSearchParams(location.search).get("src") || "live";
const Q = SRC === "live" ? "" : `?src=${encodeURIComponent(SRC)}`;
let snap = null;
let skew = 0;            // server clock minus client clock, seconds
let prevStatus = {};
let showQueued = false;     // leaderboard key -> status, to flash rows that just became verified

// ------------------------------------------------------------------ helpers
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const now = () => Date.now() / 1000 + skew;
const fmt = (v, d = 1) => (v == null || isNaN(v) ? "—" : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));
const usd = (v, d = 2) => (v == null ? "—" : "$" + fmt(v, d));
function ago(ts) {
  if (!ts) return "";
  const s = Math.max(0, now() - ts);
  if (s < 60) return `${Math.floor(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}`;
}
function hms(ts) {
  if (!ts) return "--:--";
  const d = new Date((ts - skew) * 1000);
  return d.toTimeString().slice(0, 5);
}
function dur(s) {
  if (s == null) return "—";
  const neg = s < 0; s = Math.abs(Math.round(s));
  const m = Math.floor(s / 60), r = s % 60;
  return (neg ? "-" : "") + (m >= 60 ? `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}m` : `${m}:${String(r).padStart(2, "0")}`);
}
const tipAttr = (t) => ` data-tip="${esc(t)}"`;
const empty = (t) => `<div class="empty">${esc(t)}</div>`;

const ICON = {
  check: '<svg viewBox="0 0 16 16"><path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  x: '<svg viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>',
  dash: '<svg viewBox="0 0 16 16"><path d="M4 8h8" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>',
  run: '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="4" fill="currentColor"/></svg>',
  clock: '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M8 5v3.3l2 1.4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  warn: '<svg viewBox="0 0 16 16"><path d="M8 2l6.5 11.5h-13z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/><path d="M8 6.5v3.2M8 11.6v.1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  stop: '<svg viewBox="0 0 16 16"><rect x="3.5" y="3.5" width="9" height="9" rx="1.5" fill="currentColor"/></svg>',
  play: '<svg viewBox="0 0 16 16"><path d="M5 3.5l7 4.5-7 4.5z" fill="currentColor"/></svg>',
  loop: '<svg viewBox="0 0 16 16"><path d="M12.5 6.5A4.8 4.8 0 0 0 3.6 6M3.5 9.5a4.8 4.8 0 0 0 8.9.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M12.8 3v3.6H9.2M3.2 13V9.4h3.6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  hand: '<svg viewBox="0 0 16 16"><circle cx="8" cy="5" r="2.6" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M3 14c.6-2.8 2.6-4.2 5-4.2s4.4 1.4 5 4.2" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  ring: '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-dasharray="2.5 2"/></svg>',
};
const STATUS = {
  verified: ["verified", ICON.check], running: ["running", ICON.run], "no-gain": ["no gain", ICON.dash],
  queued: ["queued", ICON.ring], failed: ["failed", ICON.x],
};
const pill = (s) => {
  const [label, icon] = STATUS[s] || [s, ICON.ring];
  return `<span class="pill ${esc(STATUS[s] ? s : "queued")}">${icon}${esc(label)}</span>`;
};

// ------------------------------------------------------------------ header
function renderHeader(s) {
  $("fixture-banner").hidden = !s.fixture;
  document.title = (s.fixture ? "[fixture] " : "") + "recipe-forge";
  const t = $("src-toggle");
  t.textContent = SRC === "live" ? "fixture view" : "live view";
  t.href = SRC === "live" ? "?src=fixture" : location.pathname;
  const lb = s.leaderboard || [];
  const ver = lb.filter((r) => r.status === "verified");
  const bestDelta = Math.max(...ver.map((r) => r.delta_pct ?? -Infinity));
  const live = (s.missions || []).filter((m) => ["running", "active", "started", "live"].includes(m.status)).length;
  const gatesBusy = (s.gates || []).filter((g) => g.running).length;
  const spend = (s.spend?.crusoe?.total || 0) + (s.spend?.vast?.total_usd || 0);
  const k = [
    [`${ver.length}<span class="muted" style="font-size:.55em"> / ${lb.length}</span>`, "verified rows"],
    [isFinite(bestDelta) ? `+${fmt(bestDelta, 1)}%` : "—", "best delta"],
    [live, "live missions"],
    [`${gatesBusy}<span class="muted" style="font-size:.55em"> / ${(s.gates || []).length}</span>`, "gates busy"],
    [usd(spend), "spent"],
  ];
  $("kpis").innerHTML = k.map(([v, l]) => `<div class="kpi"><div class="v num">${v}</div><div class="l">${l}</div></div>`).join("");
}

// ------------------------------------------------------------------ 1. leaderboard
function deltaBar(d, max) {
  if (d == null) return '<span class="muted">—</span>';
  const W = 100, frac = Math.min(1, Math.abs(d) / (max || 1));
  const w = Math.max(1.5, frac * 62);
  const neg = d < 0;
  return `<svg class="deltabar" viewBox="0 0 ${W} 16" preserveAspectRatio="none"${tipAttr(`${d >= 0 ? "+" : ""}${fmt(d, 2)}% vs NVIDIA baseline`)}>
    <rect class="track" x="0" y="5" width="62" height="6" rx="3"/>
    <rect class="bar${neg ? " neg" : ""}" x="0" y="5" width="${w}" height="6" rx="3"/>
  </svg>`;
}
function renderLeaderboard(s) {
  const rows = s.leaderboard || [];
  if (!rows.length) { $("leaderboard").innerHTML = empty("No recipes or rows yet. Waiting for forge/recipes and results/leaderboard.jsonl."); return; }
  const max = Math.max(5, ...rows.map((r) => Math.abs(r.delta_pct || 0)));
  const flash = new Set();
  const next = {};
  for (const r of rows) {
    const key = `${r.recipe}|${r.gpu_class}`;
    next[key] = r.status;
    if (snap && prevStatus[key] && prevStatus[key] !== "verified" && r.status === "verified") flash.add(key);
  }
  prevStatus = next;
  // queued rows are the untouched catalog: show a few, fold the rest into one clickable line
  const active = rows.filter((r) => r.status !== "queued"), queued = rows.filter((r) => r.status === "queued");
  const nQ = showQueued ? queued.length : (active.length ? 4 : 8);
  const shown = active.concat(queued.slice(0, nQ));
  const rest = queued.length - Math.min(nQ, queued.length);
  const body = shown.map((r) => {
    const key = `${r.recipe}|${r.gpu_class}`;
    const unit = r.metric ? `<span class="unit">${esc(String(r.metric).replace(/_tok_s$/, "").replace("_", " "))}</span>` : "";
    const proof = r.verdict_id
      ? `<span${tipAttr(`gate verdict ${r.verdict_id}${r.machine ? " on " + r.machine : ""}${r.mission ? " · mission " + r.mission : ""}`)}>${esc(r.verdict_id)} · ${esc(r.pairs ?? "?")} pairs · ${r.lossless === false ? '<span class="bad">lossy</span>' : r.lossless ? '<span class="ok">lossless ✓</span>' : "lossless ?"}</span>`
      : '<span class="muted">—</span>';
    const dtxt = r.delta_pct == null ? "" : `${r.delta_pct >= 0 ? "+" : ""}${fmt(r.delta_pct, 1)}%`;
    const runNote = r.running && r.status === "verified" ? `<span class="run-note">+${r.running} running</span>` : "";
    return `<tr class="${esc(r.status)}${flash.has(key) ? " flash" : ""}">
      <td><div class="rid">${esc(r.recipe)}</div><div class="model">${esc(r.model || "")}</div></td>
      <td><span class="chip gpu"${tipAttr(r.gpu_class)}>${esc(gpuName(r.gpu_class))}</span></td>
      <td class="r num">${fmt(r.baseline_tok_s)}${r.baseline_tok_s != null ? unit : ""}</td>
      <td class="r num best">${r.status === "verified" ? fmt(r.best_tok_s) : '<span class="muted">—</span>'}</td>
      <td class="delta"><div style="display:flex;align-items:center;gap:.6em"><div style="flex:1">${deltaBar(r.status === "verified" ? r.delta_pct : null, max)}</div><b class="num" style="min-width:4.2em;text-align:right">${esc(dtxt)}</b></div></td>
      <td class="change"${tipAttr(r.change || "")}>${esc(r.change || "")}</td>
      <td class="proof">${proof}</td>
      <td>${pill(r.status)}${runNote}</td>
    </tr>`;
  }).join("");
  $("leaderboard").innerHTML = `<table class="lb">
    <thead><tr><th class="model-col">Recipe</th><th>GPU</th><th class="r">NVIDIA baseline tok/s</th><th class="r">Best verified tok/s</th>
    <th>Delta</th><th class="change">Change</th><th class="proof-col">Gate proof</th><th>Status</th></tr></thead>
    <tbody>${body}${rest > 0 || showQueued ? `<tr class="more"><td colspan="8"><a href="#" id="toggle-queued">${showQueued ? "fold queued rows" : `+ ${rest} more queued recipe × GPU rows`}</a></td></tr>` : ""}</tbody></table>`;
  const tq = $("toggle-queued");
  if (tq) tq.onclick = (e) => { e.preventDefault(); showQueued = !showQueued; renderLeaderboard(snap); };
}
const GPU_NAMES = { rtxpro6000_95g: "RTX PRO 6000", rtx5090_32g: "RTX 5090" };
function gpuName(k) {
  if (!k) return "?";
  if (/\s/.test(k)) return k;
  return GPU_NAMES[k] || String(k).replace(/_(\d+)g$/i, "").replace(/^rtx/i, "RTX ").replace(/pro/i, "PRO ").toUpperCase();
}

// ------------------------------------------------------------------ 2. missions
function meter(label, value, frac, cls = "") {
  const f = Math.max(0, Math.min(1, frac ?? 0));
  return `<div class="meter"><div class="lbl"><span>${label}</span><b>${value}</b></div><div class="trk"><div class="fill ${cls}" style="width:${(f * 100).toFixed(1)}%"></div></div></div>`;
}
function renderMissions(s) {
  const ms = s.missions || [];
  const live = ms.filter((m) => ["running", "active", "started", "live"].includes(m.status));
  $("missions-sub").textContent = `${live.length} live · ${ms.length - live.length} ended`;
  if (!ms.length) { $("missions").innerHTML = empty("No missions yet (missions/<id>/events.jsonl)."); return; }
  $("missions").innerHTML = ms.slice(0, 12).map((m) => {
    const isLive = ["running", "active", "started", "live"].includes(m.status);
    const left = m.deadline != null ? m.deadline - now() : null;
    const total = m.deadline && m.started_at ? m.deadline - m.started_at : null;
    const tcls = left != null && left < 300 ? "crit" : left != null && left < 900 ? "warn" : "";
    const bfrac = m.budget_usd ? m.usd / m.budget_usd : null;
    const team = (m.team || []).map((a) => {
      const idle = !a.last_ts || now() - a.last_ts > 120;
      const role = (a.role || "unknown").toLowerCase();
      return `<div class="nm"><span class="dot ${isLive && !idle ? "live" : "idle"}"></span>${esc(a.name)}</div>
        <span class="role ${esc(role)}">${esc(role)}</span>
        <span class="tool"${tipAttr([a.model, a.task].filter(Boolean).join(" · "))}>${a.tool ? esc(a.tool) : '<span class="muted">—</span>'}${a.lane ? ` <span class="chip lane">${esc(a.lane)}</span>` : ""}</span>
        <span class="muted num">${a.step != null ? "step " + esc(a.step) : ""}</span>
        <span class="ago">${a.last_ts ? ago(a.last_ts) + " ago" : ""}</span>`;
    }).join("");
    return `<div class="mission">
      <div class="mh">
        <span class="title">${esc(m.recipe || "?")}</span><span class="at">on</span><b>${esc(m.machine || "?")}</b>
        ${m.lane ? `<span class="chip lane">${esc(m.lane)}</span>` : ""}
        ${isLive ? "" : `<span class="pill no-gain">${esc(m.status)}</span>`}
        <span class="muted mono" style="font-size:.78em">${esc(m.id)}</span>
        <div class="right">
          ${meter("spent", usd(m.usd) + (m.budget_usd ? ` / ${usd(m.budget_usd, 0)}` : ""), bfrac, bfrac > 0.85 ? "crit" : bfrac > 0.6 ? "warn" : "")}
          ${isLive ? meter("time left", `<span class="ticker" data-deadline="${m.deadline ?? ""}">${dur(left)}</span>`, total ? left / total : null, tcls) : ""}
        </div>
      </div>
      <div class="team">${team}</div>
    </div>`;
  }).join("");
}

// ------------------------------------------------------------------ 3. gates
function pairDots(gains) {
  if (!gains || !gains.length) return "";
  const r = 4.5, gap = 3;
  const w = gains.length * (2 * r + gap);
  return `<svg class="pairs" viewBox="0 0 ${w} ${2 * r}" style="width:${w / 10}em"${tipAttr("pair log-gains: " + gains.map((g) => fmt(g, 3)).join(", "))}>${gains
    .map((g, i) => `<circle cx="${r + i * (2 * r + gap)}" cy="${r}" r="${r - 0.5}" fill="${g > 0 ? "var(--good)" : "var(--critical)"}"/>`).join("")}</svg>`;
}
function renderGates(s) {
  const gs = s.gates || [];
  if (!gs.length) { $("gates").innerHTML = empty("No gates reporting (results/gate/<machine>/)."); return; }
  $("gates").innerHTML = gs.map((g) => {
    const run = g.running;
    const nowLine = run
      ? `<div class="now"><span class="dot live"></span> gating <b>${esc(run.mission || run.request_id || "")}</b> ${run.phase ? "· " + esc(run.phase) : ""} ${run.started_at ? "· " + ago(run.started_at) : ""}</div>`
      : `<div class="now muted">idle</div>`;
    const q = (g.queued || []).map((x) => `<span class="chip"${tipAttr(x.claim || x.bundle || "")}>${ICON.clock.replace("<svg", '<svg style="width:1em;height:1em"')} ${esc(x.mission || x.request_id || "?")}</span>`).join("");
    const vs = (g.verdicts || []).slice(0, 5).map((v) => {
      const res = v.win && v.lossless !== false ? '<span class="badge win">WIN</span>' : v.lossless === false ? '<span class="badge lossy">LOSSY</span>' : '<span class="badge lose">no win</span>';
      const tip = `${v.verdict_id} · ${v.kind || ""} · ${v.pairs ?? "?"} pairs · mean log-gain ${fmt(v.mean_log_gain, 4)} (bar ${fmt(v.threshold, 4)})${v.lossless_detail?.top1 != null ? ` · top-1 ${v.lossless_detail.top1}` : ""}${v.reason ? " · " + v.reason : ""}`;
      return `<div class="vrow"${tipAttr(tip)}><span class="vid">${esc(v.verdict_id)}</span><span class="rec">${esc(v.recipe || v.mission || "")}</span>${pairDots(v.pair_gains) || `<span class="muted">${esc(v.pairs ?? "")} pairs</span>`}<span class="g">${v.delta_pct != null ? (v.delta_pct >= 0 ? "+" : "") + fmt(v.delta_pct, 1) + "%" : ""}</span>${res}</div>`;
    }).join("");
    return `<div class="gate">
      <div class="gh"><span class="m">${esc(g.machine)}</span>${g.quiet === false ? '<span class="sev warning">' + ICON.warn + "not quiet</span>" : ""}
        <span class="right">${g.n_wins || 0} win${g.n_wins === 1 ? "" : "s"} / ${g.n_verdicts || 0} verdicts · ${(g.queued || []).length} queued</span></div>
      ${nowLine}${q ? `<div class="q">${q}</div>` : ""}${vs || '<div class="muted" style="font-size:.85em">no verdicts yet</div>'}
    </div>`;
  }).join("");
}

// ------------------------------------------------------------------ 4. decisions, 5. alerts
const DEC_ICON = { continue: ICON.play, replan: ICON.loop, stop: ICON.stop, ask_human: ICON.hand };
function renderDecisions(s) {
  const ds = s.decisions || [];
  if (!ds.length) { $("decisions").innerHTML = empty("No decisions logged yet."); return; }
  $("decisions").innerHTML = ds.slice(0, 30).map((d) => `<div class="item">
    <span class="when">${hms(d.ts)}</span>
    <div class="what"><div class="line1"><span class="dec ${esc(d.decision)}">${DEC_ICON[d.decision] || ""}${esc(d.decision)}</span>
      <span class="ctx">${esc(d.mission || "")}${d.agent ? " · " + esc(d.agent) : ""}${d.evidence ? " · " + esc(d.evidence) : ""}</span></div>
      <div class="why">${esc(d.rationale)}</div></div></div>`).join("");
}
const SEV_ICON = { critical: ICON.x, serious: ICON.warn, warning: ICON.warn, info: ICON.ring };
function renderAlerts(s) {
  const as = s.alerts || [];
  if (!as.length) { $("alerts").innerHTML = empty("No alerts. Quiet is good."); return; }
  $("alerts").innerHTML = as.slice(0, 30).map((a) => {
    const sev = SEV_ICON[a.severity] ? a.severity : "warning";
    return `<div class="item"><span class="when">${hms(a.ts)}</span>
    <div class="what"><div class="line1"><span class="sev ${sev}">${SEV_ICON[sev]}${esc(a.detector)}</span>
      <span class="fault ${esc(a.fault)}">${esc(a.fault)}</span>${a.action ? `<span class="ctx">→ ${esc(a.action)}</span>` : ""}
      <span class="ctx">${esc(a.mission || a.machine || "")}${a.agent ? " · " + esc(a.agent) : ""}</span></div>
      ${a.message ? `<div class="why">${esc(a.message)}</div>` : ""}</div></div>`;
  }).join("");
}

// ------------------------------------------------------------------ 6. N+1
const VER_ICON = {
  promoted: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="9" fill="var(--good)"/><path d="M6 10.5l2.6 2.6L14.2 7.5" fill="none" stroke="#0b0b0a" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  rejected: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="9" fill="var(--critical)"/><path d="M7 7l6 6M13 7l-6 6" stroke="#0b0b0a" stroke-width="2.4" stroke-linecap="round"/></svg>',
  candidate: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="var(--surface)" stroke="var(--muted)" stroke-width="2" stroke-dasharray="3 2.5"/></svg>',
};
function renderN1(s) {
  const vs = s.n1?.versions || [];
  if (!vs.length) { $("n1").innerHTML = empty("No playbook versions yet (n1/versions.jsonl)."); return; }
  const nodes = vs.map((v, i) => `<div class="ver"${tipAttr(v.summary || "")}>
      ${i < vs.length - 1 ? '<div class="edge"></div>' : ""}
      <div class="node">${VER_ICON[v.status] || VER_ICON.candidate}<span>${esc(v.version)}</span></div>
      <div class="vs ${esc(VER_ICON[v.status] ? v.status : "candidate")}">${esc(v.status)}${v.paired_delta != null ? ` · ${v.paired_delta >= 0 ? "+" : ""}${fmt(v.paired_delta, 1)} pts` : ""}${v.missions ? ` · ${esc(v.missions)} missions` : ""}</div>
      <div class="vsum">${esc(v.summary)}</div></div>`).join("");

  // dumbbell: unseen-recipe delta % before (previous version) -> after (this version)
  const items = [];
  for (const v of vs) for (const u of v.unseen || []) if (u.before != null || u.after != null) items.push({ v, u });
  let chart = '<div class="muted" style="font-size:.85em">No unseen-recipe comparisons yet.</div>';
  if (items.length) {
    const vals = items.flatMap(({ u }) => [u.before, u.after]).filter((x) => x != null);
    let lo = Math.min(0, ...vals), hi = Math.max(1, ...vals);
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const W = 600, L = 250, R = 16, rowH = 24, top = 6, H = top + items.length * rowH + 22;
    const x = (val) => L + ((val - lo) / (hi - lo)) * (W - L - R);
    const ticks = niceTicks(lo, hi, 5);
    let g = ticks.map((t) => `<line class="${t === 0 ? "zero" : "grid"}" x1="${x(t)}" x2="${x(t)}" y1="${top - 2}" y2="${H - 18}"/><text class="ax" x="${x(t)}" y="${H - 4}" text-anchor="middle">${t > 0 ? "+" : ""}${t}%</text>`).join("");
    items.forEach(({ v, u }, i) => {
      const y = top + i * rowH + rowH / 2;
      const col = v.status === "promoted" ? "var(--good)" : v.status === "rejected" ? "var(--critical)" : "var(--muted)";
      const tip = `${v.version} (${v.status}) on ${u.recipe}: ${fmt(u.before, 1)}% → ${fmt(u.after, 1)}%`;
      g += `<g${tipAttr(tip)}><rect x="0" y="${y - rowH / 2}" width="${W}" height="${rowH}" fill="transparent"/>
        <text x="${L - 10}" y="${y + 4}" text-anchor="end">${esc(v.version)} · ${esc(u.recipe)}</text>`;
      if (u.before != null && u.after != null) g += `<line class="link" x1="${x(u.before)}" x2="${x(u.after)}" y1="${y}" y2="${y}" stroke="${col}" stroke-opacity=".55"/>`;
      if (u.before != null) g += `<circle class="before" cx="${x(u.before)}" cy="${y}" r="5"/>`;
      if (u.after != null) g += `<circle class="after" cx="${x(u.after)}" cy="${y}" r="6.5" fill="${col}"/>`;
      g += "</g>";
    });
    chart = `<svg class="dumb" viewBox="0 0 ${W} ${H}">${g}</svg>
      <div class="legend"><span><svg width="12" height="12" style="vertical-align:-1px;margin-right:.35em"><circle cx="6" cy="6" r="4.5" fill="var(--surface)" stroke="var(--ink-2)" stroke-width="2"/></svg>previous version</span>
      <span><span class="sw" style="background:var(--good);border-radius:50%"></span>promoted</span><span><span class="sw" style="background:var(--critical);border-radius:50%"></span>rejected</span>
      <span class="muted">x: delta % vs NVIDIA baseline on recipes the version had not seen</span></div>`;
  }
  $("n1").innerHTML = `<div class="versions">${nodes}</div>${chart}`;
}
function niceTicks(lo, hi, n) {
  const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.round(t * 1e6) / 1e6);
  return out;
}

// ------------------------------------------------------------------ 7. brain
const FACT_COLOR = { measured: "var(--good)", verified: "var(--good)", in_test: "var(--c1)", open: "var(--axis)", proposed: "var(--axis)", blocked: "var(--c4)", refuted: "var(--critical)" };
const factColor = (st) => FACT_COLOR[st] || "var(--c7)";
function renderBrain(s) {
  const b = s.brain || {};
  $("brain-sub").textContent = `${b.total || 0} facts across missions`;
  if (!b.total) { $("brain").innerHTML = empty("No facts yet (brain/facts.jsonl)."); return; }
  const order = Object.entries(b.by_status || {}).sort((a, b2) => {
    const rank = (k) => ({ measured: 0, verified: 0, in_test: 1, open: 2, proposed: 2, blocked: 3, refuted: 4 }[k] ?? 5);
    return rank(a[0]) - rank(b2[0]);
  });
  let xx = 0, segs = "";
  const W = 400;
  for (const [st, n] of order) {
    const w = (n / b.total) * W;
    segs += `<rect x="${xx}" y="0" width="${Math.max(0, w - 2)}" height="14" rx="3" fill="${factColor(st)}"${tipAttr(`${st}: ${n}`)}/>`;
    xx += w;
  }
  const legend = order.map(([st, n]) => `<span><span class="sw" style="background:${factColor(st)}"></span>${esc(st.replace("_", " "))} <b>${n}</b></span>`).join("");
  const facts = (b.latest || []).map((f) => `<div class="fact"${tipAttr(f.claim + (f.evidence ? " · evidence " + f.evidence : ""))}>
    <span class="st" style="color:${["proposed", "open"].includes(f.status) ? "var(--ink-2)" : factColor(f.status)}">${esc(f.status.replace("_", " "))}</span>
    <span class="cl">${esc(f.claim)}</span>
    <span class="meta">${esc(f.recipe || "")}${f.lane ? " · " + esc(f.lane) : ""} · ${ago(f.ts)}</span></div>`).join("");
  $("brain").innerHTML = `<svg class="stack" viewBox="0 0 ${W} 14" preserveAspectRatio="none">${segs}</svg>
    <div class="legend">${legend}</div><div class="facts">${facts}</div>`;
}

// ------------------------------------------------------------------ 8. spend
const ROLE_COLOR = { lead: "var(--c1)", worker: "var(--c3)", verifier: "var(--c5)", mechanic: "var(--c4)" };
function areaChart(series) {
  if (!series || series.length < 2) return '<div class="muted" style="font-size:.85em">No Crusoe calls logged yet.</div>';
  const W = 400, H = 70, t0 = series[0][0], t1 = series[series.length - 1][0], vmax = series[series.length - 1][1] || 1;
  const x = (t) => ((t - t0) / Math.max(1, t1 - t0)) * W, y = (v) => H - (v / vmax) * (H - 6);
  const pts = series.map(([t, v]) => `${x(t).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  return `<svg class="area" viewBox="0 0 ${W} ${H + 14}" data-area='${esc(JSON.stringify({ t0, t1, vmax, W, H }))}'>
    <polygon class="ar" points="0,${H} ${pts} ${W},${H}"/><polyline class="ln" points="${pts}"/>
    <line class="base" x1="0" x2="${W}" y1="${H}" y2="${H}"/>
    <text x="0" y="${H + 12}">${hms(t0)}</text><text x="${W}" y="${H + 12}" text-anchor="end">${hms(t1)}</text>
    <line class="hov" x1="0" x2="0" y1="0" y2="${H}" visibility="hidden"/><circle class="hovd" r="4" visibility="hidden"/></svg>`;
}
function renderSpend(s) {
  const c = s.spend?.crusoe || {}, v = s.spend?.vast || {};
  const roles = Object.entries(c.by_role || {});
  const rmax = Math.max(1e-9, ...roles.map(([, x]) => x));
  const bars = roles.length ? roles.map(([r, x]) => `<div class="hbar"><span>${esc(r)}</span><div class="t"><div class="f" style="width:${((x / rmax) * 100).toFixed(1)}%;background:${ROLE_COLOR[r] || "var(--c7)"}"></div></div><span class="v">${usd(x)}</span></div>`).join("")
    : '<div class="muted" style="font-size:.85em">No calls yet.</div>';
  const rent = (v.machines || []).map((m) => `<span class="dot ${m.active ? "on" : "idle"}"></span><span>${esc(m.machine)} <span class="muted">${esc(m.gpu_class || "")}</span></span>
    <span class="r muted">${m.usd_per_hr != null ? usd(m.usd_per_hr) + "/h" : ""}</span><span class="r">${usd(m.spent_usd)}</span>`).join("");
  $("spend").innerHTML = `<div class="tiles">
      <div class="tile"><div class="l">Crusoe inference</div><div class="v num">${usd(c.total)}</div><div class="s">${roles.length} roles</div></div>
      <div class="tile"><div class="l">Vast burn</div><div class="v num">${usd(v.burn_usd_per_hr)}<span class="muted" style="font-size:.5em">/h</span></div><div class="s">${(v.machines || []).filter((m) => m.active).length} boxes active</div></div>
      <div class="tile"><div class="l">Vast total</div><div class="v num">${usd(v.total_usd)}</div><div class="s">${(v.machines || []).length} rentals</div></div>
    </div>
    <div class="spend-cols">
      <div><h3>Crusoe $ by role</h3>${bars}<h3 style="margin-top:.8em">Crusoe $ cumulative</h3>${areaChart(c.series)}</div>
      <div><h3>Vast rentals</h3>${rent ? `<div class="rent">${rent}</div>` : '<div class="muted" style="font-size:.85em">No rentals.</div>'}</div>
    </div>`;
}

// ------------------------------------------------------------------ footer
function renderFoot(s) {
  const src = Object.entries(s.sources || {});
  const ok = src.filter(([, v]) => v.ok).map(([k]) => k);
  const err = src.filter(([, v]) => v.err).map(([k, v]) => `${k}: ${v.err}`);
  $("foot").innerHTML = `<span>root ${esc(s.root)}</span><span>reading: ${esc(ok.join(", ") || "nothing yet")}</span>
    ${err.length ? `<span class="bad">${esc(err.join("; "))}</span>` : ""}${s.server?.error ? `<span class="bad">server: ${esc(s.server.error)}</span>` : ""}
    <span>keys 1-8 zoom a panel, Esc back</span>`;
}

// ------------------------------------------------------------------ main loop
function render(s) {
  skew = s.generated_at ? s.generated_at - Date.now() / 1000 : 0;
  const parts = [renderHeader, renderLeaderboard, renderMissions, renderGates, renderDecisions, renderAlerts, renderN1, renderBrain, renderSpend, renderFoot];
  for (const f of parts) {
    try { f(s); } catch (e) { console.error(f.name, e); }
  }
  snap = s;
}
function tick() {
  $("clock").textContent = new Date().toTimeString().slice(0, 8);
  for (const el of document.querySelectorAll(".ticker")) {
    const d = parseFloat(el.dataset.deadline);
    if (!isNaN(d)) el.textContent = dur(d - now());
  }
}
function connect() {
  const es = new EventSource("/api/stream" + Q);
  es.addEventListener("snapshot", (e) => {
    $("conn-dot").className = "dot on";
    try { render(JSON.parse(e.data)); } catch (err) { console.error(err); }
  });
  es.onopen = () => { $("conn-dot").className = "dot on"; };
  es.onerror = () => { $("conn-dot").className = "dot off"; };  // EventSource reconnects by itself
}

// tooltips: any element with data-tip
const tip = $("tip");
document.addEventListener("mousemove", (e) => {
  const area = e.target.closest && e.target.closest("svg.area");
  if (area) return areaHover(area, e);
  const el = e.target.closest && e.target.closest("[data-tip]");
  if (!el || !el.dataset.tip) { tip.hidden = true; return; }
  tip.textContent = el.dataset.tip;
  tip.hidden = false;
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + "px";
  tip.style.top = (e.clientY + 16) + "px";
});
function areaHover(svg, e) {
  const m = JSON.parse(svg.dataset.area), series = snap?.spend?.crusoe?.series || [];
  const r = svg.getBoundingClientRect(), fx = (e.clientX - r.left) / r.width, t = m.t0 + fx * (m.t1 - m.t0);
  let best = series[0];
  for (const p of series) if (Math.abs(p[0] - t) < Math.abs(best[0] - t)) best = p;
  const px = ((best[0] - m.t0) / Math.max(1, m.t1 - m.t0)) * m.W, py = m.H - (best[1] / (m.vmax || 1)) * (m.H - 6);
  const ln = svg.querySelector(".hov"), dot = svg.querySelector(".hovd");
  ln.setAttribute("x1", px); ln.setAttribute("x2", px); ln.setAttribute("visibility", "visible");
  dot.setAttribute("cx", px); dot.setAttribute("cy", py); dot.setAttribute("visibility", "visible");
  tip.textContent = `${hms(best[0])}  ${usd(best[1], 3)} cumulative`;
  tip.hidden = false;
  tip.style.left = Math.min(e.clientX + 14, innerWidth - 200) + "px";
  tip.style.top = (e.clientY + 16) + "px";
  svg.onmouseleave = () => { ln.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tip.hidden = true; };
}

// zoom a panel with 1-8
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  const cur = document.querySelector(".panel.zoom");
  if (e.key === "Escape" || (cur && cur.dataset.key === e.key)) {
    document.body.classList.remove("zoomed"); cur && cur.classList.remove("zoom"); return;
  }
  const p = document.querySelector(`.panel[data-key="${e.key}"]`);
  if (p) {
    cur && cur.classList.remove("zoom");
    p.classList.add("zoom"); document.body.classList.add("zoomed");
  }
});

setInterval(tick, 1000);
tick();
// ?nostream=1 renders one snapshot and stops (headless screenshots never settle with an open stream)
const NOSTREAM = new URLSearchParams(location.search).has("nostream");
fetch("/api/snapshot" + Q).then((r) => r.json()).then(render).catch(() => {}).finally(() => { if (!NOSTREAM) connect(); });
