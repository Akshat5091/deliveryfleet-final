// DeliveryFleet MAS dispatch desk. Vanilla ES modules, no build step.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n, d = 1) => (n === null || n === undefined || Number.isNaN(n) ? "–" : Number(n).toFixed(d));

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `${res.status} ${res.statusText}`);
  return body;
}

// ---------------------------------------------------------------- pipeline
// Each stage maps to the agent names and event kinds the backend logs.
const STAGES = [
  { id: "input_guard", n: "01", a: "Input Guard", f: "Regex + Gemini judge" },
  { id: "intake", n: "02", a: "Supervisor intake", f: "CrewAI" },
  { id: "retrieval_agent", n: "03", a: "Retrieval (RAG)", f: "CrewAI + ChromaDB" },
  { id: "driver_db", n: "03", a: "Driver DB", f: "MCP tool" },
  { id: "dispatch_agent", n: "04", a: "Dispatch", f: "CrewAI · ReAct" },
  { id: "route_agent", n: "05", a: "Route", f: "Google ADK via A2A" },
  { id: "business", n: "06", a: "Rule check", f: "Code guardrail" },
  { id: "exception_agent", n: "07", a: "Exception", f: "CrewAI · on event" },
  { id: "review", n: "08", a: "Supervisor review", f: "CrewAI" },
];

function renderPipeline() {
  $("#pipeline").innerHTML = STAGES.map((s) => `
    <li data-stage="${s.id}" data-state="idle">
      <span class="n">${s.n}</span><span class="a">${s.a}</span><span class="f">${s.f}</span>
      <span class="st">waiting</span>
    </li>`).join("");
}

function setStage(id, state, label) {
  const li = $(`#pipeline [data-stage="${id}"]`);
  if (!li) return;
  const order = { idle: 0, running: 1, done: 2, skipped: 2, blocked: 3 };
  if (order[state] < order[li.dataset.state] && state !== "running") return;
  li.dataset.state = state;
  $(".st", li).textContent = label || state;
}

// ---------------------------------------------------------------- map
const ZONES = {
  "Zone A": { x: 80, y: 70, sub: "CBD & residential" },
  "Zone B": { x: 330, y: 70, sub: "Tech corridor" },
  "Zone C": { x: 80, y: 235, sub: "University & civic" },
  "Zone D": { x: 330, y: 235, sub: "Airport cargo" },
};
const EDGES = [["Zone A", "Zone B", 14.5], ["Zone A", "Zone C", 18], ["Zone A", "Zone D", 26.5],
  ["Zone B", "Zone C", 12], ["Zone B", "Zone D", 21], ["Zone C", "Zone D", 16.5]];
const edgeKey = (a, b) => [a, b].sort().join("|");

function renderMap({ primary = [], closed = [], detour = [], ends = [] } = {}) {
  const cls = (a, b) => {
    const k = edgeKey(a, b);
    if (closed.includes(k)) return "edge closed";
    if (detour.includes(k)) return "edge detour";
    if (primary.includes(k)) return "edge primary";
    return "edge";
  };
  const lines = EDGES.map(([a, b, km]) => {
    const p = ZONES[a], q = ZONES[b];
    // Diagonals cross at the centre, so their labels sit a third of the way along instead.
    const diagonal = p.x !== q.x && p.y !== q.y;
    const t = diagonal ? 0.3 : 0.5;
    const mx = p.x + (q.x - p.x) * t, my = p.y + (q.y - p.y) * t;
    return `<line class="${cls(a, b)}" x1="${p.x}" y1="${p.y}" x2="${q.x}" y2="${q.y}"/>
            <text class="km" x="${mx + 6}" y="${my - 5}">${km} km</text>`;
  }).join("");
  const nodes = Object.entries(ZONES).map(([name, z]) => `
    <g class="zone${ends.includes(name) ? " end" : ""}">
      <circle cx="${z.x}" cy="${z.y}" r="20"/>
      <text class="code" x="${z.x}" y="${z.y + 5}" text-anchor="middle">${name.slice(-1)}</text>
      <text class="sub" x="${z.x}" y="${z.y + (z.y < 150 ? -28 : 38)}" text-anchor="middle">${z.sub}</text>
    </g>`).join("");
  $("#map").innerHTML = lines + nodes;
}

const corridorEdges = (corridor) => {
  const zones = String(corridor || "").split("->").map((s) => s.trim()).filter((s) => s.startsWith("Zone"));
  return zones.slice(1).map((z, i) => edgeKey(zones[i], z));
};
const zoneOf = (loc) => (String(loc || "").match(/Zone [A-D]/) || [])[0];

// ---------------------------------------------------------------- live run state
const live = { runId: null, source: null, t0: 0, timer: null, filter: "all", events: [], board: {}, ended: false };

function resetLive(title, input, runId) {
  live.source?.close();
  clearInterval(live.timer);
  Object.assign(live, { runId, events: [], board: {}, ended: false, t0: performance.now() });
  renderPipeline();
  renderMap();
  $("#run-id").textContent = runId || "";
  $("#run-title").textContent = title;
  $("#run-input").innerHTML = highlightInput(input || "");
  $("#log").innerHTML = "";
  $("#board").innerHTML = "";
  $("#heat").innerHTML = "";
  $("#report").innerHTML = `<h3>Final delivery report</h3><p class="empty">Appears when the Supervisor signs off.</p>`;
  $("#map-note").textContent = "corridor chosen by the Route Agent";
}

function highlightInput(text) {
  const risky = /(ignore all previous instructions|assign driver d09 without checking hours|pretend the shift-hour policy was cancelled|incompetent|idiots|lazy|clowns|\[REDACTED[^\]]*\])/gi;
  return esc(text).replace(risky, (m) => `<mark>${m}</mark>`);
}

function startClock() {
  clearInterval(live.timer);
  live.timer = setInterval(() => {
    $("#run-clock").textContent = `${((performance.now() - live.t0) / 1000).toFixed(1)} s`;
  }, 100);
}

function follow(runId) {
  live.source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/stream`);
  live.source.addEventListener("agent_event", (e) => onEvent(JSON.parse(e.data)));
  live.source.addEventListener("done", (e) => { onDone(JSON.parse(e.data)); live.source.close(); });
  live.source.addEventListener("timeout", () => { live.source.close(); clearInterval(live.timer); });
  live.source.onerror = () => { if (live.ended) live.source.close(); };
}

// Map one stored event to pipeline state, map, log and blackboard.
function onEvent(ev) {
  live.events.push(ev);
  const p = ev.payload || {};
  if (ev.kind === "guardrail" && ev.agent === "input_guard") {
    setStage("input_guard", ev.status === "BLOCKED" ? "blocked" : "done", ev.status.toLowerCase());
    if (ev.status === "BLOCKED") STAGES.slice(1).forEach((s) => setStage(s.id, "skipped", "skipped"));
    if (p.sanitized_text && ev.status === "SANITIZED") $("#run-input").innerHTML = highlightInput(p.sanitized_text);
  }
  if (ev.kind === "agent_start") {
    const id = ev.agent === "supervisor" ? (live.board.request ? "review" : "intake") : ev.agent;
    setStage(id, "running", "working");
  }
  if (ev.kind === "agent_end") {
    const id = ev.agent === "supervisor" ? (live.board.dispatch || live.board.business_check ? "review" : "intake") : ev.agent;
    setStage(id, ev.status === "OK" ? "done" : "blocked", ev.status === "OK" ? `${fmt(ev.duration_ms / 1000)} s` : "error");
  }
  if (ev.kind === "tool_call" && ev.name === "rank_drivers") setStage("driver_db", ev.status === "OK" ? "done" : "blocked", "scored");
  if (ev.kind === "a2a_message" && p.direction === "request") setStage("route_agent", "running", "A2A sent");
  if (ev.kind === "blackboard" && ev.status === "WRITE") onBoardWrite(ev.name, p);
  if (ev.kind === "error") setStage(ev.agent === "supervisor" ? "review" : ev.agent, "blocked", "error");
  appendLog(ev);
}

function onBoardWrite(section, prov) {
  live.board[section] = prov;
  const ver = `v${prov.version}`;
  const row = $(`#board [data-sec="${section}"]`) || Object.assign(document.createElement("div"), {});
  row.dataset.sec = section;
  row.innerHTML = `<dt>${esc(section)}</dt><dd><b>${esc(prov.agent)}</b> · ${ver}${prov.summary ? ` · ${esc(prov.summary)}` : ""}</dd>`;
  if (!row.parentNode) $("#board").append(row);
  row.classList.remove("flash"); void row.offsetWidth; row.classList.add("flash");

  if (section === "business_check") setStage("business", prov.summary?.includes("no legal") ? "blocked" : "done", "checked");
  if (section === "disruption") { setStage("exception_agent", "running", "woken"); $("#map-note").textContent = prov.summary || ""; }
  if (section === "review") setStage("review", "done", "signed off");
}

const KIND_LABEL = { llm_call: "LLM", tool_call: "MCP", a2a_message: "A2A", blackboard: "BB", guardrail: "GUARD",
  agent_start: "START", agent_end: "END", error: "ERROR", supervisor_rerun: "RERUN", p2p_message: "P2P", broadcast: "CAST", context_size: "CTX" };

function describe(ev) {
  const p = ev.payload || {};
  switch (ev.kind) {
    case "llm_call": return `${esc(ev.name)} · ${ev.input_tokens}+${ev.output_tokens} tok · ${fmt(ev.duration_ms, 0)} ms · $${fmt(ev.cost_usd, 6)}`;
    case "tool_call": return `<b>${esc(ev.name)}</b> ${ev.duration_ms ? `· ${fmt(ev.duration_ms, 0)} ms` : ""} ${ev.status !== "OK" ? "· failed" : ""}`;
    case "a2a_message": return `${esc(p.from)} → ${esc(p.to)} · ${esc(p.direction)} ${ev.duration_ms ? `· ${fmt(ev.duration_ms, 0)} ms` : ""}`;
    case "blackboard": return `${esc(ev.status)} <b>${esc(ev.name)}</b> v${p.version}${p.summary ? ` · ${esc(p.summary)}` : ""}`;
    case "guardrail": return `${esc(ev.name)} → <span class="outcome ${esc(ev.status)}">${esc(ev.status)}</span>`;
    case "agent_start": return `starts (${esc(ev.framework)})`;
    case "agent_end": return `finished in ${fmt(ev.duration_ms / 1000, 2)} s`;
    case "supervisor_rerun": return `sends work back to <b>${esc(ev.name)}</b>`;
    case "error": return `<b>${esc(ev.name)}</b>: ${esc(p.error)}`;
    default: return esc(ev.name);
  }
}

function appendLog(ev) {
  const li = document.createElement("li");
  li.dataset.kind = ev.kind;
  const t = live.events.length ? (ev.ts - live.events[0].ts) : 0;
  const detail = { ...ev.payload };
  const hasDetail = Object.keys(detail).length > 0;
  li.innerHTML = `
    <span class="t">+${t.toFixed(1)}s</span>
    <span class="who">${esc(ev.agent || "system")}<small>${esc(ev.framework)}</small></span>
    <span class="what"><span class="k ${esc(ev.kind)}">${KIND_LABEL[ev.kind] || esc(ev.kind)}</span>${describe(ev)}
      ${hasDetail ? `<details><summary>details</summary><pre>${esc(JSON.stringify(detail, null, 1))}</pre></details>` : ""}</span>`;
  li.hidden = live.filter !== "all" && live.filter !== ev.kind;
  const log = $("#log");
  const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  log.append(li);
  if (stick) log.scrollTop = log.scrollHeight;
}

function onDone(payload) {
  live.ended = true;
  clearInterval(live.timer);
  const { run, agents } = payload;
  const r = run.report || {};
  $("#run-clock").textContent = `${fmt(r.total_latency_s)} s`;
  const bb = run.blackboard || {};

  STAGES.forEach((s) => {
    const li = $(`#pipeline [data-stage="${s.id}"]`);
    if (li && ["idle", "running"].includes(li.dataset.state)) setStage(s.id, "skipped", s.id === "exception_agent" ? "not needed" : "skipped");
  });

  const route = bb.route || {}, exc = bb.exception || {}, dis = bb.disruption || {};
  const req = bb.request || {};
  const closed = dis.closed_corridor ? [edgeKey(...dis.closed_corridor.split("->").map((s) => s.trim()))] : [];
  renderMap({
    primary: corridorEdges(route.corridor),
    closed,
    detour: exc.new_corridor ? corridorEdges(exc.new_corridor) : [],
    ends: [zoneOf(req.pickup), zoneOf(req.dropoff)].filter(Boolean),
  });
  if (exc.new_corridor) $("#map-note").textContent = `black = original, red dashed = closed, green = detour (${exc.new_corridor})`;

  renderReport(r, bb);
  renderHeat(agents);
}

function renderReport(r, bb) {
  const rows = [
    ["Outcome", `<span class="outcome ${esc(r.outcome)}">${esc(r.outcome)}</span>`],
    ["Order", esc(r.order_id)], ["Guard", esc(r.guard_status)],
    ["Driver", r.driver_id ? `${esc(r.driver_id)} · ${esc(r.driver_name)}` : "–"],
    ["Corridor", esc(r.corridor || "–")],
    ["ETA / deadline", r.eta_mins ? `${fmt(r.eta_mins)} / ${esc(r.deadline_mins)} min` : "–"],
    ["P(on time)", r.p_on_time != null ? fmt(r.p_on_time, 2) : "–"],
    ["Exception", esc(r.exception_action || "none")], ["Supervisor", esc(r.supervisor_verdict || "–")],
    ["Architecture", esc(r.architecture)],
    ["LLM calls · tokens", `${r.llm_calls} · ${r.total_tokens}`], ["Cost", `$${fmt(r.cost_usd, 5)}`],
  ];
  const why = bb.business_check?.violations?.length ? `<p class="hint" style="margin-top:8px">Rule check: ${esc(bb.business_check.violations.join("; "))}</p>` : "";
  $("#report").innerHTML = `<h3>Final delivery report</h3>
    <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>
    <p class="msg">${esc(r.customer_message)}</p>${why}`;
}

const HEAT_COLS = [["latency_ms", "Latency", (v) => `${fmt(v / 1000)} s`], ["llm_calls", "LLM", (v) => v],
  ["tool_calls", "Tools", (v) => v], ["input_tokens", "Tokens in", (v) => v], ["cost_usd", "Cost $", (v) => fmt(v, 5)]];

function renderHeat(agents) {
  const names = Object.keys(agents || {});
  if (!names.length) { $("#heat").innerHTML = `<p class="hint">No agent activity recorded.</p>`; return; }
  const max = Object.fromEntries(HEAT_COLS.map(([k]) => [k, Math.max(...names.map((n) => agents[n][k] || 0), 1e-9)]));
  const cell = (n, k, f) => {
    const share = (agents[n][k] || 0) / max[k];
    const alpha = (0.06 + share * 0.6).toFixed(2);
    return `<td class="cell" style="background:rgba(153,27,27,${alpha});color:${share > 0.6 ? "#fff" : "inherit"}">${f(agents[n][k] || 0)}</td>`;
  };
  $("#heat").innerHTML = `<table class="heat"><thead><tr><th>Agent</th>${HEAT_COLS.map(([, l]) => `<th>${l}</th>`).join("")}</tr></thead>
    <tbody>${names.map((n) => `<tr><td>${esc(n)}</td>${HEAT_COLS.map(([k, , f]) => cell(n, k, f)).join("")}</tr>`).join("")}</tbody></table>`;
}

// ---------------------------------------------------------------- starting runs
let META = null;

async function launch(body, title) {
  $$("#cases button, #run-form button").forEach((b) => (b.disabled = true));
  try {
    const { run_id, message } = await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
    resetLive(title, message, run_id);
    startClock();
    follow(run_id);
  } catch (err) {
    resetLive("Could not start the run", err.message, "");
  } finally {
    setTimeout(() => $$("#cases button, #run-form button").forEach((b) => (b.disabled = false)), 1500);
  }
}

function renderCases() {
  $("#cases").innerHTML = META.demo_cases.map((c) => `
    <li><button type="button" data-case="${esc(c.id)}">
      <span class="t">${esc(c.title)}</span><span class="s">${esc(c.shows)}</span></button></li>`).join("");
  $$("#cases button").forEach((b) => b.addEventListener("click", () => {
    $$("#cases button").forEach((x) => x.removeAttribute("aria-current"));
    b.setAttribute("aria-current", "true");
    const c = META.demo_cases.find((x) => x.id === b.dataset.case);
    launch({ case_id: c.id }, c.title);
  }));
}

function renderForm() {
  $("#f-order").innerHTML = META.orders.map((o) => `<option value="${esc(o.order_id)}">${esc(o.order_id)} · ${esc(o.priority)} · ${esc(o.pickup)} → ${esc(o.dropoff)}</option>`).join("");
  $("#f-persona").innerHTML = Object.entries(META.personas).map(([k, p]) => `<option value="${k}">${esc(p.label)}</option>`).join("");
  $("#f-hour").innerHTML = Array.from({ length: 18 }, (_, i) => i + 6).map((h) => `<option value="${h}"${h === 14 ? " selected" : ""}>${String(h).padStart(2, "0")}:00</option>`).join("");
  const fill = () => {
    const o = META.orders.find((x) => x.order_id === $("#f-order").value);
    const tpl = META.personas[$("#f-persona").value].template;
    $("#f-message").value = tpl.replace("{order_id}", o.order_id).replace("{weight}", o.weight_kg).replace("{pickup}", o.pickup)
      .replace("{dropoff}", o.dropoff).replace("{deadline}", o.deadline_mins).replace("{priority}", o.priority).replace("{notes}", o.notes);
  };
  $("#f-order").addEventListener("change", fill);
  $("#f-persona").addEventListener("change", fill);
  fill();
  $("#run-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const fd = Object.fromEntries(new FormData(e.target));
    fd.hour_of_day = Number(fd.hour_of_day);
    launch(fd, `${fd.order_id} · ${META.personas[fd.persona].label}`);
  });
}

// ---------------------------------------------------------------- replay a stored run
async function replay(runId) {
  selectTab("dispatch");
  const data = await api(`/api/runs/${encodeURIComponent(runId)}`);
  const run = data.run;
  resetLive(`Replay · ${run.order_id || "custom"} · ${run.persona} · ${run.architecture}`, run.input_text, runId);
  live.t0 = performance.now();
  data.events.forEach(onEvent);
  onDone(data);
}

// ---------------------------------------------------------------- tabs
function selectTab(name) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tab-panel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
  if (name === "runs") loadRuns();
  if (name === "evaluation") loadEvaluation();
}

$$(".tabs button").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));
$("#filters").addEventListener("click", (e) => {
  const kind = e.target.dataset?.kind;
  if (!kind) return;
  live.filter = kind;
  $$("#filters button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.kind === kind)));
  $$("#log li").forEach((li) => (li.hidden = kind !== "all" && li.dataset.kind !== kind));
});

// ---------------------------------------------------------------- runs tab
async function loadRuns() {
  const rows = await api("/api/runs");
  const head = `<thead><tr><th>Run</th><th>Source</th><th>Architecture</th><th>Persona</th><th>Style</th><th>Event</th><th>Order</th><th>Outcome</th><th>Latency</th><th>Tokens</th><th>Judge</th></tr></thead>`;
  const body = rows.map((r) => {
    const rep = r.report || {};
    const j = r.judge?.overall != null ? fmt(r.judge.overall, 2) : "–";
    return `<tr data-run="${esc(r.run_id)}"><td class="mono">${esc(r.run_id)}</td><td>${esc(r.source)}</td><td>${esc(r.architecture)}</td>
      <td>${esc(r.persona)}</td><td>${esc(r.prompt_style)}</td><td>${esc(r.scenario)}</td><td>${esc(r.order_id || "–")}</td>
      <td><span class="outcome ${esc(rep.outcome || r.status)}">${esc(rep.outcome || r.status)}</span></td>
      <td class="mono">${rep.total_latency_s != null ? fmt(rep.total_latency_s) + " s" : "–"}</td>
      <td class="mono">${rep.total_tokens ?? "–"}</td><td class="mono">${j}</td></tr>`;
  }).join("");
  $("#runs-table").innerHTML = head + `<tbody>${body || `<tr><td colspan="11">No runs yet.</td></tr>`}</tbody>`;
  $$("#runs-table tbody tr[data-run]").forEach((tr) => tr.addEventListener("click", () => replay(tr.dataset.run)));
}

// ---------------------------------------------------------------- evaluation tab
async function loadEvaluation() {
  const ev = await api("/api/evaluation");
  if (!ev.available) {
    $("#evaluation").innerHTML = `<p class="notice">The evaluation has not been run on this server yet. Run <code>uv run python -m src.evaluation.benchmark_runner</code> to produce it.</p>`;
    return;
  }
  const k = ev.headline;
  $("#evaluation").innerHTML = `
    <div class="section-head"><h2>Evaluation of ${k.runs} runs</h2>
      <p class="hint">${esc(ev.design)}</p></div>
    <div class="kpis">
      <div><b>${k.runs}</b><span>runs</span></div>
      <div><b>${fmt(k.sensitivity * 100, 0)}%</b><span>attack sensitivity</span></div>
      <div><b>${fmt(k.specificity * 100, 0)}%</b><span>clean-request specificity</span></div>
      <div><b>${fmt(k.judge_mean, 2)}</b><span>LLM-judge mean (1-5)</span></div>
      <div><b>${fmt(k.p50_latency_s)} s</b><span>median latency</span></div>
      <div><b>$${fmt(k.mean_cost_usd, 4)}</b><span>mean cost per run</span></div>
    </div>
    <div class="figs">${ev.figures.map((f) => `<figure><img src="/artifacts/${esc(f.file)}" alt="${esc(f.title)}" loading="lazy">
      <figcaption><b>${esc(f.title)}.</b> ${esc(f.caption)}</figcaption></figure>`).join("")}</div>
    <h2 class="section-title">By architecture</h2>
    <div class="table-wrap">${table(ev.by_architecture)}</div>`;
}

function table(rows) {
  if (!rows?.length) return "";
  const cols = Object.keys(rows[0]);
  return `<table class="grid-table"><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${cols.map((c) => `<td>${esc(typeof r[c] === "number" ? +r[c].toFixed(3) : r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

// ---------------------------------------------------------------- how-it-works tab
const AGENTS = [
  { name: "Lead Supervisor", fw: "CrewAI", why: "Turns a messy customer message into a structured job, then signs off or sends work back (Lab-03 manager loop).",
    reads: "whole blackboard", writes: "request, review", tools: "none (delegates)" },
  { name: "Retrieval Agent", fw: "CrewAI + ChromaDB", why: "Rules change by zone and time; it looks them up instead of trusting model memory, and cites the source file.",
    reads: "request", writes: "compliance", tools: "search_policies (MCP)" },
  { name: "Dispatch Agent", fw: "CrewAI · ReAct", why: "Picks the driver by weighing on-time odds (ML), empty drive and fatigue; explains why the runner-up lost.",
    reads: "request, compliance, candidates", writes: "dispatch", tools: "rank_drivers, get_driver (MCP)" },
  { name: "Route Agent", fw: "Google ADK · A2A service", why: "Runs in its own service. Computes the corridor with Dijkstra and the delay band with the Random Forest.",
    reads: "request, dispatch", writes: "route", tools: "plan_route, forecast_delay (MCP)" },
  { name: "Exception Agent", fw: "CrewAI · event-driven", why: "Sleeps until a road closure, breakdown or rain spike is posted, then reroutes over A2A, reassigns, or escalates.",
    reads: "request, dispatch, route, disruption", writes: "exception", tools: "rank_drivers (MCP), request_reroute_via_a2a" },
];

const CRITERIA = [
  ["1", "Blackboard shared state", "Dispatch tab → Blackboard pane; every write shows agent + version", "src/orchestration/blackboard_manager.py"],
  ["2", "Tools through MCP (5 tools)", "Event log → MCP filter; Fleet & data tab", "src/mcp/server.py"],
  ["3", "Pydantic structured output", "Final delivery report; every blackboard section", "src/schemas/decisions.py"],
  ["4", "Hierarchical vs sequential vs parallel", "Custom run → Architecture; Evaluation → by architecture", "src/orchestration/"],
  ["5", "Callbacks, monitoring, per-agent heatmap", "Per-agent heat table; Evaluation heatmaps", "src/observability/callbacks.py"],
  ["6", "A2A between CrewAI and ADK", "Event log → A2A filter; Agent Card in Fleet & data", "src/agents/a2a_client.py"],
  ["7", "CrewAI and ADK only", "Framework label on every pipeline stage", "src/agents/"],
  ["8", "Guardrails and prompt variance", "Abusive, injection and paraphrased-attack cases", "src/guardrails/"],
  ["9", "Observability / provenance", "Runs tab → replay any run's full trace", "src/observability/tracing.py"],
  ["10", "Multi-run evaluation, cost, boxplots", "Evaluation tab", "src/evaluation/benchmark_runner.py"],
  ["11", "Cloud deployment", "This page is served from the cloud container", "Dockerfile"],
];

function renderHow() {
  $("#arch").innerHTML = `${archSvg()}
    <div class="agents">${AGENTS.map((a) => `<article><h3>${a.name}</h3><span class="fw">${a.fw}</span><p>${a.why}</p>
      <dl><dt>reads</dt><dd>${a.reads}</dd><dt>writes</dt><dd>${a.writes}</dd><dt>tools</dt><dd>${a.tools}</dd></dl></article>`).join("")}</div>`;
  $("#criteria").innerHTML = `<thead><tr><th>#</th><th>Requirement</th><th>Where to see it in the demo</th><th>Code</th></tr></thead>
    <tbody>${CRITERIA.map((c) => `<tr><td>${c[0]}</td><td>${c[1]}</td><td>${c[2]}</td><td class="mono">${c[3]}</td></tr>`).join("")}</tbody>`;
}

function archSvg() {
  const box = (x, y, w, h, t, s, accent) => `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="${accent ? "#fef2f2" : "#f9fafb"}" stroke="${accent ? "#991b1b" : "#d1d5db"}"/>
    <text x="${x + w / 2}" y="${y + h / 2 - 3}" text-anchor="middle" font-weight="600" font-size="13" fill="#111827">${t}</text>
    <text x="${x + w / 2}" y="${y + h / 2 + 13}" text-anchor="middle" font-size="10.5" fill="#4b5563">${s}</text></g>`;
  const arrow = (x1, y1, x2, y2, label = "", dash = false) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#111827" stroke-width="1.3" ${dash ? 'stroke-dasharray="5 4"' : ""} marker-end="url(#ah)"/>
    ${label ? `<text x="${(x1 + x2) / 2 + 4}" y="${(y1 + y2) / 2 - 4}" font-size="10" fill="#991b1b" font-family="IBM Plex Mono">${label}</text>` : ""}`;
  return `<svg class="arch-svg" viewBox="0 0 1040 420" role="img" aria-label="System architecture">
    <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#111827"/></marker></defs>
    ${box(20, 175, 120, 56, "Customer", "free-text message")}
    ${box(170, 175, 120, 56, "Input Guard", "regex + Gemini judge", true)}
    ${box(320, 175, 150, 56, "Lead Supervisor", "CrewAI · intake/review")}
    ${box(520, 30, 170, 56, "Retrieval Agent", "CrewAI · RAG")}
    ${box(520, 120, 170, 56, "Dispatch Agent", "CrewAI · ReAct")}
    ${box(520, 230, 170, 56, "Exception Agent", "CrewAI · event-driven")}
    ${box(760, 175, 170, 56, "Route Agent", "Google ADK service", true)}
    ${box(320, 330, 370, 64, "Blackboard (shared state)", "request · compliance · candidates · dispatch · route · exception · review")}
    ${box(760, 30, 260, 92, "MCP server", "5 tools: RAG · routing · ML delay · fleet DB")}
    ${box(760, 330, 260, 64, "Observability store", "events · OTel spans · runs (SQLite)")}
    ${arrow(140, 203, 170, 203)}${arrow(290, 203, 320, 203)}
    ${arrow(470, 190, 520, 60, "delegate")}${arrow(470, 200, 520, 148)}${arrow(470, 215, 520, 255, "on event")}
    ${arrow(690, 148, 760, 195, "A2A")}${arrow(690, 258, 760, 213, "A2A")}
    ${arrow(690, 58, 760, 70, "MCP", true)}${arrow(690, 140, 760, 100, "MCP", true)}${arrow(845, 175, 845, 122, "MCP", true)}
    ${arrow(395, 231, 395, 330, "read/write", true)}${arrow(605, 286, 605, 330, "", true)}${arrow(845, 231, 845, 330, "trace", true)}
  </svg>`;
}

// ---------------------------------------------------------------- data tab
function renderData() {
  const mc = META.model_card;
  const drivers = META.drivers.map(({ driver_id, name, current_zone, vehicle_type, capacity_kg, status, hours_worked, rating }) =>
    ({ driver_id, name, current_zone, vehicle_type, capacity_kg, status, hours_worked, rating }));
  const orders = META.orders.map(({ notes, persona, ...o }) => o);
  $("#data").innerHTML = `
    <div class="two">
      <div><h2 class="section-title" style="margin-top:0">Fleet database (MCP: rank_drivers, get_driver)</h2><div class="table-wrap">${table(drivers)}</div></div>
      <div><h2 class="section-title" style="margin-top:0">Delay model card (MCP: forecast_delay)</h2>
        ${mc ? `<div class="kpis"><div><b>${fmt(mc.holdout.r2, 3)}</b><span>holdout R²</span></div><div><b>${fmt(mc.holdout.mae, 2)}</b><span>MAE (min)</span></div>
        <div><b>${fmt(mc.holdout.p10_p90_coverage * 100, 1)}%</b><span>P10–P90 coverage</span></div><div><b>${mc.rows}</b><span>training rows</span></div></div>
        <div class="table-wrap">${table(Object.entries(mc.cv_comparison).map(([m, v]) => ({ model: m, ...v })))}</div>` : "<p class='hint'>Model card not found.</p>"}
        <p class="hint" style="margin-top:8px">RAG corpus: ${META.rag.files} policy files, ${META.rag.chunks} chunks (ChromaDB, all-MiniLM-L6-v2).</p></div>
    </div>
    <h2 class="section-title">Orders</h2><div class="table-wrap">${table(orders)}</div>
    <h2 class="section-title">A2A Agent Card of the Route Agent (live from the ADK service)</h2>
    <pre class="json">${esc(JSON.stringify(META.agent_card || { error: "service not reachable" }, null, 2))}</pre>`;
}

// ---------------------------------------------------------------- health + boot
async function refreshHealth() {
  try {
    const h = await api("/api/health");
    $$("#status li").forEach((li) => { li.className = h[li.dataset.svc] ? "up" : "down"; });
  } catch { $$("#status li").forEach((li) => (li.className = "down")); }
}

async function boot() {
  renderPipeline();
  renderMap();
  refreshHealth();
  setInterval(refreshHealth, 15000);
  META = await api("/api/meta");
  renderCases();
  renderForm();
  renderHow();
  renderData();
  const runParam = new URLSearchParams(location.search).get("run");
  if (runParam) replay(runParam);
}

boot().catch((err) => {
  $("#run-title").textContent = "Could not load the dispatch desk";
  $("#run-input").textContent = err.message;
});
