// DeliveryFleet web app: customer booking + control room. Vanilla ES module, no build step.

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n, d = 1) => (n === null || n === undefined || Number.isNaN(Number(n)) ? "–" : Number(n).toFixed(d));
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`);
  return body;
}

let META = null;

// =================================================================== city map
// Zone hubs in a 640 x 460 drawing. Distances on the roads are the real network values.
const HUB = { "Zone A": [160, 130], "Zone B": [480, 130], "Zone C": [160, 330], "Zone D": [480, 330] };
const AREA = { "Zone A": [0, 0, "zone-a", "CBD & residential"], "Zone B": [320, 0, "zone-b", "Tech corridor"],
  "Zone C": [0, 230, "zone-c", "University & civic"], "Zone D": [320, 230, "zone-d", "Airport cargo"] };
const ROADS = [["Zone A", "Zone B", 14.5], ["Zone A", "Zone C", 18], ["Zone A", "Zone D", 26.5],
  ["Zone B", "Zone C", 12], ["Zone B", "Zone D", 21], ["Zone C", "Zone D", 16.5]];
const key2 = (a, b) => [a, b].sort().join("|");
const zoneOf = (loc) => (String(loc || "").match(/Zone [A-D]/) || [])[0];

function hash(s) { let h = 2166136261; for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return h >>> 0; }
function locPoint(loc) {
  const z = zoneOf(loc); const [hx, hy] = HUB[z];
  const h = hash(loc); const ang = (h % 360) * Math.PI / 180; const r = 34 + (h >> 9) % 30;
  return [Math.round(hx + Math.cos(ang) * r), Math.round(hy + Math.sin(ang) * r * 0.75)];
}
const zonesIn = (corridor) => String(corridor || "").split("->").map((s) => s.trim()).filter((s) => /^Zone [A-D]/.test(s)).map((s) => s.slice(0, 6));

function pathPoints(from, to, zones) {
  if (!from || !to) return [];
  const hops = zones?.length ? zones : [zoneOf(from), zoneOf(to)];
  const pts = [locPoint(from)];
  if (new Set(hops).size > 1) hops.forEach((z) => pts.push(HUB[z]));
  pts.push(locPoint(to));
  return pts;
}

function drawCity(svg, { from, to, primary, detour, closed, label = true } = {}) {
  // Labels sit in each zone's outer corner, away from the roads that cross the middle.
  const areas = Object.entries(AREA).map(([z, [x, y, cls, sub]]) => {
    const right = x > 0, bottom = y > 0;
    const tx = right ? x + 296 : x + 24, anchor = right ? "end" : "start";
    const ty = bottom ? y + 196 : y + 32;
    return `<rect class="zone-area" x="${x + 6}" y="${y + 6}" width="308" height="218" rx="18" fill="var(--${cls})"/>
    ${label ? `<text class="zone-name" x="${tx}" y="${ty}" text-anchor="${anchor}">${z}</text><text class="zone-sub" x="${tx}" y="${ty + 18}" text-anchor="${anchor}">${sub}</text>` : ""}`;
  }).join("");
  const roads = ROADS.map(([a, b, km]) => {
    const [x1, y1] = HUB[a], [x2, y2] = HUB[b];
    const diag = x1 !== x2 && y1 !== y2; const t = diag ? 0.32 : 0.5;
    return `<line class="road" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/><line class="road-mid" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>
      <text class="km" x="${x1 + (x2 - x1) * t + 8}" y="${y1 + (y2 - y1) * t - 8}">${km} km</text>`;
  }).join("");
  const hubs = Object.values(HUB).map(([x, y]) => `<circle class="hub" cx="${x}" cy="${y}" r="7"/>`).join("");
  const poly = (pts, cls) => pts.length ? `<polyline class="${cls}" points="${pts.map((p) => p.join(",")).join(" ")}"/>` : "";
  let overlay = "";
  if (closed) {
    const [a, b] = closed.split("->").map((s) => s.trim());
    if (HUB[a] && HUB[b]) {
      const [x1, y1] = HUB[a], [x2, y2] = HUB[b]; const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      overlay += `<line class="closed" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>
        <path class="closed-x" d="M${mx - 9} ${my - 9} L${mx + 9} ${my + 9} M${mx + 9} ${my - 9} L${mx - 9} ${my + 9}"/>`;
    }
  }
  const route = from && to ? poly(pathPoints(from, to, primary), "route") : "";
  const det = from && to && detour ? poly(pathPoints(from, to, detour), "route detour") : "";
  const pin = (loc, cls, letter) => {
    if (!loc) return "";
    const [x, y] = locPoint(loc); const name = loc.split(" - ").slice(1).join(" - ");
    return `<g class="pt ${cls}"><circle class="ring" cx="${x}" cy="${y}" r="9"/><circle cx="${x}" cy="${y}" r="3.5" fill="${cls === "to" ? "var(--crimson)" : "var(--ink)"}"/>
      <text x="${x + 14}" y="${y + 4}">${esc(letter)} · ${esc(name)}</text></g>`;
  };
  svg.innerHTML = areas + roads + hubs + route + overlay + det + pin(from, "from", "Pickup") + pin(to, "to", "Drop") +
    `<g id="${svg.id}-rider" hidden><circle class="rider-pulse" r="18"/><circle class="rider-dot" r="8"/></g>`;
}

// Move the rider marker along a polyline; resolves when it arrives.
function animateRider(svg, pts, ms, stopAt = 1) {
  const g = $(`#${svg.id}-rider`, svg);
  if (!g || pts.length < 2) return Promise.resolve();
  g.hidden = false;
  const segs = pts.slice(1).map((p, i) => Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]));
  const total = segs.reduce((a, b) => a + b, 0);
  const place = (f) => {
    let d = f * total, i = 0;
    while (i < segs.length - 1 && d > segs[i]) { d -= segs[i]; i++; }
    const t = segs[i] ? d / segs[i] : 0;
    const x = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, y = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t;
    g.setAttribute("transform", `translate(${x.toFixed(1)} ${y.toFixed(1)})`);
  };
  if (reduceMotion) { place(stopAt); return Promise.resolve(); }
  return new Promise((done) => {
    const t0 = performance.now();
    const step = (now) => {
      const f = Math.min((now - t0) / ms, 1) * stopAt;
      place(f);
      if (f < stopAt) requestAnimationFrame(step); else done();
    };
    requestAnimationFrame(step);
  });
}

// =================================================================== tabs
function selectTab(name) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".panel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
  if (name === "runs") loadRuns();
  if (name === "evaluation") loadEvaluation();
  history.replaceState(null, "", `#${name}`);
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));
$$("[data-go]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); selectTab(a.dataset.go); }));

// =================================================================== event stream (shared)
function stream(runId, onEvent, onDone) {
  const src = new EventSource(`/api/runs/${encodeURIComponent(runId)}/stream`);
  src.addEventListener("agent_event", (e) => onEvent(JSON.parse(e.data)));
  src.addEventListener("done", (e) => { src.close(); onDone(JSON.parse(e.data)); });
  src.addEventListener("timeout", () => src.close());
  return src;
}

// =================================================================== BOOK (customer)
const book = { runId: null, src: null, booking: null };

function quote() {
  const f = new FormData($("#book-form"));
  const from = f.get("pickup"), to = f.get("dropoff"), w = Number(f.get("weight_kg"));
  const urg = f.get("urgency");
  const sameZone = zoneOf(from) === zoneOf(to);
  const km = sameZone ? 4 : (ROADS.find(([a, b]) => key2(a, b) === key2(zoneOf(from), zoneOf(to)))?.[2] || 15) + 3;
  const fare = Math.round(39 + km * 9 + Math.max(0, w - 10) * 1.5 + { standard: 0, express: 40, critical: 90 }[urg]);
  $("#b-quote").innerHTML = `<span>Estimated fare (demo)</span><b>₹${fare}</b>`;
  drawCity($("#bmap"), { from, to });
  book.fare = fare;
}

function initBook() {
  const locs = META.locations;
  const opts = (sel) => locs.map((l) => `<option value="${esc(l)}"${l === sel ? " selected" : ""}>${esc(l.replace(" - ", " · "))}</option>`).join("");
  $("#b-pickup").innerHTML = opts("Zone A - Warehouse 1");
  $("#b-dropoff").innerHTML = opts("Zone B - Tech Park");
  $("#b-hour").innerHTML = Array.from({ length: 18 }, (_, i) => i + 6).map((h) => `<option value="${h}"${h === 14 ? " selected" : ""}>${String(h).padStart(2, "0")}:00</option>`).join("");
  $("#b-weight").addEventListener("input", (e) => { $("#b-weight-out").textContent = `${e.target.value} kg`; quote(); });
  $("#book-form").addEventListener("change", quote);
  $("#book-form").addEventListener("submit", submitBooking);
  $("#t-new").addEventListener("click", resetBooking);
  $("#t-inside").addEventListener("click", () => book.runId && openInControl(book.runId));
  quote();
}

function showView(name) { $$(".sheet-view").forEach((v) => (v.hidden = v.dataset.view !== name)); }

function resetBooking() {
  book.src?.close();
  Object.assign(book, { runId: null, src: null, booking: null });
  showView("form");
  $("#b-submit").disabled = false;
  $("#bmap-legend").hidden = true;
  quote();
}

async function submitBooking(e) {
  e.preventDefault();
  const f = Object.fromEntries(new FormData(e.target));
  if (f.pickup === f.dropoff) { showBookError("Pickup and drop are the same place. Choose a different drop."); return; }
  f.weight_kg = Number(f.weight_kg); f.hour_of_day = Number(f.hour_of_day);
  $("#b-submit").disabled = true; $("#b-error").hidden = true;
  try {
    const res = await api("/api/bookings", { method: "POST", body: JSON.stringify(f) });
    book.runId = res.run_id; book.booking = { ...f, ...res };
    startTracking();
  } catch (err) {
    showBookError(err.message);
    $("#b-submit").disabled = false;
  }
}
function showBookError(msg) { $("#b-error").textContent = msg; $("#b-error").hidden = false; }

const STEP_OF = { input_guard: "guard", retrieval_agent: "rules", dispatch_agent: "match", route_agent: "route" };

function startTracking() {
  const b = book.booking;
  showView("track");
  $("#t-booking").textContent = `${b.booking_id} · ${b.priority} · deliver within ${b.deadline_mins} min`;
  $("#t-title").textContent = "Finding your rider";
  $("#t-sub").textContent = `${b.pickup.split(" - ")[1]} to ${b.dropoff.split(" - ")[1]}, ${b.weight_kg} kg`;
  $$("#t-finding li").forEach((li, i) => (li.className = i === 0 ? "now" : ""));
  $("#t-finding").hidden = false;
  ["#t-rider", "#t-notice", "#t-timeline"].forEach((s) => ($(s).hidden = true));
  $("#t-inside").hidden = false;
  book.src = stream(book.runId, onBookEvent, onBookDone);
}

function markStep(step, cls) {
  const li = $(`#t-finding [data-step="${step}"]`);
  if (!li) return;
  const order = ["guard", "rules", "match", "route"];
  order.slice(0, order.indexOf(step)).forEach((s) => { const x = $(`#t-finding [data-step="${s}"]`); if (x.className !== "bad") x.className = "ok"; });
  li.className = cls;
}

function onBookEvent(ev) {
  if (ev.kind === "guardrail" && ev.agent === "input_guard") markStep("guard", ev.status === "BLOCKED" ? "bad" : "ok");
  if (ev.kind === "agent_start" && STEP_OF[ev.agent]) markStep(STEP_OF[ev.agent], "now");
  if (ev.kind === "agent_end" && STEP_OF[ev.agent]) markStep(STEP_OF[ev.agent], ev.status === "OK" ? "ok" : "bad");
  if (ev.kind === "a2a_message" && ev.payload?.direction === "request") markStep("route", "now");
}

async function onBookDone({ run }) {
  const r = run.report || {}, bb = run.blackboard || {}, b = book.booking;
  const req = bb.request || {}, route = bb.route || {}, exc = bb.exception || {}, dis = bb.disruption || {}, disp = bb.dispatch || {};
  const from = req.pickup || b.pickup, to = req.dropoff || b.dropoff;

  if (r.outcome === "BLOCKED") {
    markStep("guard", "bad");
    $("#t-title").textContent = "We could not place this booking";
    $("#t-sub").textContent = "Your note contained instructions our system will not follow. Please rebook with delivery details only.";
    return;
  }
  if (r.outcome === "FAILED") {
    $("#t-title").textContent = "Something went wrong";
    $("#t-sub").textContent = r.customer_message;
    return;
  }
  $$("#t-finding li").forEach((li) => { if (li.className !== "bad") li.className = "ok"; });
  setTimeout(() => ($("#t-finding").hidden = true), reduceMotion ? 0 : 700);

  if (r.outcome === "ESCALATED" && !r.driver_id) {
    $("#t-title").textContent = "A dispatcher will call you";
    $("#t-sub").textContent = "No rider can legally take this parcel right now.";
    showNotice("bad", "Needs a human dispatcher", r.customer_message);
    drawCity($("#bmap"), { from, to });
    return;
  }

  const driver = META.drivers.find((d) => d.driver_id === (disp.driver_id || r.driver_id)) || {};
  const late = route.eta_mins > (req.deadline_mins || b.deadline_mins);
  $("#t-title").textContent = r.outcome === "ESCALATED" ? "Rider found, deadline at risk" : "Rider assigned";
  $("#t-sub").textContent = r.outcome === "ESCALATED" ? "A dispatcher is reviewing this delivery with you." : "Your parcel is on its way to pickup.";
  renderRider(driver, route.eta_mins, late, disp.p_on_time);
  drawCity($("#bmap"), { from, to, primary: zonesIn(route.corridor) });
  legend(false);

  const steps = [["Booked", b.booking_id], ["Rider assigned", `${driver.name || disp.driver_name} · ${driver.vehicle_type || disp.vehicle_type}`],
    ["Picked up", from.split(" - ")[1]], ["In transit", route.corridor], ["Delivered", to.split(" - ")[1]]];
  renderTimeline(steps, 1);
  const pts = pathPoints(from, to, zonesIn(route.corridor));

  await wait(900); renderTimeline(steps, 2);
  if (dis.kind) {
    await animateRider($("#bmap"), pts, 3500, 0.22);
    renderTimeline(steps, 3);
    const newDriver = exc.new_driver_id && exc.new_driver_id !== disp.driver_id
      ? META.drivers.find((d) => d.driver_id === exc.new_driver_id) : null;
    showNotice(exc.action === "ESCALATE" ? "bad" : "warn",
      { CORRIDOR_CLOSED: "Road closed on your route", VEHICLE_BREAKDOWN: "Your rider's vehicle broke down", WEATHER_SPIKE: "Heavy rain on your route" }[dis.kind] || "Delay on your route",
      exc.customer_update || dis.detail);
    drawCity($("#bmap"), { from, to, primary: zonesIn(route.corridor), closed: dis.closed_corridor,
      detour: exc.new_corridor ? zonesIn(exc.new_corridor) : null });
    legend(true);
    if (newDriver) renderRider(newDriver, exc.new_eta_mins, exc.new_eta_mins > (req.deadline_mins || b.deadline_mins), null, true);
    else if (exc.new_eta_mins) $("#t-eta b").textContent = Math.round(exc.new_eta_mins);
    if (exc.action !== "ESCALATE") {
      await wait(600);
      await animateRider($("#bmap"), pathPoints(from, to, zonesIn(exc.new_corridor || route.corridor)), 6500);
      renderTimeline(steps, 5);
      $("#t-title").textContent = "Delivered";
      $("#t-sub").textContent = `Recovered after the disruption. Fare ₹${book.fare}.`;
    }
  } else {
    renderTimeline(steps, 3);
    await animateRider($("#bmap"), pts, 7000);
    renderTimeline(steps, 5);
    $("#t-title").textContent = r.outcome === "ESCALATED" ? "Delivered, late" : "Delivered";
    $("#t-sub").textContent = `${r.customer_message} Fare ₹${book.fare}.`;
  }
}

function renderRider(d, eta, late, p, reassigned = false) {
  const initials = (d.name || "?").split(" ").map((w) => w[0]).join("").slice(0, 2);
  $("#t-rider").innerHTML = `
    <div class="avatar" aria-hidden="true">${esc(initials)}</div>
    <div><h2>${esc(d.name || "Rider")}${reassigned ? ' <span class="tag warn">new rider</span>' : ""}</h2>
      <p><span>${esc(d.vehicle_type || "")}</span><span class="rating"><svg><use href="#i-star"/></svg>${fmt(d.rating, 2)}</span>
      ${p != null ? `<span>${Math.round(p * 100)}% on-time likelihood</span>` : ""}</p></div>
    <div class="eta${late ? " late" : ""}" id="t-eta"><b>${eta ? Math.round(eta) : "–"}</b><span>min to drop</span></div>`;
  $("#t-rider").hidden = false;
}

function renderTimeline(steps, reached) {
  $("#t-timeline").innerHTML = steps.map(([t, s], i) =>
    `<li class="${i < reached - 1 ? "done" : i === reached - 1 ? "now" : ""}">${esc(t)}<small>${esc(s || "")}</small></li>`).join("");
  $("#t-timeline").hidden = false;
}

function showNotice(kind, title, text) {
  const n = $("#t-notice");
  n.className = `notice${kind === "bad" ? " bad" : ""}`;
  n.innerHTML = `<svg><use href="#i-alert"/></svg><div><b>${esc(title)}</b>${esc(text || "")}</div>`;
  n.hidden = false;
}

function legend(withDetour) {
  const l = $("#bmap-legend");
  l.innerHTML = `<span><i style="background:var(--ink)"></i>planned route</span>` +
    (withDetour ? `<span><i style="background:var(--crimson)"></i>closed</span><span><i style="background:var(--green)"></i>new route</span>` : "");
  l.hidden = false;
}
const wait = (ms) => new Promise((r) => setTimeout(r, reduceMotion ? 0 : ms));

// =================================================================== CONTROL ROOM
const STAGES = [
  { id: "input_guard", a: "Input guard", f: "Regex + Gemini judge", short: "GUARD" },
  { id: "intake", a: "Supervisor", f: "CrewAI · intake", short: "SUPERVISOR" },
  { id: "retrieval_agent", a: "Retrieval", f: "CrewAI · ChromaDB RAG", short: "RETRIEVAL" },
  { id: "driver_db", a: "Driver DB", f: "MCP tool", short: "FLEET DB" },
  { id: "dispatch_agent", a: "Dispatch", f: "CrewAI · ReAct", short: "DISPATCH" },
  { id: "route_agent", a: "Route", f: "Google ADK · A2A", short: "ROUTE" },
  { id: "business", a: "Rule check", f: "Code guardrail", short: "RULES" },
  { id: "exception_agent", a: "Exception", f: "CrewAI · on event", short: "EXCEPTION" },
  { id: "review", a: "Supervisor", f: "CrewAI · review", short: "REVIEW" },
];
const live = { runId: null, src: null, t0: 0, timer: null, filter: "all", events: [], board: {}, started: {} };

function renderCounters() {
  $("#pipeline").innerHTML = STAGES.map((s) => `
    <li data-stage="${s.id}" data-state="idle"><span class="a">${s.a}</span><span class="f">${s.f}</span><span class="st">Waiting</span></li>`).join("");
}

function postmark(stage, label, tone) {
  const t = new Date().toTimeString().slice(0, 8);
  const id = `pm-${stage.id}-${Math.random().toString(36).slice(2, 7)}`;
  const rot = (hash(stage.id + t) % 17) - 11;
  return `<svg class="postmark land" viewBox="0 0 60 60" style="--rot:${rot}deg" aria-hidden="true"${tone === "bad" ? ' data-bad="1"' : ""}>
    <defs><path id="${id}" d="M30 30 m-21 0 a21 21 0 1 1 42 0 a21 21 0 1 1 -42 0"/></defs>
    <circle cx="30" cy="30" r="27" fill="none" stroke="currentColor" stroke-width="1.6"/>
    <circle cx="30" cy="30" r="15" fill="none" stroke="currentColor" stroke-width="1"/>
    <text font-size="6.4" font-weight="700" letter-spacing="1.2"><textPath href="#${id}">${esc(stage.short)} · ${esc(label)}</textPath></text>
    <text x="30" y="32.5" text-anchor="middle" font-size="7" font-weight="700">${t.slice(0, 5)}</text>
  </svg>`;
}

function setStage(id, state, label) {
  const li = $(`#pipeline [data-stage="${id}"]`);
  if (!li) return;
  const rank = { idle: 0, running: 1, done: 2, skipped: 2, blocked: 3 };
  if (rank[state] < rank[li.dataset.state] && state !== "running") return;
  const was = li.dataset.state;
  li.dataset.state = state;
  $(".st", li).textContent = label;
  if ((state === "done" || state === "blocked") && was !== state) {
    $(".postmark", li)?.remove();
    li.insertAdjacentHTML("beforeend", postmark(STAGES.find((s) => s.id === id), state === "done" ? "OK" : "STOP", state === "blocked" ? "bad" : ""));
  }
}

function resetLive(title, input, runId) {
  live.src?.close(); clearInterval(live.timer);
  Object.assign(live, { runId, events: [], board: {}, t0: performance.now() });
  renderCounters();
  drawCity($("#map"), {});
  $("#run-id").textContent = runId || "No run yet";
  $("#run-title").textContent = title;
  $("#run-input").innerHTML = highlight(input || "");
  $("#log").innerHTML = `<li class="empty">Events appear here as each agent works.</li>`;
  $("#board").innerHTML = `<div class="empty">Sections appear as agents write them.</div>`;
  $("#heat").innerHTML = "";
  $("#report").innerHTML = `<div class="slip-head"><b>Consignment slip</b><span>filled when the Supervisor signs off</span></div>`;
  $("#map-note").textContent = "corridor chosen by the Route Agent";
  $("#run-clock").textContent = "0.0 s";
}

function highlight(text) {
  const risky = /(ignore all previous instructions[^.]*|assign driver d09 without checking hours|pretend the shift-hour policy was cancelled[^.]*|incompetent|idiots|lazy|clowns|\[REDACTED[^\]]*\])/gi;
  return esc(text).replace(risky, (m) => `<mark>${m}</mark>`);
}

function tick() {
  clearInterval(live.timer);
  live.timer = setInterval(() => ($("#run-clock").textContent = `${((performance.now() - live.t0) / 1000).toFixed(1)} s`), 100);
}

function supervisorStage() { return live.board.dispatch || live.board.business_check ? "review" : "intake"; }

function onControlEvent(ev) {
  if (!live.events.length) $("#log").innerHTML = "";
  live.events.push(ev);
  const p = ev.payload || {};
  if (ev.kind === "guardrail" && ev.agent === "input_guard") {
    const blocked = ev.status === "BLOCKED";
    setStage("input_guard", blocked ? "blocked" : "done", { ALLOW: "Clean", SANITIZED: "Masked & allowed", BLOCKED: "Blocked" }[ev.status] || ev.status);
    if (blocked) STAGES.slice(1).forEach((s) => setStage(s.id, "skipped", "Not run"));
    if (ev.status === "SANITIZED" && p.sanitized_text) $("#run-input").innerHTML = highlight(p.sanitized_text);
  }
  if (ev.kind === "agent_start") setStage(ev.agent === "supervisor" ? supervisorStage() : ev.agent, "running", "Working");
  if (ev.kind === "agent_end") {
    const id = ev.agent === "supervisor" ? supervisorStage() : ev.agent;
    setStage(id, ev.status === "OK" ? "done" : "blocked", ev.status === "OK" ? `${fmt(ev.duration_ms / 1000)} s` : "Error");
  }
  if (ev.kind === "tool_call" && ev.name === "rank_drivers" && ev.agent === "supervisor") setStage("driver_db", ev.status === "OK" ? "done" : "blocked", "10 drivers scored");
  if (ev.kind === "a2a_message" && p.direction === "request" && !live.board.route) setStage("route_agent", "running", "A2A sent");
  if (ev.kind === "blackboard" && ev.status === "WRITE") boardWrite(ev.name, p);
  if (ev.kind === "supervisor_rerun") setStage(ev.name, "running", "Sent back");
  if (ev.kind === "error") setStage(ev.agent === "supervisor" ? "review" : ev.agent, "blocked", "Error");
  appendLog(ev);
}

function boardWrite(section, prov) {
  live.board[section] = prov;
  $("#board .empty")?.remove();
  let row = $(`#board [data-sec="${section}"]`);
  if (!row) { row = document.createElement("div"); row.dataset.sec = section; $("#board").append(row); }
  row.innerHTML = `<dt>${esc(section)}</dt><dd><b>${esc(prov.agent)}</b> · v${prov.version}${prov.summary ? ` · ${esc(prov.summary)}` : ""}</dd>`;
  row.classList.remove("flash"); void row.offsetWidth; row.classList.add("flash");
  if (section === "business_check") setStage("business", (prov.summary || "").includes("no legal") ? "blocked" : "done", (prov.summary || "").includes("no legal") ? "No legal driver" : "Checked");
  if (section === "disruption") { setStage("exception_agent", "running", "Woken by event"); $("#map-note").textContent = prov.summary || ""; }
}

const KIND = { llm_call: "LLM", tool_call: "MCP", a2a_message: "A2A", blackboard: "BOARD", guardrail: "GUARD", agent_start: "START",
  agent_end: "END", error: "ERROR", supervisor_rerun: "RERUN", p2p_message: "P2P", broadcast: "BROADCAST", context_size: "CONTEXT" };

function describe(ev) {
  const p = ev.payload || {};
  switch (ev.kind) {
    case "llm_call": return `${esc(ev.name)} · ${ev.input_tokens} in / ${ev.output_tokens} out tokens · ${fmt(ev.duration_ms, 0)} ms`;
    case "tool_call": return `<b>${esc(ev.name)}</b>${ev.duration_ms ? ` · ${fmt(ev.duration_ms, 0)} ms` : ""}${ev.status !== "OK" ? " · failed" : ""}`;
    case "a2a_message": return `${esc(p.from)} to ${esc(p.to)} · ${esc(p.direction)}${ev.duration_ms ? ` · ${fmt(ev.duration_ms, 0)} ms` : ""}`;
    case "blackboard": return `${ev.status === "WRITE" ? "wrote" : "read"} <b>${esc(ev.name)}</b> v${p.version}${p.summary ? ` · ${esc(p.summary)}` : ""}`;
    case "guardrail": return `${esc(ev.name.replaceAll("_", " "))} <span class="tag ${ev.status === "BLOCKED" ? "bad" : ev.status === "SANITIZED" ? "warn" : "ok"}">${esc(ev.status)}</span>`;
    case "agent_start": return `started`;
    case "agent_end": return `finished in ${fmt(ev.duration_ms / 1000, 2)} s`;
    case "supervisor_rerun": return `sent the work back to <b>${esc(ev.name)}</b>`;
    case "error": return `<b>${esc(ev.name)}</b>: ${esc(p.error)}`;
    default: return esc(ev.name);
  }
}

function appendLog(ev) {
  const li = document.createElement("li");
  li.dataset.kind = ev.kind;
  const t = ev.ts - live.events[0].ts;
  const detail = ev.payload && Object.keys(ev.payload).length ? `<details><summary>details</summary><pre>${esc(JSON.stringify(ev.payload, null, 1))}</pre></details>` : "";
  li.innerHTML = `<span class="t">+${t.toFixed(1)}s</span><span class="who">${esc(ev.agent || "system")}<small>${esc(ev.framework)}</small></span>
    <span class="what"><span class="k ${esc(ev.kind)}">${KIND[ev.kind] || esc(ev.kind)}</span>${describe(ev)}${detail}</span>`;
  li.hidden = live.filter !== "all" && live.filter !== ev.kind;
  const log = $("#log");
  const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 40;
  log.append(li);
  if (stick) log.scrollTop = log.scrollHeight;
}

function onControlDone({ run, agents }) {
  clearInterval(live.timer);
  const r = run.report || {}, bb = run.blackboard || {};
  $("#run-clock").textContent = `${fmt(r.total_latency_s)} s`;
  STAGES.forEach((s) => {
    const li = $(`#pipeline [data-stage="${s.id}"]`);
    if (li && ["idle", "running"].includes(li.dataset.state)) setStage(s.id, "skipped", s.id === "exception_agent" ? "No event" : "Not run");
  });
  const req = bb.request || {}, route = bb.route || {}, exc = bb.exception || {}, dis = bb.disruption || {};
  drawCity($("#map"), { from: req.pickup, to: req.dropoff, primary: zonesIn(route.corridor), closed: dis.closed_corridor,
    detour: exc.new_corridor ? zonesIn(exc.new_corridor) : null });
  if (exc.new_corridor) $("#map-note").textContent = `black: planned · red dashed: closed · green: detour via ${exc.new_corridor}`;
  renderSlip(r, bb);
  renderHeat(agents);
}

const OUTCOME_TONE = { DISPATCHED: "ok", RECOVERED_AFTER_DISRUPTION: "ok", ESCALATED: "warn", BLOCKED: "bad", FAILED: "bad" };
const OUTCOME_TEXT = { DISPATCHED: "Dispatched", RECOVERED_AFTER_DISRUPTION: "Recovered", ESCALATED: "Escalated to human", BLOCKED: "Blocked", FAILED: "Failed" };

function renderSlip(r, bb) {
  const cells = [
    ["Outcome", `<span class="tag ${OUTCOME_TONE[r.outcome] || "idle"}">${esc(OUTCOME_TEXT[r.outcome] || r.outcome)}</span>`],
    ["Order", esc(r.order_id || "–")],
    ["Driver", r.driver_id ? `${esc(r.driver_id)} · ${esc(r.driver_name)}` : "–"],
    ["ETA / deadline", r.eta_mins ? `<span class="big">${Math.round(r.eta_mins)}</span> / ${esc(r.deadline_mins)} min` : "–", true],
    ["Corridor", esc(r.corridor || "–")],
    ["On-time likelihood", r.p_on_time != null ? `${Math.round(r.p_on_time * 100)}%` : "–"],
    ["Input guard", esc(r.guard_status)],
    ["Exception plan", esc(r.exception_action || "None")],
    ["Supervisor", esc(r.supervisor_verdict || "–")],
    ["Architecture", esc(r.architecture)],
    ["LLM calls · tokens", `${r.llm_calls} · ${Number(r.total_tokens).toLocaleString("en-IN")}`],
    ["Cost", `$${fmt(r.cost_usd, 5)}`],
  ];
  const why = bb.business_check?.violations?.length ? `<p class="slip-why">Rule check: ${esc(bb.business_check.violations.join("; "))}</p>` : "";
  $("#report").innerHTML = `<div class="slip-head"><b>Consignment slip</b><span class="mono">${esc(r.run_id)}</span></div>
    <dl class="slip-body">${cells.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>
    <p class="slip-msg">${esc(r.customer_message)}</p>${why}`;
}

const HEAT_COLS = [["latency_ms", "Time", (v) => `${fmt(v / 1000)} s`], ["llm_calls", "LLM", (v) => v], ["tool_calls", "Tools", (v) => v],
  ["input_tokens", "Tokens in", (v) => Number(v).toLocaleString("en-IN")], ["cost_usd", "Cost $", (v) => fmt(v, 5)]];
function renderHeat(agents) {
  const names = Object.keys(agents || {});
  if (!names.length) { $("#heat").innerHTML = `<p class="dim" style="font-size:.85rem">No agent activity in this run.</p>`; return; }
  const max = Object.fromEntries(HEAT_COLS.map(([k]) => [k, Math.max(...names.map((n) => agents[n][k] || 0), 1e-9)]));
  $("#heat").innerHTML = `<table class="heat"><thead><tr><th>Agent</th>${HEAT_COLS.map(([, l]) => `<th>${l}</th>`).join("")}</tr></thead><tbody>
    ${names.map((n) => `<tr><td>${esc(n)}</td>${HEAT_COLS.map(([k, , f]) => {
      const s = (agents[n][k] || 0) / max[k];
      return `<td style="background:rgba(153,27,27,${(0.04 + s * 0.62).toFixed(2)});color:${s > 0.62 ? "#fff" : "inherit"}">${f(agents[n][k] || 0)}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>`;
}

function initControl() {
  const sel = $("#c-scenario");
  sel.innerHTML = META.demo_cases.map((c) => `<option value="${esc(c.id)}">${esc(c.title)}</option>`).join("") +
    `<option value="__custom">Custom message (type your own)</option>`;
  const note = () => {
    const c = META.demo_cases.find((x) => x.id === sel.value);
    $("#c-custom-row").hidden = sel.value !== "__custom";
    $("#c-note").textContent = c ? c.shows.replaceAll("->", "→").replaceAll("||", "∥") : "Type any customer message. It goes through the same guard and agents.";
  };
  sel.addEventListener("change", note); note();
  $("#scenario-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target));
    const body = { architecture: f.architecture, prompt_style: f.prompt_style };
    if (f.case_id === "__custom") Object.assign(body, { persona: "custom", message: $("#c-custom").value.trim() });
    else body.case_id = f.case_id;
    const ev = $("#c-event").value; if (ev) body.scenario = ev;
    $("#c-run").disabled = true;
    try {
      const { run_id, message } = await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
      const title = f.case_id === "__custom" ? "Custom message" : sel.selectedOptions[0].textContent;
      resetLive(title, message, run_id); tick();
      live.src = stream(run_id, onControlEvent, onControlDone);
    } catch (err) {
      resetLive("Could not start the run", err.message, "");
    } finally { setTimeout(() => ($("#c-run").disabled = false), 1500); }
  });
  $("#filters").addEventListener("click", (e) => {
    const kind = e.target.dataset?.kind; if (!kind) return;
    live.filter = kind;
    $$("#filters button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.kind === kind)));
    $$("#log li[data-kind]").forEach((li) => (li.hidden = kind !== "all" && li.dataset.kind !== kind));
  });
  renderCounters();
  drawCity($("#map"), {});
}

async function openInControl(runId) {
  selectTab("control");
  const data = await api(`/api/runs/${encodeURIComponent(runId)}`);
  const run = data.run;
  resetLive(`${run.source === "booking" ? "Customer booking" : "Replay"} · ${run.order_id || "custom"} · ${run.architecture}`, run.input_text, runId);
  data.events.forEach(onControlEvent);
  if (run.status === "RUNNING") {
    tick();
    const last = data.events.at(-1)?.id || 0;
    live.src = new EventSource(`/api/runs/${encodeURIComponent(runId)}/stream`);
    live.src.addEventListener("agent_event", (e) => { const ev = JSON.parse(e.data); if (ev.id > last) onControlEvent(ev); });
    live.src.addEventListener("done", (e) => { live.src.close(); onControlDone(JSON.parse(e.data)); });
  } else onControlDone(data);
}

// =================================================================== RUNS
async function loadRuns() {
  const src = $("#runs-source").value;
  const rows = await api(`/api/runs${src ? `?source=${src}` : ""}`);
  const head = `<thead><tr><th>Run</th><th>Source</th><th>Architecture</th><th>Persona</th><th>Style</th><th>Event</th><th>Order</th><th>Outcome</th><th class="num">Time</th><th class="num">Tokens</th><th class="num">Judge</th></tr></thead>`;
  const body = rows.map((r) => {
    const rep = r.report || {};
    return `<tr data-run="${esc(r.run_id)}" tabindex="0"><td class="mono">${esc(r.run_id)}</td><td>${esc(r.source)}</td><td>${esc(r.architecture)}</td>
      <td>${esc(r.persona)}</td><td>${esc(r.prompt_style || "–")}</td><td>${esc(r.scenario)}</td><td>${esc(r.order_id || "–")}</td>
      <td><span class="tag ${OUTCOME_TONE[rep.outcome] || "idle"}">${esc(OUTCOME_TEXT[rep.outcome] || r.status)}</span></td>
      <td class="num">${rep.total_latency_s != null ? fmt(rep.total_latency_s) + " s" : "–"}</td>
      <td class="num">${rep.total_tokens != null ? Number(rep.total_tokens).toLocaleString("en-IN") : "–"}</td>
      <td class="num">${r.judge?.overall != null ? fmt(r.judge.overall, 2) : "–"}</td></tr>`;
  }).join("");
  $("#runs-table").innerHTML = head + `<tbody>${body || `<tr><td colspan="11" class="dim">No runs yet. Book a delivery or run a scenario.</td></tr>`}</tbody>`;
  $$("#runs-table tr[data-run]").forEach((tr) => {
    tr.addEventListener("click", () => openInControl(tr.dataset.run));
    tr.addEventListener("keydown", (e) => e.key === "Enter" && openInControl(tr.dataset.run));
  });
}
$("#runs-source").addEventListener("change", loadRuns);

// =================================================================== EVALUATION
async function loadEvaluation() {
  const ev = await api("/api/evaluation");
  if (!ev.available) {
    $("#evaluation").innerHTML = `<div class="panel-head"><h2>Evaluation</h2></div><p class="notice-box">The 72-run evaluation has not been run on this server yet. It is produced by <code class="mono">uv run python -m src.evaluation.benchmark_runner</code>.</p>`;
    return;
  }
  const k = ev.headline;
  $("#evaluation").innerHTML = `
    <div class="panel-head"><h2>Evaluation across ${k.runs} runs</h2><p>${esc(ev.design)}</p></div>
    <div class="kpis">
      <div><b>${k.runs}</b><span>runs</span></div>
      <div><b>${fmt(k.sensitivity * 100, 0)}%</b><span>attacks caught (sensitivity)</span></div>
      <div><b>${fmt(k.specificity * 100, 0)}%</b><span>clean requests served (specificity)</span></div>
      <div><b>${fmt(k.judge_mean, 2)}</b><span>LLM-judge mean, 1 to 5</span></div>
      <div><b>${fmt(k.p50_latency_s)} s</b><span>median time per run</span></div>
      <div><b>$${fmt(k.mean_cost_usd, 4)}</b><span>mean cost per run</span></div>
    </div>
    <div class="figs">${ev.figures.map((f) => `<figure><img src="/artifacts/${esc(f.file)}" alt="${esc(f.title)}" loading="lazy">
      <figcaption><b>${esc(f.title)}.</b> ${esc(f.caption)}</figcaption></figure>`).join("")}</div>
    <h3 class="h3">By architecture</h3><div class="table-wrap">${table(ev.by_architecture)}</div>`;
}

function table(rows) {
  if (!rows?.length) return "";
  const cols = Object.keys(rows[0]);
  return `<table class="table"><thead><tr>${cols.map((c) => `<th>${esc(c.replaceAll("_", " "))}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${cols.map((c) => `<td${typeof r[c] === "number" ? ' class="num"' : ""}>${esc(typeof r[c] === "number" ? +r[c].toFixed(3) : r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

// =================================================================== DATA
function renderData() {
  const mc = META.model_card;
  const drivers = META.drivers.map(({ driver_id, name, current_zone, vehicle_type, capacity_kg, status, hours_worked, rating }) =>
    ({ driver_id, name, zone: current_zone, vehicle: vehicle_type, capacity_kg, status, hours_worked, rating }));
  const orders = META.orders.map(({ notes, persona, ...o }) => o);
  $("#data").innerHTML = `
    <div class="panel-head"><h2>Fleet and data</h2><p>All data is synthetic, made for this project. Agents reach it only through MCP tools.</p></div>
    <div class="two">
      <div><h3 class="h3" style="margin-top:0">Fleet database <span class="dim" style="font-weight:400;font-size:.85rem">rank_drivers, get_driver</span></h3><div class="table-wrap">${table(drivers)}</div></div>
      <div><h3 class="h3" style="margin-top:0">Delay model <span class="dim" style="font-weight:400;font-size:.85rem">forecast_delay</span></h3>
        ${mc ? `<div class="kpis"><div><b>${fmt(mc.holdout.r2, 3)}</b><span>holdout R²</span></div><div><b>${fmt(mc.holdout.mae, 2)}</b><span>mean error, min</span></div>
          <div><b>${fmt(mc.holdout.p10_p90_coverage * 100, 1)}%</b><span>trips inside P10–P90 band</span></div><div><b>${mc.rows}</b><span>training rows</span></div></div>
          <div class="table-wrap">${table(Object.entries(mc.cv_comparison).map(([m, v]) => ({ model: m.replaceAll("_", " "), ...v })))}</div>` : `<p class="dim">Model card not found.</p>`}
        <p class="dim" style="margin-top:10px;font-size:.88rem">Policy knowledge base: ${META.rag.files} files, ${META.rag.chunks} chunks in ChromaDB (all-MiniLM-L6-v2 embeddings).</p></div>
    </div>
    <h3 class="h3">Orders</h3><div class="table-wrap">${table(orders)}</div>
    <h3 class="h3">A2A Agent Card of the Route Agent <span class="dim" style="font-weight:400;font-size:.85rem">served live by the ADK service</span></h3>
    <pre class="json">${esc(JSON.stringify(META.agent_card || { error: "route service not reachable" }, null, 2))}</pre>`;
}

// =================================================================== HOW IT WORKS
const AGENTS = [
  { name: "Lead Supervisor", fw: "CrewAI", why: "Turns the customer's message into a structured job, then signs off or sends work back to one worker (Lab-03 manager loop).", reads: "whole blackboard", writes: "request, review", tools: "none, delegates" },
  { name: "Retrieval Agent", fw: "CrewAI + ChromaDB", why: "Zone and labour rules change by place and time; it looks them up and cites the file instead of trusting model memory.", reads: "request", writes: "compliance", tools: "search_policies" },
  { name: "Dispatch Agent", fw: "CrewAI, ReAct", why: "Chooses the driver by weighing on-time odds from the delay model, the empty drive to pickup, and fatigue.", reads: "request, compliance, candidates", writes: "dispatch", tools: "rank_drivers, get_driver" },
  { name: "Route Agent", fw: "Google ADK, A2A service", why: "Runs in its own service. Finds the corridor with Dijkstra and the delay band with the Random Forest.", reads: "request, dispatch", writes: "route", tools: "plan_route, forecast_delay" },
  { name: "Exception Agent", fw: "CrewAI, event-driven", why: "Sleeps until a road closure, breakdown or rain spike is posted, then reroutes over A2A, reassigns, or escalates.", reads: "request, dispatch, route, disruption", writes: "exception", tools: "rank_drivers, request_reroute_via_a2a" },
];
const CRITERIA = [
  ["1", "Blackboard shared state", "Control room, Blackboard panel: each write shows agent and version", "src/orchestration/blackboard_manager.py"],
  ["2", "Tools through MCP", "Event log, MCP filter; Fleet & data tab", "src/mcp/server.py"],
  ["3", "Pydantic structured output", "Consignment slip; every blackboard section", "src/schemas/decisions.py"],
  ["4", "Hierarchical, sequential, parallel", "Control room, Architecture dropdown; Evaluation tab", "src/orchestration/"],
  ["5", "Callbacks, monitoring, per-agent heatmap", "Per-agent heat; Evaluation heatmaps", "src/observability/callbacks.py"],
  ["6", "A2A between CrewAI and ADK", "Event log, A2A filter; Agent Card in Fleet & data", "src/agents/a2a_client.py"],
  ["7", "CrewAI and ADK only", "Framework under every agent in the control room", "src/agents/"],
  ["8", "Guardrails and prompt variance", "Abusive, injection and paraphrased-attack scenarios", "src/guardrails/"],
  ["9", "Observability and provenance", "Runs tab: replay any run's full trace", "src/observability/tracing.py"],
  ["10", "Multi-run evaluation, cost, boxplots", "Evaluation tab", "src/evaluation/benchmark_runner.py"],
  ["11", "Cloud deployment", "This page is served from the cloud container", "Dockerfile"],
];
function renderHow() {
  const box = (x, y, w, h, t, s, accent) => `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${accent ? "#fbeeee" : "#f7f8f9"}" stroke="${accent ? "#991b1b" : "#d1d5db"}"/>
    <text x="${x + w / 2}" y="${y + h / 2 - 3}" text-anchor="middle" font-weight="700" font-size="14" fill="#111827">${t}</text>
    <text x="${x + w / 2}" y="${y + h / 2 + 14}" text-anchor="middle" font-size="11.5" fill="#4b5563">${s}</text></g>`;
  const arrow = (x1, y1, x2, y2, label = "", dash = false) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#111827" stroke-width="1.4" ${dash ? 'stroke-dasharray="5 4"' : ""} marker-end="url(#ah)"/>
    ${label ? `<text x="${(x1 + x2) / 2 + 5}" y="${(y1 + y2) / 2 - 5}" font-size="11" font-weight="600" fill="#991b1b">${label}</text>` : ""}`;
  $("#arch").innerHTML = `<svg class="arch-svg" viewBox="0 0 1040 420" role="img" aria-label="System architecture" font-family="Archivo, sans-serif">
    <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#111827"/></marker></defs>
    ${box(20, 175, 120, 58, "Customer", "booking or message")}${box(170, 175, 120, 58, "Input guard", "regex + Gemini judge", true)}
    ${box(320, 175, 150, 58, "Lead Supervisor", "CrewAI · intake, review")}
    ${box(520, 28, 170, 58, "Retrieval Agent", "CrewAI · RAG")}${box(520, 120, 170, 58, "Dispatch Agent", "CrewAI · ReAct")}
    ${box(520, 232, 170, 58, "Exception Agent", "CrewAI · on event")}${box(760, 175, 170, 58, "Route Agent", "Google ADK service", true)}
    ${box(320, 330, 370, 64, "Blackboard (shared state)", "request · compliance · candidates · dispatch · route · exception · review")}
    ${box(760, 28, 260, 92, "MCP tool server", "RAG · routing · delay model · fleet DB")}
    ${box(760, 330, 260, 64, "Trace store", "events · OpenTelemetry spans · runs")}
    ${arrow(140, 204, 170, 204)}${arrow(290, 204, 320, 204)}${arrow(470, 190, 520, 58, "delegate")}${arrow(470, 200, 520, 150)}${arrow(470, 216, 520, 260, "on event")}
    ${arrow(690, 150, 760, 196, "A2A")}${arrow(690, 260, 760, 214, "A2A")}${arrow(690, 58, 760, 70, "MCP", true)}${arrow(690, 140, 760, 100, "MCP", true)}${arrow(845, 175, 845, 120, "MCP", true)}
    ${arrow(395, 233, 395, 330, "read / write", true)}${arrow(605, 290, 605, 330, "", true)}${arrow(845, 233, 845, 330, "trace", true)}
  </svg>
  <div class="agents">${AGENTS.map((a) => `<article><h3>${a.name}</h3><span class="fw">${a.fw}</span><p>${a.why}</p>
    <dl><dt>reads</dt><dd>${a.reads}</dd><dt>writes</dt><dd>${a.writes}</dd><dt>tools</dt><dd>${a.tools}</dd></dl></article>`).join("")}</div>`;
  $("#criteria").innerHTML = `<thead><tr><th>#</th><th>Requirement</th><th>Where to see it</th><th>Code</th></tr></thead>
    <tbody>${CRITERIA.map((c) => `<tr><td>${c[0]}</td><td>${c[1]}</td><td>${c[2]}</td><td class="mono">${c[3]}</td></tr>`).join("")}</tbody>`;
}

// =================================================================== boot
async function refreshHealth() {
  try {
    const h = await api("/api/health");
    $$("#status li").forEach((li) => (li.className = h[li.dataset.svc] ? "up" : "down"));
  } catch { $$("#status li").forEach((li) => (li.className = "down")); }
}

async function boot() {
  refreshHealth(); setInterval(refreshHealth, 15000);
  META = await api("/api/meta");
  initBook(); initControl(); renderHow(); renderData();
  const params = new URLSearchParams(location.search);
  const tab = location.hash.slice(1);
  if (params.get("run")) openInControl(params.get("run"));
  else if (tab && $(`#tab-${tab}`)) selectTab(tab);
}
addEventListener("hashchange", () => { const t = location.hash.slice(1); if ($(`#tab-${t}`)) selectTab(t); });
boot().catch((err) => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="notice-box">Could not load the app: ${esc(err.message)}</p>`); });
