"use strict";

const POLL_MS = 5000;
const TABLES = ["performance", "accounts", "risk", "positions", "exposure", "open_risk", "movers", "wallets"];
const sortState = {};           // table id -> {key, dir}
let lastRendered = null;        // last_ok of the data currently drawn
let latest = null;              // latest /api/state document

// ---------------------------------------------------------------- formatting
const usdFormat = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function isNum(value) { return typeof value === "number" && isFinite(value); }

function usd(value) {
  if (!isNum(value)) return "";
  return (value < 0 ? "-$" : "$") + usdFormat.format(Math.abs(value));
}
function signedUsd(value) {
  if (!isNum(value)) return "";
  return (value < 0 ? "-$" : value > 0 ? "+$" : "$") + usdFormat.format(Math.abs(value));
}
function pct(value, digits = 1, signed = false) {
  if (!isNum(value)) return "";
  const text = (value * 100).toFixed(digits) + "%";
  return signed && value > 0 ? "+" + text : text;
}
function number(value) {
  if (!isNum(value)) return "";
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(3);
  return Number(value.toPrecision(6)).toLocaleString("en-US", { maximumFractionDigits: 6 });
}
function compactUsd(value) {
  if (!isNum(value)) return "n/a";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-$" : "$";
  if (abs >= 1e6) return sign + (abs / 1e6).toFixed(2) + "M";
  if (abs >= 1e4) return sign + (abs / 1e3).toFixed(1) + "K";
  return sign + usdFormat.format(abs);
}

const FORMATS = {
  usd: { text: usd, num: true },
  susd: { text: signedUsd, num: true, signed: true },
  pct: { text: (v) => pct(v, 1), num: true },
  spct: { text: (v) => pct(v, 2, true), num: true, signed: true },
  num: { text: number, num: true },
  num2: { text: (v) => (isNum(v) ? v.toFixed(2) : ""), num: true },
  int: { text: (v) => (isNum(v) ? Math.round(v).toLocaleString("en-US") : ""), num: true },
  text: { text: (v) => (v === null || v === undefined ? "" : String(v)), num: false },
};

function signClass(value) {
  if (!isNum(value) || value === 0) return "";
  return value > 0 ? "pos" : "neg";
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "text") node.textContent = value;
    else if (key === "class") node.className = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of [].concat(children)) {
    if (child !== null && child !== undefined) node.append(child);
  }
  return node;
}

function ago(iso, now) {
  if (!iso) return "never";
  const seconds = Math.max(0, (now - new Date(iso)) / 1000);
  if (seconds < 45) return "just now";
  if (seconds < 90) return "1 min ago";
  if (seconds < 3600) return Math.round(seconds / 60) + " min ago";
  if (seconds < 86400) return Math.round(seconds / 3600) + " h ago";
  return Math.round(seconds / 86400) + " d ago";
}
function clock(iso) {
  return iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
}
function day(iso) { return iso ? String(iso).slice(0, 10) : ""; }

// ---------------------------------------------------------------- tables
function renderTable(id, spec) {
  const host = document.getElementById("t-" + id);
  if (!host) return;
  host.replaceChildren();
  if (!spec || !spec.columns || !spec.columns.length) {
    host.append(el("p", { class: "empty-table", text: "Nothing to show." }));
    return;
  }
  const state = sortState[id];
  let rows = spec.rows.slice();
  if (state) {
    rows.sort((a, b) => {
      const x = a[state.key], y = b[state.key];
      if (isNum(x) && isNum(y)) return state.dir * (x - y);
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      return state.dir * String(x).localeCompare(String(y));
    });
  }
  const head = el("tr", {}, spec.columns.map((column) => {
    const format = FORMATS[column.fmt] || FORMATS.text;
    const sorted = state && state.key === column.key ? (state.dir > 0 ? "ascending" : "descending") : "none";
    const th = el("th", { class: format.num ? "num" : "", "aria-sort": sorted, tabindex: "0", text: column.key });
    const toggle = () => {
      const current = sortState[id];
      const dir = current && current.key === column.key ? -current.dir : (format.num ? -1 : 1);
      sortState[id] = { key: column.key, dir };
      renderTable(id, spec);
    };
    th.addEventListener("click", toggle);
    th.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); toggle(); } });
    return th;
  }));
  const cell = (row, column) => {
    const format = FORMATS[column.fmt] || FORMATS.text;
    const value = row[column.key];
    const classes = [format.num ? "num" : "", format.signed ? signClass(value) : ""].filter(Boolean).join(" ");
    return el("td", { class: classes, text: format.text(value) });
  };
  const body = el("tbody", {}, rows.map((row) => el("tr", {}, spec.columns.map((column) => cell(row, column)))));
  const parts = [el("thead", {}, head), body];
  if (spec.total) {
    parts.push(el("tfoot", {}, el("tr", { class: "total" }, spec.columns.map((column) => cell(spec.total, column)))));
  }
  host.append(el("table", {}, parts));
}

// ---------------------------------------------------------------- tiles
function tile(label, value, sub, extra) {
  return el("div", { class: "tile" }, [
    el("div", { class: "label", text: label }),
    el("div", { class: "value " + (extra && extra.valueClass ? extra.valueClass : ""), text: value }),
    extra && extra.meter ? extra.meter : null,
    sub ? el("div", { class: "sub", text: sub }) : null,
  ]);
}

function renderTiles(k) {
  const r = k.returns || {};
  const tiles = [
    tile("Desk value", compactUsd(k.desk_value),
      `Trading ${compactUsd(k.trading_equity)} · wallets ${compactUsd(k.wallet_value)}`),
    tile("Unrealized PnL", signedUsd(k.unrealized),
      isNum(k.unrealized_share) ? `${pct(k.unrealized_share, 1, true)} of ${compactUsd(k.gross)} in ${k.positions} positions` : "",
      { valueClass: signClass(k.unrealized) }),
    tile("Return 30D", pct(r["30D"], 1, true),
      `7D ${pct(r["7D"], 1, true) || "n/a"} · 90D ${pct(r["90D"], 1, true) || "n/a"}`,
      { valueClass: signClass(r["30D"]) }),
    tile("Return since start", pct(r.since_start, 1, true), `Net of deposits, since ${day(k.since)}`,
      { valueClass: signClass(r.since_start) }),
    tile("Drawdown now", pct(k.drawdown, 1), k.drawdown < 0 ? `Below the peak of ${day(k.peak_date)}` : "At the peak",
      { valueClass: k.drawdown <= -0.10 ? "neg" : "" }),
    tile("Gross leverage", isNum(k.gross_leverage) ? k.gross_leverage.toFixed(2) + "×" : "n/a",
      isNum(k.net_leverage) ? `Net ${k.net_leverage >= 0 ? "+" : ""}${k.net_leverage.toFixed(2)}× trading equity` : ""),
  ];
  if (isNum(k.risk_used) && isNum(k.risk_cap) && k.risk_cap > 0) {
    const share = k.risk_used / k.risk_cap;
    const level = share > 1 ? "critical" : share > 0.8 ? "warning" : "";
    const meter = el("div", { class: "meter " + level, role: "meter", "aria-valuemin": "0", "aria-valuemax": "100",
      "aria-valuenow": String(Math.round(share * 100)), "aria-label": "Open risk as a share of the cap" },
      el("span", { style: `width:${Math.min(100, share * 100).toFixed(1)}%` }));
    tiles.push(tile("Open risk vs cap", `${pct(share, 0)} of cap`,
      `${usd(k.risk_used)} of ${usd(k.risk_cap)} · next position ${usd(k.position_budget)}`, { meter }));
  }
  if (k.largest) {
    tiles.push(tile("Largest position", `${k.largest.symbol}`,
      `${String(k.largest.side || "").toLowerCase()} ${compactUsd(k.largest.value)} · ${pct(k.largest.share, 0)} of desk · ${k.largest.account}`));
  }
  document.getElementById("tiles").replaceChildren(...tiles);
}

function renderAccountStatus(rows, now) {
  const host = document.getElementById("accounts-status");
  host.replaceChildren(...(rows || []).map((row) => el("span", { class: "chip", title: row.problem || "" }, [
    el("span", { class: "icon " + (row.ok ? "good" : "bad"), "aria-hidden": "true", text: row.ok ? "✓" : "!" }),
    el("span", { text: `${row.name}: ${row.ok ? "read" : "unreadable" + (row.last_seen ? ", last read " + ago(row.last_seen, now) : "")}` }),
  ])));
}

function renderHighlights(id, rows) {
  const host = document.getElementById(id);
  if (!rows || !rows.length) {
    host.replaceChildren(el("li", {}, el("span", { class: "empty", text: "Nothing to report." })));
    return;
  }
  host.replaceChildren(...rows.map(([area, text]) => el("li", {}, [
    el("span", { class: "area", text: area }), el("span", { class: "text", text }),
  ])));
}

// ---------------------------------------------------------------- charts
function token(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function withAlpha(hex, alpha) {
  const value = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(value.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${alpha})`;
}

function baseLayout(extra) {
  const ink = token("--ink-2"), muted = token("--muted"), grid = token("--grid"), axis = token("--axis");
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", size: 12, color: ink },
    margin: { l: 56, r: 72, t: 8, b: 36 },
    hovermode: "x unified",
    hoverlabel: { bgcolor: token("--surface"), bordercolor: axis, font: { color: token("--ink") } },
    legend: { orientation: "h", x: 0, y: 1.08, font: { color: ink } },
    xaxis: { gridcolor: grid, linecolor: axis, tickfont: { color: muted }, showspikes: true,
             spikemode: "across", spikethickness: 1, spikecolor: axis, spikedash: "solid" },
    yaxis: { gridcolor: grid, linecolor: axis, zerolinecolor: axis, tickfont: { color: muted } },
  }, extra);
}

const CONFIG = { displayModeBar: false, responsive: true };

function endLabels(series, format) {
  return series.filter((s) => s.y.length).map((s) => ({
    x: s.x[s.x.length - 1], y: s.y[s.y.length - 1], xanchor: "left", xshift: 6, showarrow: false,
    text: format(s.y[s.y.length - 1]), font: { color: token("--ink-2"), size: 11 },
  }));
}

function renderCharts(charts) {
  if (typeof Plotly === "undefined" || !charts) return;
  const colors = [1, 2, 3, 4, 5].map((i) => token("--series-" + i));
  const value = (charts.value || []).map((s, i) => ({
    type: "scatter", mode: "lines", name: s.name, x: s.x, y: s.y, stackgroup: "value",
    line: { width: 2, color: colors[i % colors.length] }, fillcolor: withAlpha(colors[i % colors.length], 0.18),
    hovertemplate: "%{fullData.name}: $%{y:,.2f}<extra></extra>",
  }));
  Plotly.react("chart-value", value, baseLayout({ yaxis: Object.assign(baseLayout().yaxis, { tickprefix: "$", separatethousands: true }) }), CONFIG);

  const twr = (charts.twr || []).map((s, i) => ({
    type: "scatter", mode: "lines", name: s.name, x: s.x, y: s.y,
    line: { width: 2, color: colors[i % colors.length], shape: "linear" },
    hovertemplate: "%{fullData.name}: %{y:.1f}<extra></extra>",
  }));
  const labels = endLabels(charts.twr || [], (v) => v.toFixed(1));
  Plotly.react("chart-twr", twr, baseLayout({
    annotations: labels,
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 100, y1: 100, line: { color: token("--axis"), width: 1 } }],
  }), CONFIG);
}

// ---------------------------------------------------------------- page
function renderBanners(doc, now) {
  const host = document.getElementById("banners");
  const banners = [];
  const status = doc.status || {};
  const data = doc.data;
  if (data && data.problems && data.problems.length) {
    banners.push(el("div", { class: "banner critical", role: "alert" }, [
      el("span", { class: "icon bad", "aria-hidden": "true", text: "!" }),
      el("div", {}, [el("strong", { text: "Unreadable accounts are left out of every total. " }),
        el("span", { text: data.problems.join(" · ") })]),
    ]));
  }
  if (status.error) {
    banners.push(el("div", { class: "banner warning", role: "alert" }, [
      el("span", { class: "icon warn", "aria-hidden": "true", text: "!" }),
      el("div", {}, [el("strong", { text: `Last refresh failed (${clock(status.last_attempt)}). ` }),
        el("span", { text: `${status.error}. ` + (status.last_ok ? `Showing data from ${ago(status.last_ok, now)}.` : "") })]),
    ]));
  } else if (status.last_ok && (now - new Date(status.last_ok)) / 1000 > 3 * (status.interval_s || 180) + 120) {
    banners.push(el("div", { class: "banner warning", role: "alert" }, [
      el("span", { class: "icon warn", "aria-hidden": "true", text: "!" }),
      el("div", { text: `Data is ${ago(status.last_ok, now)}; the refresh may be stuck.` }),
    ]));
  }
  host.replaceChildren(...banners);
}

function renderStatus(doc, now) {
  const status = doc.status || {};
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  const button = document.getElementById("refresh");
  button.disabled = !!status.refreshing;
  if (status.refreshing) {
    dot.className = "dot busy";
    text.textContent = status.last_ok ? `Refreshing… (updated ${ago(status.last_ok, now)})` : "Reading accounts…";
  } else if (status.error) {
    dot.className = "dot error";
    text.textContent = `Refresh failed · data ${ago(status.last_ok, now)}`;
  } else if (status.last_ok) {
    dot.className = "dot live";
    const next = new Date(new Date(status.last_ok).getTime() + (status.interval_s || 180) * 1000);
    const wait = Math.max(0, Math.round((next - now) / 60000));
    text.textContent = `Live · updated ${ago(status.last_ok, now)} · next ${wait < 1 ? "within a minute" : "in " + wait + " min"}`;
  } else {
    dot.className = "dot";
    text.textContent = "Waiting for the first refresh…";
  }
}

function render(doc) {
  const now = new Date(doc.server_time || Date.now());
  renderStatus(doc, now);
  renderBanners(doc, now);
  const data = doc.data;
  if (!data) return;
  document.getElementById("loading").hidden = true;
  document.getElementById("content").hidden = false;
  renderAccountStatus(data.accounts_status, now);
  if (lastRendered === doc.status.last_ok) return;   // same data: keep tables, sort and hover as they are
  lastRendered = doc.status.last_ok;
  renderTiles(data.kpis || {});
  renderHighlights("hl-well", data.highlights && data.highlights.well);
  renderHighlights("hl-attention", data.highlights && data.highlights.attention);
  for (const id of TABLES) renderTable(id, data.tables && data.tables[id]);
  renderCharts(data.charts);
  document.getElementById("notes").replaceChildren(...(data.notes || []).map((note) => el("li", { text: note })));
}

async function load() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    latest = await response.json();
    render(latest);
  } catch (error) {
    document.getElementById("status-dot").className = "dot error";
    document.getElementById("status-text").textContent = "Dashboard server not reachable (is python -m desk.dashboard running?)";
  }
}

function poll() {          // the one polling loop
  load().finally(() => setTimeout(poll, POLL_MS));
}

document.getElementById("refresh").addEventListener("click", async () => {
  document.getElementById("refresh").disabled = true;
  try { await fetch("/api/refresh", { method: "POST" }); } catch (error) { /* the next poll reports it */ }
  setTimeout(load, 500);   // one extra read, not a second loop
});

// Redraw charts in the other theme's colors when the system theme changes.
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (latest && latest.data) renderCharts(latest.data.charts);
});

poll();
