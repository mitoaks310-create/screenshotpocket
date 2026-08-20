/* Swing-trade screener dashboard.
 *
 * Reads the three JSON files written by `screener.cli screen` and renders them.
 * Position sizing is recomputed here rather than only server-side, so changing
 * the account inputs updates every row without regenerating the data.
 */

"use strict";

const LOT = 100;

const state = {
  screen: null,
  backtest: null,
  charts: null,
  sort: { key: "rank", dir: 1 },
  openCode: null,
};

// ------------------------------------------------------------- formatting

const yen = (v, d = 0) =>
  v == null || !isFinite(v) ? "—" : "¥" + Number(v).toLocaleString("ja-JP", {
    minimumFractionDigits: d, maximumFractionDigits: d });
const num = (v, d = 2) =>
  v == null || !isFinite(v) ? "—" : Number(v).toLocaleString("ja-JP", {
    minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 1) => (v == null || !isFinite(v) ? "—" : (v * 100).toFixed(d) + "%");
const signed = (v, d = 3) => (v == null || !isFinite(v) ? "—" : (v >= 0 ? "+" : "") + v.toFixed(d));
const oku = (v) => (v == null || !isFinite(v) ? "—" : (v / 1e8).toFixed(1) + "億");
const cls = (v) => (v == null || !isFinite(v) ? "" : v > 0 ? "pos" : v < 0 ? "neg" : "");

// ------------------------------------------------------------------ boot

async function boot() {
  try {
    const [screen, backtest, charts] = await Promise.all([
      fetchJSON("data/screen.json"),
      fetchJSON("data/backtest.json"),
      fetchJSON("data/charts.json"),
    ]);
    state.screen = screen;
    state.backtest = backtest;
    state.charts = charts;
    render();
  } catch (err) {
    const box = document.getElementById("fatal");
    box.style.display = "block";
    box.innerHTML =
      "<strong>データを読み込めませんでした。</strong><br>" +
      escapeHtml(String(err && err.message ? err.message : err)) +
      "<br><br>先に <code class=\"inline\">python -m screener.cli screen</code> を実行し、" +
      "<code class=\"inline\">python -m screener.cli serve</code> で配信してください" +
      "（file:// で直接開くと fetch がブロックされます）。";
  }
}

async function fetchJSON(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

function render() {
  renderHeader();
  renderRegime();
  renderControls();
  renderTable();
  renderValidation();
  renderFactors();
  wireTabs();
}

// ---------------------------------------------------------------- header

function renderHeader() {
  const s = state.screen;
  const badge = document.getElementById("provider-badge");
  badge.textContent = s.is_demo ? "デモ（合成データ）" : s.provider;
  badge.className = "badge" + (s.is_demo ? " demo" : "");

  document.getElementById("subtitle").textContent =
    `基準日 ${s.as_of ?? "—"} ／ 対象 ${s.universe_size.toLocaleString("ja-JP")} 銘柄 ` +
    `／ 生成 ${(s.generated_at || "").replace("T", " ").slice(0, 16)}`;

  if (s.is_demo) document.getElementById("demo-banner").classList.remove("hidden");

  // Two independent reasons to warn: the model failed validation, or it passed
  // but says today has nothing worth taking.
  const mono = state.backtest?.monotonicity;
  const best = Math.max(...s.candidates.map((c) => c.ev_r ?? -Infinity));
  const msgs = [];
  if (mono != null && mono < 0.6) {
    msgs.push(
      `<strong>検証が不合格です（単調性 ${signed(mono, 2)}）。</strong> ` +
      "スコアが高いほど成績が良いという関係が、学習外の期間では確認できませんでした。" +
      "下の期待値は信頼できる推定値として扱わないでください。"
    );
  }
  if (isFinite(best) && best <= 0) {
    msgs.push(
      `<strong>本日は期待値がプラスの候補がありません（最高 ${signed(best)}R）。</strong> ` +
      "順位は相対的な強さを示しているだけで、モデルとしては「今日は見送り」という判断です。"
    );
  }
  if (msgs.length) {
    const el = document.getElementById("edge-banner");
    el.innerHTML = msgs.join("<hr style='border:0;border-top:1px solid currentColor;opacity:.25;margin:10px 0'>");
    el.classList.remove("hidden");
  }
}

function renderRegime() {
  const r = state.screen.regime || {};
  const trendUp = r.trend === "up";
  const items = [
    {
      k: "ベンチマーク",
      v: r.trend ? (trendUp ? "上昇" : "下降") : "—",
      s: r.benchmark_close ? `${num(r.benchmark_close, 1)}（75日線 ${num(r.benchmark_sma75, 1)}）` : "",
      c: r.trend ? (trendUp ? "pos" : "neg") : "",
    },
    {
      k: "市場の広がり",
      v: pct(r.breadth_above_sma200, 0),
      s: "200日線より上にある銘柄の比率",
      c: r.breadth_above_sma200 == null ? "" : r.breadth_above_sma200 >= 0.5 ? "pos" : "neg",
    },
    { k: "スクリーニング対象", v: (r.universe_size ?? 0).toLocaleString("ja-JP"), s: "フィルタ通過銘柄数" },
    {
      k: "判定",
      v: trendUp ? "順風" : "逆風",
      s: trendUp ? "買い戦略が機能しやすい" : "買いは分が悪い局面",
      c: trendUp ? "pos" : "neg",
    },
  ];
  document.getElementById("regime-stats").innerHTML = items.map(statHTML).join("");
}

const statHTML = (i) =>
  `<div class="stat"><div class="k">${escapeHtml(i.k)}</div>` +
  `<div class="v ${i.c || ""}">${i.v}</div>` +
  `<div class="s">${escapeHtml(i.s || "")}</div></div>`;

// -------------------------------------------------------------- controls

function account() {
  const equity = Math.max(0, +document.getElementById("equity").value || 0);
  const riskPct = Math.max(0, (+document.getElementById("risk").value || 0) / 100);
  const maxPos = Math.max(0, (+document.getElementById("maxpos").value || 0) / 100);
  return { equity, riskPct, maxPos, riskYen: equity * riskPct };
}

function renderControls() {
  const a = state.screen.account;
  document.getElementById("equity").value = Math.round(a.equity_yen);
  document.getElementById("risk").value = +(a.risk_pct * 100).toFixed(2);
  document.getElementById("maxpos").value = +(a.max_position_pct * 100).toFixed(0);
  for (const id of ["equity", "risk", "maxpos"]) {
    document.getElementById(id).addEventListener("input", () => {
      updateRiskNote();
      renderTable();
    });
  }
  updateRiskNote();
}

function updateRiskNote() {
  const a = account();
  document.getElementById("risk-note").textContent =
    `1トレードあたり最大損失 ${yen(a.riskYen)} ／ 1銘柄の投資上限 ${yen(a.equity * a.maxPos)}`;
}

/* Mirror of screener.screen.position_size — kept deliberately identical so the
 * interactive numbers match what the CLI would print for the same account. */
function positionSize(entry, riskPerShare, a) {
  if (!isFinite(entry) || !isFinite(riskPerShare) || riskPerShare <= 0 || entry <= 0)
    return { shares: 0, cost: 0, risk: 0, note: "invalid" };

  let shares = Math.floor(a.riskYen / riskPerShare / LOT) * LOT;
  let note = "risk";
  const maxNotional = a.equity * a.maxPos;
  if (shares * entry > maxNotional) {
    shares = Math.floor(maxNotional / entry / LOT) * LOT;
    note = "position_cap";
  }
  if (shares <= 0) return { shares: 0, cost: 0, risk: 0, note: "min_lot_exceeds_risk" };
  return { shares, cost: shares * entry, risk: shares * riskPerShare, note };
}

// ----------------------------------------------------------------- table

const COLUMNS = [
  { key: "rank", label: "#", align: "l", fmt: (c) => c.rank },
  { key: "code", label: "コード", align: "l", fmt: (c) => `<span class="code">${c.code}</span>` },
  { key: "name", label: "銘柄名", align: "l", fmt: (c) => escapeHtml(c.name || "") },
  { key: "sector33", label: "業種", align: "l",
    fmt: (c) => `<span class="muted tiny">${escapeHtml(c.sector33 || "—")}</span>` },
  { key: "score", label: "スコア", fmt: (c) => signed(c.score, 2) },
  { key: "ev_r", label: "期待値(R)", fmt: (c) => `<span class="${cls(c.ev_r)}">${signed(c.ev_r)}</span>` },
  { key: "est_win_rate", label: "想定勝率", fmt: (c) => pct(c.est_win_rate, 0) },
  { key: "close", label: "終値", fmt: (c) => num(c.close, 1) },
  { key: "stop", label: "損切り", fmt: (c) => `${num(c.stop, 1)}<span class="muted tiny"> −${pct(c.stop_pct, 1)}</span>` },
  { key: "target", label: "目標", fmt: (c) => `${num(c.target, 1)}<span class="muted tiny"> +${pct(c.target_pct, 1)}</span>` },
  { key: "atr_pct", label: "ATR%", fmt: (c) => pct(c.atr_pct, 1) },
  { key: "turnover_ma25", label: "売買代金", fmt: (c) => oku(c.turnover_ma25) },
  { key: "_shares", label: "株数", fmt: (c) => sizedCell(c) },
  { key: "_cost", label: "必要資金", fmt: (c) => yen(c._sz.cost) },
];

function sizedCell(c) {
  if (c._sz.shares > 0) {
    const cap = c._sz.note === "position_cap" ? '<span class="flag">上限</span>' : "";
    return c._sz.shares.toLocaleString("ja-JP") + cap;
  }
  return '<span class="muted">0</span><span class="flag">1単元が予算超過</span>';
}

function sortValue(c, key) {
  if (key === "_shares") return c._sz.shares;
  if (key === "_cost") return c._sz.cost;
  return c[key];
}

function renderTable() {
  const a = account();
  const rows = state.screen.candidates.map((c) => ({
    ...c,
    _sz: positionSize(c.entry_ref, c.entry_ref - c.stop, a),
  }));

  const { key, dir } = state.sort;
  rows.sort((x, y) => {
    const vx = sortValue(x, key), vy = sortValue(y, key);
    if (vx == null) return 1;
    if (vy == null) return -1;
    if (typeof vx === "string") return dir * vx.localeCompare(vy, "ja");
    return dir * (vx - vy);
  });

  document.getElementById("screen-head").innerHTML = COLUMNS.map((col) => {
    const active = state.sort.key === col.key;
    const arrow = active ? `<span class="arrow">${state.sort.dir > 0 ? "▲" : "▼"}</span>` : "";
    return `<th class="${col.align === "l" ? "l" : ""}" data-key="${col.key}">${col.label}${arrow}</th>`;
  }).join("");

  document.getElementById("screen-body").innerHTML = rows
    .map((c) => {
      const cells = COLUMNS.map(
        (col) => `<td class="${col.align === "l" ? "l" : ""}">${col.fmt(c)}</td>`
      ).join("");
      const open = state.openCode === c.code;
      return (
        `<tr data-code="${c.code}" class="${open ? "open" : ""}">${cells}</tr>` +
        `<tr class="detail-row ${open ? "" : "hidden"}" data-detail="${c.code}">` +
        `<td colspan="${COLUMNS.length}">${open ? detailHTML(c, a) : ""}</td></tr>`
      );
    })
    .join("");

  document.querySelectorAll("#screen-head th").forEach((th) =>
    th.addEventListener("click", () => {
      const k = th.dataset.key;
      state.sort = state.sort.key === k
        ? { key: k, dir: -state.sort.dir }
        : { key: k, dir: k === "rank" || k === "code" || k === "name" ? 1 : -1 };
      renderTable();
    })
  );
  document.querySelectorAll("#screen-body tr[data-code]").forEach((tr) =>
    tr.addEventListener("click", () => {
      state.openCode = state.openCode === tr.dataset.code ? null : tr.dataset.code;
      renderTable();
    })
  );
}

// ---------------------------------------------------------------- detail

function detailHTML(c, a) {
  const series = state.charts?.series?.[c.code];
  const chart = series ? candleChart(series, c) : '<div class="muted">チャートデータがありません</div>';

  const maxAbs = Math.max(0.001, ...(c.contributions || []).map((x) => Math.abs(x.contribution)));
  const contrib = (c.contributions || [])
    .map((x) => {
      const w = (Math.abs(x.contribution) / maxAbs) * 50;
      const positive = x.contribution >= 0;
      const style = positive ? `left:50%;width:${w}%` : `right:50%;width:${w}%`;
      return (
        `<div class="contrib-row" title="${escapeHtml(x.factor)} raw=${x.raw ?? "—"} z=${x.z}">` +
        `<div class="lbl">${escapeHtml(x.label)}</div>` +
        `<div class="contrib-bar"><span class="mid" style="left:50%"></span>` +
        `<i class="${positive ? "pos" : "neg"}" style="${style}"></i></div>` +
        `<div class="val">${signed(x.contribution, 3)}</div></div>`
      );
    })
    .join("");

  const rr = c.reward_risk;
  const plan = [
    ["想定エントリー", `${num(c.entry_ref, 1)} 円`],
    ["（参考価格）", "翌営業日の寄付が実際の建値"],
    ["損切り", `${num(c.stop, 1)} 円（−${pct(c.stop_pct, 1)}）`],
    ["目標", `${num(c.target, 1)} 円（+${pct(c.target_pct, 1)}）`],
    ["リスクリワード", `1 : ${num(rr, 1)}`],
    ["最大保有期間", `${state.screen.trade_plan.max_hold_bars} 営業日`],
    ["ATR(14)", `${num(c.atr14, 1)} 円（${pct(c.atr_pct, 1)}）`],
    ["25日平均売買代金", `${oku(c.turnover_ma25)}円`],
    ["株数", c._sz.shares ? `${c._sz.shares.toLocaleString("ja-JP")} 株` : "建てられません"],
    ["必要資金", yen(c._sz.cost)],
    ["実際のリスク額", yen(c._sz.risk)],
    ["期待値", `${signed(c.ev_r)}R（${yen(c.ev_r * c._sz.risk)}）`],
  ];

  return (
    '<div class="detail"><div>' +
    "<h3>日足チャート（直近140本）</h3>" + chart +
    '<div class="legend">' +
    '<span><i style="background:var(--accent)"></i>25日線</span>' +
    '<span><i style="background:var(--warn)"></i>75日線</span>' +
    '<span><i style="background:var(--down)"></i>損切り</span>' +
    '<span><i style="background:var(--up)"></i>目標</span>' +
    "</div></div><div>" +
    "<h3>スコア内訳（重み × 偏差）</h3>" +
    `<div class="contrib">${contrib || '<div class="muted">内訳なし</div>'}</div>` +
    "<h3 style='margin-top:18px'>トレードプラン</h3>" +
    `<div class="plan-grid">${plan
      .map(([k, v]) => `<div><span>${escapeHtml(k)}</span><span>${v}</span></div>`)
      .join("")}</div>` +
    "</div></div>"
  );
}

function candleChart(s, c) {
  const W = 640, H = 300, PL = 46, PR = 8, PT = 10, PB = 18;
  const n = s.close.length;
  if (!n) return "";

  const all = [];
  for (let i = 0; i < n; i++) {
    if (isFinite(s.high[i])) all.push(s.high[i]);
    if (isFinite(s.low[i])) all.push(s.low[i]);
  }
  [c.stop, c.target].forEach((v) => { if (isFinite(v)) all.push(v); });
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.05 || 1;
  lo -= pad; hi += pad;

  const x = (i) => PL + ((W - PL - PR) * i) / Math.max(1, n - 1);
  const y = (v) => PT + (H - PT - PB) * (1 - (v - lo) / (hi - lo));
  const bw = Math.max(1.4, ((W - PL - PR) / n) * 0.62);

  let out = `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">`;

  for (let g = 0; g <= 4; g++) {
    const v = lo + ((hi - lo) * g) / 4;
    const yy = y(v);
    out += `<line class="gridline" x1="${PL}" y1="${yy.toFixed(1)}" x2="${W - PR}" y2="${yy.toFixed(1)}"/>`;
    out += `<text x="${PL - 5}" y="${(yy + 3).toFixed(1)}" text-anchor="end">${Math.round(v).toLocaleString("ja-JP")}</text>`;
  }

  for (let i = 0; i < n; i++) {
    const o = s.open[i], h = s.high[i], l = s.low[i], cl = s.close[i];
    if (![o, h, l, cl].every(isFinite)) continue;
    const up = cl >= o;
    const col = up ? "var(--up)" : "var(--down)";
    const xi = x(i);
    const yo = y(o), yc = y(cl);
    out += `<line x1="${xi.toFixed(1)}" y1="${y(h).toFixed(1)}" x2="${xi.toFixed(1)}" y2="${y(l).toFixed(1)}" stroke="${col}" stroke-width="1"/>`;
    out += `<rect x="${(xi - bw / 2).toFixed(1)}" y="${Math.min(yo, yc).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(1, Math.abs(yc - yo)).toFixed(1)}" fill="${col}"/>`;
  }

  for (const [key, klass] of [["sma25", "sma25"], ["sma75", "sma75"]]) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const v = s[key]?.[i];
      if (v != null && isFinite(v)) pts.push(`${x(i).toFixed(1)},${y(v).toFixed(1)}`);
    }
    if (pts.length > 1) out += `<polyline class="${klass}" points="${pts.join(" ")}"/>`;
  }

  for (const [v, klass, label] of [[c.stop, "stop", "損切り"], [c.target, "target", "目標"]]) {
    if (!isFinite(v) || v < lo || v > hi) continue;
    const yy = y(v);
    out += `<line class="lvl ${klass}" x1="${PL}" y1="${yy.toFixed(1)}" x2="${W - PR}" y2="${yy.toFixed(1)}"/>`;
    out += `<text x="${W - PR - 2}" y="${(yy - 3).toFixed(1)}" text-anchor="end">${label} ${Math.round(v).toLocaleString("ja-JP")}</text>`;
  }

  const first = s.date[0], last = s.date[n - 1];
  out += `<text x="${PL}" y="${H - 5}">${first}</text>`;
  out += `<text x="${W - PR}" y="${H - 5}" text-anchor="end">${last}</text>`;
  return out + "</svg>";
}

// ------------------------------------------------------------ validation

function renderValidation() {
  const b = state.backtest;
  if (!b || !b.deciles) return;

  const mono = b.monotonicity;
  const oos = b.oos_summary || {};
  const top = b.top_bucket || {};
  const base = b.baseline || {};

  document.getElementById("validation-stats").innerHTML = [
    {
      k: "単調性",
      v: signed(mono, 2),
      s: mono != null && mono >= 0.6 ? "スコアと成績が対応している" : "対応が確認できない",
      c: mono != null && mono >= 0.6 ? "pos" : "neg",
    },
    { k: "最上位グループの実現EV", v: signed(top.ev_real) + "R",
      s: `${(top.n ?? 0).toLocaleString("ja-JP")} トレード／勝率 ${pct(top.win_rate, 0)}`, c: cls(top.ev_real) },
    { k: "全シグナル平均EV", v: signed(base.expectancy_r) + "R",
      s: "無選別に売買した場合の基準線", c: cls(base.expectancy_r) },
    { k: "検証期間トータル", v: pct(oos.total_return, 1), s: `最大DD ${pct(oos.max_drawdown, 1)}`, c: cls(oos.total_return) },
    { k: "シャープレシオ", v: num(oos.sharpe, 2), s: `${(oos.trades ?? 0).toLocaleString("ja-JP")} トレード`, c: cls(oos.sharpe) },
    { k: "勝率 / ペイオフ", v: pct(oos.win_rate, 0), s: `平均利益÷平均損失 ${num(oos.payoff, 2)}` },
  ].map(statHTML).join("");

  renderDeciles(b.deciles);

  document.getElementById("decile-table").innerHTML = tableHTML(
    ["グループ", "件数", "平均スコア", "予測EV", "実現EV", "勝率"],
    b.deciles.map((d) => [
      d.bucket, (d.n ?? 0).toLocaleString("ja-JP"), signed(d.score_mean, 2),
      signed(d.ev_pred), `<span class="${cls(d.ev_real)}">${signed(d.ev_real)}</span>`, pct(d.win_rate, 1),
    ]),
    [true, false, false, false, false, false]
  );

  const curve = b.equity_curve || [];
  document.getElementById("equity-chart").innerHTML = lineChart(curve);
  // The curve only covers dates on which the model actually wanted to trade,
  // so it can stop short of the last fold — worth saying rather than leaving
  // the reader to misread the axis.
  document.getElementById("equity-note").textContent = curve.length
    ? `期待値がプラスと判定された日のみ売買。${curve[0].date} 〜 ${curve[curve.length - 1].date}、` +
      `計 ${(oos.trades ?? 0).toLocaleString("ja-JP")} トレード（1トレード ${pct(oos.risk_per_trade, 1)} リスク、` +
      `最大 ${oos.max_open_positions ?? "—"} 銘柄同時保有）。`
    : "売買条件を満たすシグナルがありませんでした。";

  document.getElementById("ic-table").innerHTML = tableHTML(
    ["ファクター", "分類", "IC", "t値", "採用重み"],
    (b.ic_table || [])
      .slice()
      .sort((a, z) => (z.weight ?? 0) - (a.weight ?? 0) || (z.ic_mean ?? 0) - (a.ic_mean ?? 0))
      .map((r) => [
        escapeHtml(r.label || r.factor), `<span class="cat">${escapeHtml(r.category || "")}</span>`,
        `<span class="${cls(r.ic_mean)}">${signed(r.ic_mean, 4)}</span>`,
        signed(r.ic_t, 1),
        r.weight > 0 ? `<strong>${num(r.weight, 3)}</strong>` : '<span class="muted">不採用</span>',
      ]),
    [true, true, false, false, false]
  );

  document.getElementById("fold-table").innerHTML = tableHTML(
    ["#", "学習期間", "学習トレード数", "検証期間", "検証シグナル数"],
    (b.folds || []).map((f) => [
      f.fold, `${f.train_start} 〜 ${f.train_end}`, (f.train_trades ?? 0).toLocaleString("ja-JP"),
      `${f.test_start} 〜 ${f.test_end}`, (f.test_signals ?? 0).toLocaleString("ja-JP"),
    ]),
    [true, true, false, true, false]
  );
}

function renderDeciles(deciles) {
  const vals = deciles.map((d) => d.ev_real ?? 0);
  const maxAbs = Math.max(0.001, ...vals.map(Math.abs));
  document.getElementById("decile-chart").innerHTML = deciles
    .map((d) => {
      const v = d.ev_real ?? 0;
      const h = (Math.abs(v) / maxAbs) * 46;
      const neg = v < 0;
      // Positive bars grow up from the midline, negative ones hang below it.
      return (
        `<div class="decile-col" title="グループ${d.bucket}: 実現EV ${signed(v)}R / ${d.n} 件">` +
        `<div style="height:50%;display:flex;flex-direction:column;justify-content:flex-end;width:100%;align-items:center">` +
        (neg ? "" : `<div class="bar" style="height:${h}%"></div>`) + "</div>" +
        `<div style="height:50%;display:flex;flex-direction:column;justify-content:flex-start;width:100%;align-items:center">` +
        (neg ? `<div class="bar neg" style="height:${h}%"></div>` : "") + "</div>" +
        "</div>"
      );
    })
    .join("");
  document.getElementById("decile-labels").innerHTML = deciles
    .map((d) => `<div>${d.bucket}</div>`)
    .join("");
}

function lineChart(points) {
  if (!points.length) return '<div class="muted">データがありません</div>';
  const W = 900, H = 240, PL = 52, PR = 10, PT = 10, PB = 22;
  const vals = points.map((p) => p.equity);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const span = hi - lo || 1;
  const x = (i) => PL + ((W - PL - PR) * i) / Math.max(1, points.length - 1);
  const y = (v) => PT + (H - PT - PB) * (1 - (v - lo) / span);

  let out = `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">`;
  for (let g = 0; g <= 4; g++) {
    const v = lo + (span * g) / 4;
    const yy = y(v);
    out += `<line class="gridline" x1="${PL}" y1="${yy.toFixed(1)}" x2="${W - PR}" y2="${yy.toFixed(1)}"/>`;
    out += `<text x="${PL - 5}" y="${(yy + 3).toFixed(1)}" text-anchor="end">${(v * 100).toFixed(0)}</text>`;
  }
  if (lo < 1 && hi > 1) {
    const y1 = y(1);
    out += `<line x1="${PL}" y1="${y1.toFixed(1)}" x2="${W - PR}" y2="${y1.toFixed(1)}" stroke="var(--faint)" stroke-dasharray="3 3"/>`;
  }
  const pts = points.map((p, i) => `${x(i).toFixed(1)},${y(p.equity).toFixed(1)}`).join(" ");
  const end = points[points.length - 1].equity;
  const color = end >= 1 ? "var(--up)" : "var(--down)";
  out += `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.6"/>`;
  out += `<text x="${PL}" y="${H - 6}">${points[0].date}</text>`;
  out += `<text x="${W - PR}" y="${H - 6}" text-anchor="end">${points[points.length - 1].date}</text>`;
  out += `<text x="${PL - 5}" y="${PT + 8}" text-anchor="end"></text>`;
  return out + "</svg>";
}

function tableHTML(headers, rows, leftAlign) {
  const th = headers
    .map((h, i) => `<th class="plain ${leftAlign?.[i] ? "l" : ""}">${h}</th>`)
    .join("");
  const tb = rows
    .map(
      (r) =>
        "<tr>" +
        r.map((c, i) => `<td class="${leftAlign?.[i] ? "l" : ""}">${c}</td>`).join("") +
        "</tr>"
    )
    .join("");
  return `<thead><tr>${th}</tr></thead><tbody>${tb}</tbody>`;
}

// --------------------------------------------------------------- factors

function renderFactors() {
  const s = state.screen;
  const p = s.trade_plan, f = s.filters;

  document.getElementById("plan-stats").innerHTML = [
    { k: "エントリー", v: "翌日寄付", s: "シグナル発生日の終値では約定できないため" },
    { k: "損切り", v: `ATR × ${p.stop_atr_mult}`, s: "この幅が 1R（リスク単位）" },
    { k: "利益確定", v: `ATR × ${p.target_atr_mult}`, s: `リスクリワード 1 : ${num(p.reward_risk, 1)}` },
    { k: "最大保有", v: `${p.max_hold_bars} 営業日`, s: "未達なら終値で手仕舞い" },
    { k: "往復コスト", v: pct(p.cost_pct, 2), s: "手数料・スプレッド・スリッページ" },
  ].map(statHTML).join("");

  document.getElementById("filter-stats").innerHTML = [
    { k: "最低売買代金", v: `${oku(f.min_turnover_yen)}円`, s: "25日平均。流動性不足を除外" },
    { k: "株価レンジ", v: `${num(f.min_price, 0)}〜${num(f.max_price, 0)}円`, s: "1単元が現実的な価格帯" },
    { k: "ATR%レンジ", v: `${pct(f.min_atr_pct, 1)}〜${pct(f.max_atr_pct, 1)}`, s: "値動きが小さすぎ／荒すぎる銘柄を除外" },
  ].map(statHTML).join("");

  const maxW = Math.max(0.0001, ...s.factors.map((x) => x.weight));
  document.getElementById("factor-grid").innerHTML = s.factors
    .slice()
    .sort((a, z) => z.weight - a.weight)
    .map((x) => {
      const dir = { higher: "高いほど良い", lower: "低いほど良い", band: "適正レンジ内が良い" }[x.direction];
      const band = x.band ? `　レンジ ${x.band[0]}〜${x.band[1]}` : "";
      return (
        `<div class="factor ${x.weight > 0 ? "" : "inactive"}">` +
        `<div class="fh"><span class="fn">${escapeHtml(x.label)}</span>` +
        `<span class="fw">${x.weight > 0 ? num(x.weight, 3) : "不採用"}</span></div>` +
        `<div class="cat">${escapeHtml(x.category)}</div>` +
        `<div class="wbar"><i style="width:${(x.weight / maxW) * 100}%"></i></div>` +
        `<div class="fr">${escapeHtml(x.rationale)}</div>` +
        `<div class="fm">${escapeHtml(x.name)}　${dir}${escapeHtml(band)}</div>` +
        "</div>"
      );
    })
    .join("");
}

// ------------------------------------------------------------------ tabs

function wireTabs() {
  const buttons = document.querySelectorAll("nav.tabs button");
  buttons.forEach((btn) =>
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
      document.querySelectorAll(".tab-panel").forEach((p) => {
        p.hidden = p.id !== "tab-" + btn.dataset.tab;
      });
    })
  );
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (m) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[m]
  );
}

boot();
