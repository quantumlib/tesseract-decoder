/* ==========================================================================
   Animation Module: qec_primitives
   Interactive step-through QEC / BP primitives shared by the slide deck and
   the standalone explorer. Every message value is computed live by a tiny
   min-sum engine (no hand-typed message numbers).
     - "bp_minsum_stepper"     : Min-Sum BP on a tree-like Tanner graph
                                 (Step 0..4: priors -> init V->C -> C->V "Min"
                                 -> V->C "Sum" -> posterior & hard decision)
     - "bp_4cycle_oscillation" : Symmetric 4-cycle oscillation stepper with a
                                 live L_v^(t) trajectory plot (alias "bp_4cycle")
     - "gari_transform"        : 2 tabs (Row-XOR elimination / Biclique factor
                                 synthesis) x Before / Step / After, showing the
                                 Tanner graph AND the parity-check matrix H
     - "multipass_funnel"      : 4-stage Tanner graph + H transformation viewer
                                 (raw DEM -> duplicate merge -> X/Z shattering ->
                                 soft 75%/25% cross-pass prior injection)
   Registered IDs: "bp_minsum_stepper", "bp_4cycle_oscillation", "bp_4cycle",
                   "gari_transform", "multipass_funnel", "qec_primitives"
   ========================================================================== */

(function () {
  "use strict";

  const C = {
    ink: "#1E2022",
    muted: "#5F6368",
    hair: "#DFDCD4",
    grey: "#94A3B8",
    blue: "#1967D2",
    red: "#D93025",
    green: "#1E8E3E",
    orange: "#D97706",
    gold: "#F9AB00",
    teal: "#00796B",
    pink: "#DB2777"
  };

  /* ------------------------------------------------------------------------
     Styles
     ------------------------------------------------------------------------ */
  function ensureStyles() {
    if (document.getElementById("tesseract-qec-primitives-styles")) return;
    const style = document.createElement("style");
    style.id = "tesseract-qec-primitives-styles";
    style.textContent = `
      .qp-root { width:100%; height:100%; min-height:0; display:flex; flex-direction:column; gap:7px;
        font-family:"DM Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; color:#1E2022; box-sizing:border-box; }
      .qp-root * { box-sizing:border-box; }
      .qp-bar { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:6px;
        padding:6px 10px; background:#F4F1EA; border:1px solid #DFDCD4; border-radius:8px; }
      .qp-bar-group { display:flex; flex-wrap:wrap; align-items:center; gap:4px; }
      .qp-title { font-family:"Space Mono",monospace; font-size:0.69rem; font-weight:700; text-transform:uppercase;
        letter-spacing:0.03em; color:#1967D2; }
      .qp-btn { font-family:"Space Mono",monospace; font-size:0.66rem; font-weight:700; padding:4px 9px; border-radius:6px;
        border:1px solid #CBD5E1; background:#FFFFFF; color:#1E293B; cursor:pointer; line-height:1.15; white-space:nowrap;
        transition:background 0.12s ease, color 0.12s ease, border-color 0.12s ease; }
      .qp-btn:hover:not(:disabled) { border-color:#1967D2; color:#1967D2; }
      .qp-btn.active { background:#1967D2; border-color:#1967D2; color:#FFFFFF; }
      .qp-btn.qp-play.active { background:#1E8E3E; border-color:#1E8E3E; color:#FFFFFF; }
      .qp-btn:disabled { opacity:0.38; cursor:default; }
      .qp-pill { border-radius:999px; font-size:0.63rem; padding:3px 9px; }
      .qp-tabs { display:flex; flex-wrap:wrap; gap:4px; }
      .qp-tab { font-family:"Space Mono",monospace; font-size:0.7rem; font-weight:700; padding:6px 12px; border-radius:7px;
        border:1px solid #CBD5E1; background:#FFFFFF; color:#334155; cursor:pointer; }
      .qp-tab.active { background:#1E2022; color:#FFFFFF; border-color:#1E2022; }
      .qp-body { flex:1; min-height:0; display:flex; flex-wrap:wrap; gap:7px; align-items:stretch; }
      .qp-card { background:#FFFFFF; border:1px solid #DFDCD4; border-radius:10px; padding:8px 10px; display:flex;
        flex-direction:column; gap:6px; min-width:0; box-shadow:0 1px 3px rgba(0,0,0,0.03); }
      .qp-card-graph { flex:1.35 1 330px; }
      .qp-card-side { flex:1 1 270px; }
      .qp-card-half { flex:1 1 300px; }
      .qp-card-h { font-family:"Space Mono",monospace; font-size:0.64rem; font-weight:700; text-transform:uppercase;
        letter-spacing:0.04em; color:#5F6368; display:flex; justify-content:space-between; align-items:center; gap:6px; flex-wrap:wrap; }
      .qp-svg { width:100%; height:auto; display:block; overflow:visible; }
      .qp-svg-wrap { flex:1; display:flex; align-items:center; justify-content:center; min-height:0; }
      .qp-calc { font-size:0.75rem; line-height:1.4; display:flex; flex-direction:column; gap:5px; }
      .qp-step-h { font-weight:700; font-size:0.8rem; color:#1E2022; }
      .qp-formula { font-family:Georgia,"Times New Roman",serif; font-style:italic; font-size:0.86rem; background:#F8FAFC;
        border:1px solid #E2E8F0; border-radius:6px; padding:6px 8px; text-align:center; line-height:1.5; }
      .qp-formula .qp-up { font-style:normal; }
      .qp-line { font-family:"Space Mono",monospace; font-size:0.66rem; padding:4px 7px; border-radius:5px;
        border:1px solid transparent; background:#FAFAF7; line-height:1.45; }
      .qp-line.qp-click { cursor:pointer; }
      .qp-line.qp-click:hover { border-color:#CBD5E1; }
      .qp-line.focus { border-color:#1967D2; background:#EFF6FF; }
      .qp-line.ok { border-color:#86EFAC; background:#F0FDF4; }
      .qp-line.bad { border-color:#FCA5A5; background:#FEF2F2; }
      .qp-neg { color:#D93025; font-weight:700; }
      .qp-pos { color:#1E8E3E; font-weight:700; }
      .qp-note { font-size:0.68rem; color:#5F6368; line-height:1.38; }
      .qp-caption { font-size:0.73rem; line-height:1.4; color:#2D3134; background:#FBF9F5; border:1px solid #DFDCD4;
        border-radius:7px; padding:6px 10px; }
      .qp-metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(96px,1fr)); gap:5px; }
      .qp-metric { background:#F4F1EA; border:1px solid #DFDCD4; border-radius:7px; padding:4px 8px; }
      .qp-metric-k { font-family:"Space Mono",monospace; font-size:0.57rem; font-weight:700; text-transform:uppercase; color:#5F6368; }
      .qp-metric-v { font-size:0.86rem; font-weight:700; }
      .qp-mat-wrap { display:flex; flex-direction:column; align-items:center; gap:2px; }
      .qp-mat-title { font-family:"Space Mono",monospace; font-size:0.63rem; font-weight:700; text-align:center; }
      table.qp-mat { border-collapse:separate; border-spacing:3px; font-family:"Space Mono",monospace; }
      .qp-mat th { font-size:0.6rem; color:#5F6368; font-weight:700; padding:0 3px; white-space:nowrap; }
      .qp-mat td { min-width:25px; height:23px; text-align:center; font-size:0.78rem; font-weight:700; border-radius:4px;
        border:1px solid #E2E8F0; background:#FFFFFF; color:#B6BEC9; transition:background 0.35s, color 0.35s, border-color 0.35s; }
      .qp-mat td.one { color:#1E2022; background:#F1F5F9; border-color:#94A3B8; }
      .qp-mat td.red { background:#FEE2E2; border-color:#D93025; color:#991B1B; }
      .qp-mat td.green { background:#DCFCE7; border-color:#1E8E3E; color:#14532D; }
      .qp-mat td.blue { background:#DBEAFE; border-color:#1967D2; color:#1E3A8A; }
      .qp-mat td.orange { background:#FEF3C7; border-color:#D97706; color:#92400E; }
      .qp-mat td.gold { background:#FEF7E0; border-color:#F9AB00; color:#8D5B00; }
      .qp-mat td.pink { background:#FCE7F3; border-color:#DB2777; color:#9D174D; }
      .qp-mat td.strike { text-decoration:line-through; opacity:0.6; }
      .qp-mat td.op { border:none; background:transparent; color:#1967D2; font-size:0.9rem; min-width:16px; }
      .qp-mat td.lbl { border:none; background:transparent; color:#5F6368; font-size:0.6rem; min-width:0; padding-right:4px; text-align:right; white-space:nowrap; }
      .qp-mat tr.sep td { border-top:2px solid #1967D2; }
      .qp-pulse { animation:qpPulse 1.1s ease-in-out infinite; }
      @keyframes qpPulse { 0%,100% { opacity:1; } 50% { opacity:0.35; } }
    `;
    document.head.appendChild(style);
  }

  /* ------------------------------------------------------------------------
     Small utilities
     ------------------------------------------------------------------------ */
  const SUB = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉" };
  function sub(n) {
    return String(n).replace(/[0-9]/g, (d) => SUB[d]);
  }

  function fmt(x, d) {
    const digits = d === undefined ? 2 : d;
    if (Math.abs(x) < 5e-9) return (0).toFixed(digits);
    return (x > 0 ? "+" : "−") + Math.abs(x).toFixed(digits);
  }

  function signCls(x) {
    if (Math.abs(x) < 5e-9) return "";
    return x < 0 ? "qp-neg" : "qp-pos";
  }

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  }

  function btn(label, cls) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "qp-btn" + (cls ? " " + cls : "");
    b.innerHTML = label;
    return b;
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function easeInOut(t) {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }

  function along(x1, y1, x2, y2, f) {
    return [x1 + (x2 - x1) * f, y1 + (y2 - y1) * f];
  }

  /** rAF tween with a timeout fallback so the final frame is always drawn. */
  function tween(ms, onFrame, onDone) {
    const t0 = performance.now();
    let rafId = 0;
    let finished = false;
    let dead = false;
    function finish() {
      if (finished || dead) return;
      finished = true;
      onFrame(1);
      if (onDone) onDone();
    }
    function frame(now) {
      if (dead || finished) return;
      const t = Math.min(1, (now - t0) / ms);
      if (t >= 1) {
        finish();
        return;
      }
      onFrame(easeInOut(t));
      rafId = requestAnimationFrame(frame);
    }
    onFrame(0);
    rafId = requestAnimationFrame(frame);
    const fallback = setTimeout(finish, ms + 160);
    return function cancel() {
      dead = true;
      cancelAnimationFrame(rafId);
      clearTimeout(fallback);
    };
  }

  /* ------------------------------------------------------------------------
     SVG primitives (return markup strings)
     ------------------------------------------------------------------------ */
  function svgCheck(x, y, label, o) {
    const opt = o || {};
    const s = opt.size || 34;
    const op = opt.opacity === undefined ? 1 : opt.opacity;
    let out = `<g opacity="${op}">`;
    if (opt.glow) {
      out += `<rect class="qp-pulse" x="${x - s / 2 - 6}" y="${y - s / 2 - 6}" width="${s + 12}" height="${s + 12}" rx="7" fill="none" stroke="${opt.glow}" stroke-width="3"/>`;
    }
    out += `<rect x="${x - s / 2}" y="${y - s / 2}" width="${s}" height="${s}" rx="4" fill="${opt.fill || "#FFFFFF"}" stroke="${opt.stroke || C.ink}" stroke-width="${opt.sw || 2}"/>`;
    out += `<text x="${x}" y="${y + 5}" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="${opt.fs || 15}" fill="${opt.text || C.ink}">${label}</text>`;
    return out + `</g>`;
  }

  function svgVar(x, y, label, o) {
    const opt = o || {};
    const r = opt.r || 18;
    const op = opt.opacity === undefined ? 1 : opt.opacity;
    let out = `<g opacity="${op}">`;
    if (opt.glow) {
      out += `<circle class="qp-pulse" cx="${x}" cy="${y}" r="${r + 6}" fill="none" stroke="${opt.glow}" stroke-width="3"/>`;
    }
    if (opt.rx) {
      out += `<ellipse cx="${x}" cy="${y}" rx="${opt.rx}" ry="${r}" fill="${opt.fill || "#FFFFFF"}" stroke="${opt.stroke || C.ink}" stroke-width="${opt.sw || 2}"/>`;
    } else {
      out += `<circle cx="${x}" cy="${y}" r="${r}" fill="${opt.fill || "#FFFFFF"}" stroke="${opt.stroke || C.ink}" stroke-width="${opt.sw || 2}"/>`;
    }
    out += `<text x="${x}" y="${y + 5}" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="${opt.fs || 15}" fill="${opt.text || C.ink}">${label}</text>`;
    return out + `</g>`;
  }

  function svgLine(x1, y1, x2, y2, o) {
    const opt = o || {};
    return `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${opt.color || C.grey}" stroke-width="${opt.w || 2}" stroke-linecap="round"${opt.dash ? ` stroke-dasharray="${opt.dash}"` : ""} opacity="${opt.opacity === undefined ? 1 : opt.opacity}"/>`;
  }

  /** Text label with a rounded background box. `lines` may be a string or array. */
  function svgTag(x, y, lines, o) {
    const opt = o || {};
    const arr = Array.isArray(lines) ? lines : [lines];
    const fs = opt.fs || 10.5;
    const lh = fs * 1.25;
    const maxLen = Math.max(...arr.map((s) => String(s).replace(/<[^>]+>/g, "").length));
    const w = maxLen * fs * 0.6 + 10;
    const h = arr.length * lh + 6;
    const op = opt.opacity === undefined ? 1 : opt.opacity;
    let out = `<g opacity="${op}">`;
    out += `<rect x="${(x - w / 2).toFixed(1)}" y="${(y - h / 2).toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="4" fill="${opt.bg || "#FFFFFF"}" stroke="${opt.color || C.ink}" stroke-width="${opt.sw || 1.1}"/>`;
    arr.forEach((s, i) => {
      const ty = y - h / 2 + 3 + lh * (i + 0.78);
      out += `<text x="${x.toFixed(1)}" y="${ty.toFixed(1)}" text-anchor="middle" font-family="'Space Mono',monospace" font-size="${fs}" font-weight="${opt.weight || 700}" fill="${(opt.colors && opt.colors[i]) || opt.color || C.ink}">${s}</text>`;
    });
    return out + `</g>`;
  }

  function svgText(x, y, text, o) {
    const opt = o || {};
    return `<text x="${x}" y="${y}" text-anchor="${opt.anchor || "middle"}" font-family="${opt.font || "'DM Sans',sans-serif"}" font-size="${opt.fs || 11}" font-weight="${opt.weight || 600}" fill="${opt.color || C.muted}"${opt.italic ? ` font-style="italic"` : ""} opacity="${opt.opacity === undefined ? 1 : opt.opacity}">${text}</text>`;
  }

  /** Message packet (moving dot + value) for animations. */
  function svgPacket(x, y, text, color) {
    const w = Math.max(30, String(text).length * 6.4 + 10);
    return `<g><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill="${color}" stroke="#FFFFFF" stroke-width="1.5"/>` +
      `<rect x="${(x + 8).toFixed(1)}" y="${(y - 9).toFixed(1)}" width="${w.toFixed(1)}" height="16" rx="8" fill="${color}" opacity="0.94"/>` +
      `<text x="${(x + 8 + w / 2).toFixed(1)}" y="${(y + 3).toFixed(1)}" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#FFFFFF">${text}</text></g>`;
  }

  /* ------------------------------------------------------------------------
     Parity-check matrix renderer (HTML table)
     cellCls(r, c, val) -> extra CSS class ("red", "green", "strike red", ...)
     ------------------------------------------------------------------------ */
  function matrixHtml(rows, rowLabels, colLabels, cellCls, title, titleColor) {
    let h = `<div class="qp-mat-wrap">`;
    if (title) h += `<div class="qp-mat-title" style="color:${titleColor || C.ink};">${title}</div>`;
    h += `<table class="qp-mat"><tr><th></th>`;
    colLabels.forEach((cl) => {
      h += `<th>${cl}</th>`;
    });
    h += `</tr>`;
    rows.forEach((row, r) => {
      h += `<tr><th>${rowLabels[r]}</th>`;
      row.forEach((v, c) => {
        const extra = cellCls ? cellCls(r, c, v) || "" : "";
        const base = v ? "one" : "";
        h += `<td class="${base} ${extra}">${v}</td>`;
      });
      h += `</tr>`;
    });
    return h + `</table></div>`;
  }

  function countOnes(H) {
    return H.reduce((a, row) => a + row.reduce((b, v) => b + (v ? 1 : 0), 0), 0);
  }

  /** Number of 4-cycles = sum over check pairs of C(|shared vars|, 2). */
  function count4Cycles(H) {
    let n = 0;
    for (let a = 0; a < H.length; a++) {
      for (let b = a + 1; b < H.length; b++) {
        let k = 0;
        for (let v = 0; v < H[a].length; v++) if (H[a][v] && H[b][v]) k++;
        n += (k * (k - 1)) / 2;
      }
    }
    return n;
  }

  /* ------------------------------------------------------------------------
     Tiny flooding min-sum engine (normalized min-sum with factor alpha)
       m^(0)_{v->c} = L0_v
       mu^(t)_{c->v} = alpha (-1)^{s_c} prod_{v' != v} sgn(m^(t-1)_{v'->c}) min_{v' != v} |m^(t-1)_{v'->c}|
       m^(t)_{v->c}  = L0_v + sum_{c' != c} mu^(t)_{c'->v}
       L^(t)_v       = L0_v + sum_c mu^(t)_{c->v}
     ------------------------------------------------------------------------ */
  function minsumRun(H, L0, s, alpha, T) {
    const edges = [];
    H.forEach((row, c) =>
      row.forEach((v, j) => {
        if (v) edges.push({ c, v: j });
      })
    );
    const key = (c, v) => c + "," + v;
    const iters = [];
    let m = {};
    edges.forEach((e) => {
      m[key(e.c, e.v)] = L0[e.v];
    });
    iters.push({ t: 0, m: Object.assign({}, m), mu: null, L: L0.slice(), ehat: L0.map((x) => (x < 0 ? 1 : 0)) });
    for (let t = 1; t <= T; t++) {
      const mu = {};
      edges.forEach((e) => {
        const others = edges.filter((o) => o.c === e.c && o.v !== e.v);
        let sign = s[e.c] ? -1 : 1;
        let mn = Infinity;
        others.forEach((o) => {
          const x = m[key(e.c, o.v)];
          sign *= x > 0 ? 1 : x < 0 ? -1 : 0;
          mn = Math.min(mn, Math.abs(x));
        });
        if (!others.length) {
          sign = 0;
          mn = 0;
        }
        mu[key(e.c, e.v)] = alpha * sign * mn;
      });
      const L = L0.map((l0, v) => edges.filter((e) => e.v === v).reduce((a, e) => a + mu[key(e.c, e.v)], l0));
      const nm = {};
      edges.forEach((e) => {
        nm[key(e.c, e.v)] = edges
          .filter((o) => o.v === e.v && o.c !== e.c)
          .reduce((a, o) => a + mu[key(o.c, o.v)], L0[e.v]);
      });
      m = nm;
      iters.push({ t, m: Object.assign({}, nm), mu, L, ehat: L.map((x) => (x < 0 ? 1 : 0)) });
    }
    return { edges, key, iters };
  }

  function syndromeOf(H, e) {
    return H.map((row) => row.reduce((a, v, j) => a ^ (v & e[j]), 0));
  }

  /* ------------------------------------------------------------------------
     Step controller: [◀ Prev] [▶ Play/Pause] [Next ▶] + step pills
     ------------------------------------------------------------------------ */
  function createStepper(opts) {
    const n = opts.labels.length;
    const controls = el("div", "qp-bar-group");
    const prev = btn("◀ Prev");
    const play = btn("▶ Play", "qp-play");
    const next = btn("Next ▶");
    controls.appendChild(prev);
    controls.appendChild(play);
    controls.appendChild(next);
    const pillsWrap = el("div", "qp-bar-group");
    let idx = -1;
    let timer = null;
    const pills = opts.labels.map((lab, i) => {
      const b = btn(lab, "qp-pill");
      b.onclick = () => {
        stop();
        go(i);
      };
      pillsWrap.appendChild(b);
      return b;
    });
    function go(i) {
      const j = Math.max(0, Math.min(n - 1, i));
      const p = idx;
      idx = j;
      pills.forEach((b, k) => b.classList.toggle("active", k === j));
      prev.disabled = j === 0;
      next.disabled = j === n - 1;
      opts.onChange(j, p);
    }
    function tick() {
      if (!document.body.contains(opts.host)) {
        stop();
        return;
      }
      if (opts.host.offsetParent === null) return;
      if (idx >= n - 1) go(0);
      else go(idx + 1);
    }
    function start() {
      if (timer) return;
      if (idx >= n - 1) go(0);
      timer = setInterval(tick, opts.intervalMs || 2800);
      play.innerHTML = "❚❚ Pause";
      play.classList.add("active");
    }
    function stop() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
      play.innerHTML = "▶ Play";
      play.classList.remove("active");
    }
    prev.onclick = () => {
      stop();
      go(idx - 1);
    };
    next.onclick = () => {
      stop();
      go(idx + 1);
    };
    play.onclick = () => (timer ? stop() : start());
    return {
      controls,
      pills: pillsWrap,
      go,
      stop,
      start,
      index: () => idx
    };
  }

  /* ==========================================================================
     1. MIN-SUM BP STEPPER ("bp_minsum_stepper")
        Tree-like Tanner graph: c1 = {v1, v2}, c2 = {v2, v3};
        priors L0 = (+0.8, +2.5, +3.2); syndrome s = (1, 0); alpha = 0.8.
     ========================================================================== */
  function mountBpMinsumStepper(container, props) {
    ensureStyles();
    const p = props || {};
    const H = [
      [1, 1, 0],
      [0, 1, 1]
    ];
    const L0 = [0.8, 2.5, 3.2];
    const s = [1, 0];
    const alpha = typeof p.alpha === "number" ? p.alpha : 0.8;
    const run = minsumRun(H, L0, s, alpha, 1);
    const K = run.key;
    const E = run.edges;
    const it0 = run.iters[0];
    const it1 = run.iters[1];
    const ehat = it1.ehat;
    const synd = syndromeOf(H, ehat);

    const PC = [
      [170, 74],
      [350, 74]
    ];
    const PV = [
      [80, 254],
      [260, 254],
      [440, 254]
    ];
    const cName = (c) => `c${sub(c + 1)}`;
    const vName = (v) => `v${sub(v + 1)}`;
    const probOf = (L) => 1 / (1 + Math.exp(L));

    const STEP_LABELS = ["Step 0 · Priors", "Step 1 · Init V→C", "Step 2 · C→V “Min”", "Step 3 · V→C “Sum”", "Step 4 · Posterior ê"];
    const DEFAULT_FOCUS = [null, null, { dir: "cv", c: 0, v: 0 }, { dir: "vc", c: 0, v: 1 }, { dir: "post", v: 0 }];

    container.innerHTML = "";
    const root = el("div", "qp-root");
    const bar = el("div", "qp-bar");
    bar.appendChild(el("span", "qp-title", "Min-Sum BP · Step-Through on a Tanner Graph (α = " + alpha.toFixed(1) + ")"));
    const pillBar = el("div", "qp-bar");
    const body = el("div", "qp-body");
    const gCard = el("div", "qp-card qp-card-graph");
    const gHead = el("div", "qp-card-h");
    const svgWrap = el("div", "qp-svg-wrap");
    svgWrap.innerHTML = `<svg class="qp-svg" viewBox="0 0 520 330" style="max-height:340px;" xmlns="http://www.w3.org/2000/svg"><g class="qp-base"></g><g class="qp-overlay"></g></svg>`;
    gCard.appendChild(gHead);
    gCard.appendChild(svgWrap);
    const sCard = el("div", "qp-card qp-card-side");
    body.appendChild(gCard);
    body.appendChild(sCard);
    const caption = el("div", "qp-caption");
    root.appendChild(bar);
    root.appendChild(pillBar);
    root.appendChild(body);
    root.appendChild(caption);
    container.appendChild(root);

    const svg = svgWrap.querySelector("svg");
    const gBase = svg.querySelector(".qp-base");
    const gOver = svg.querySelector(".qp-overlay");

    let step = 0;
    let focus = null;
    let cancelAnim = null;

    function edgeGeom(e) {
      const [cx, cy] = PC[e.c];
      const [vx, vy] = PV[e.v];
      const dx = vx - cx;
      const dy = vy - cy;
      const len = Math.hypot(dx, dy);
      const ux = dx / len;
      const uy = dy / len;
      // start at check border, end at variable border
      return { x1: cx + ux * 20, y1: cy + uy * 20, x2: vx - ux * 20, y2: vy - uy * 20 };
    }

    /** Which edges are inputs / output for the current focus. */
    function focusRoles() {
      const roles = {};
      if (!focus) return roles;
      if (focus.dir === "cv") {
        E.forEach((e) => {
          if (e.c === focus.c && e.v !== focus.v) roles[K(e.c, e.v)] = "in";
          if (e.c === focus.c && e.v === focus.v) roles[K(e.c, e.v)] = "out";
        });
      } else if (focus.dir === "vc") {
        E.forEach((e) => {
          if (e.v === focus.v && e.c !== focus.c) roles[K(e.c, e.v)] = "in";
          if (e.v === focus.v && e.c === focus.c) roles[K(e.c, e.v)] = "out";
        });
      } else if (focus.dir === "post") {
        E.forEach((e) => {
          if (e.v === focus.v) roles[K(e.c, e.v)] = "in";
        });
      }
      return roles;
    }

    function edgeValue(e, st) {
      const k = K(e.c, e.v);
      if (st === 1) return { txt: `m=${fmt(it0.m[k])}`, val: it0.m[k], kind: "m" };
      if (st === 2 || st === 4) return { txt: `μ=${fmt(it1.mu[k])}`, val: it1.mu[k], kind: "mu" };
      if (st === 3) return { txt: `m=${fmt(it1.m[k])}`, val: it1.m[k], kind: "m" };
      return null;
    }

    function renderGraph(showLabels) {
      const roles = focusRoles();
      const hasFocus = Object.keys(roles).length > 0;
      let h = "";
      // Row labels
      h += svgText(12, 20, "CHECK NODES (squares)", { anchor: "start", fs: 9.5, font: "'Space Mono',monospace", weight: 700 });
      h += svgText(12, 318, "VARIABLE / FAULT NODES (circles)", { anchor: "start", fs: 9.5, font: "'Space Mono',monospace", weight: 700 });

      // Edges
      E.forEach((e) => {
        const g = edgeGeom(e);
        const role = roles[K(e.c, e.v)];
        let color = C.grey;
        let w = 2.4;
        let op = 1;
        if (step >= 1) {
          color = "#475569";
        }
        if (role === "in") {
          color = C.blue;
          w = 4;
        } else if (role === "out") {
          const val = edgeValue(e, step);
          color = val && val.val < 0 ? C.red : C.green;
          w = 4.4;
        } else if (hasFocus) {
          op = 0.35;
        }
        h += svgLine(g.x1, g.y1, g.x2, g.y2, { color, w, opacity: op });
      });

      // Edge labels
      if (showLabels && step >= 1) {
        E.forEach((e) => {
          const v = edgeValue(e, step);
          if (!v) return;
          const g = edgeGeom(e);
          const role = roles[K(e.c, e.v)];
          const [lx, ly] = along(g.x1, g.y1, g.x2, g.y2, 0.47);
          let color = v.kind === "m" ? C.blue : v.val < 0 ? C.red : C.green;
          let op = hasFocus && !role ? 0.4 : 1;
          if (step === 4) op = 0.75;
          h += svgTag(lx, ly, v.txt, {
            color,
            bg: role === "out" ? (v.val < 0 ? "#FEE2E2" : "#DCFCE7") : "#FFFFFF",
            sw: role ? 2 : 1.1,
            opacity: op,
            fs: 10.5
          });
        });
      }

      // Check nodes
      PC.forEach(([x, y], c) => {
        const unsat = s[c] === 1;
        const glow = step === 2 ? (focus && focus.dir === "cv" && focus.c === c ? C.blue : "#93C5FD") : null;
        h += svgCheck(x, y, cName(c), {
          fill: unsat ? "#FEE2E2" : "#DCFCE7",
          stroke: unsat ? C.red : C.green,
          text: unsat ? "#991B1B" : "#14532D",
          glow,
          size: 36
        });
        let tagTxt = unsat ? `s=1 ✗ unsatisfied` : `s=0 ✓ satisfied`;
        let tagCol = unsat ? C.red : C.green;
        if (step === 4) {
          tagTxt = `(Hê)=${synd[c]} = s ✓`;
          tagCol = synd[c] === s[c] ? C.green : C.red;
        }
        h += svgTag(x, y - 34, tagTxt, { color: tagCol, fs: 9.5, bg: "#FFFFFF" });
      });

      // Variable nodes
      PV.forEach(([x, y], v) => {
        const isFault = step === 4 && ehat[v] === 1;
        const glow = step === 3 ? (focus && focus.dir === "vc" && focus.v === v ? C.blue : "#93C5FD") : step === 4 && focus && focus.v === v ? C.blue : null;
        h += svgVar(x, y, vName(v), {
          fill: isFault ? "#FEE2E2" : step === 4 ? "#EFF6FF" : "#FFFFFF",
          stroke: isFault ? C.red : step === 4 ? C.blue : C.ink,
          text: isFault ? "#991B1B" : C.ink,
          glow,
          r: 19
        });
        if (step === 4) {
          const L = it1.L[v];
          h += svgTag(x, y + 38, [`L⁽¹⁾=${fmt(L)}`, `ê=${ehat[v]}`], {
            color: L < 0 ? C.red : C.blue,
            bg: L < 0 ? "#FEF2F2" : "#EFF6FF",
            fs: 9.5
          });
        } else {
          h += svgTag(x, y + 34, `L⁽⁰⁾=${fmt(L0[v])}`, { color: "#334155", fs: 9.5 });
        }
      });
      if (step === 0) {
        h += svgText(PV[0][0] + 22, PV[0][1] - 26, "← weakest prior: likely fault", { anchor: "start", fs: 9.5, color: C.red, weight: 700 });
      }
      gBase.innerHTML = h;
    }

    function packetsForStep(st) {
      const list = [];
      E.forEach((e) => {
        const g = edgeGeom(e);
        const v = edgeValue(e, st);
        if (!v) return;
        const fromVar = v.kind === "m";
        const color = v.kind === "m" ? C.blue : v.val < 0 ? C.red : C.green;
        list.push({
          x1: fromVar ? g.x2 : g.x1,
          y1: fromVar ? g.y2 : g.y1,
          x2: fromVar ? g.x1 : g.x2,
          y2: fromVar ? g.y1 : g.y2,
          txt: fmt(v.val),
          color
        });
      });
      return list;
    }

    function line(html, f, cls) {
      const d = f ? ` data-f='${JSON.stringify(f)}'` : "";
      return `<div class="qp-line${f ? " qp-click" : ""}${cls ? " " + cls : ""}"${d}>${html}</div>`;
    }

    function sameFocus(a, b) {
      return a && b && a.dir === b.dir && a.c === b.c && a.v === b.v;
    }

    function renderSide() {
      let h = "";
      if (step === 0) {
        h += `<div class="qp-step-h">Step 0 — Channel priors L⁽⁰⁾ and syndrome s</div>`;
        h += `<div class="qp-formula">L<sub>v</sub><sup>(0)</sup> = ln[(1 − p<sub>v</sub>)/p<sub>v</sub>], &nbsp; s = H e mod 2</div>`;
        h += matrixHtml(H, ["c₁", "c₂"], ["v₁", "v₂", "v₃"], null, "H (each 1 = one Tanner edge)", C.blue);
        L0.forEach((l, v) => {
          h += line(`${vName(v)}: L⁽⁰⁾ = ${fmt(l)} &nbsp;(p ≈ ${probOf(l).toFixed(2)})${v === 0 ? " ← least reliable" : ""}`);
        });
        h += line(`c₁: s = 1 → odd parity among {v₁, v₂} required`, null, "bad");
        h += line(`c₂: s = 0 → even parity among {v₂, v₃} required`, null, "ok");
      } else if (step === 1) {
        h += `<div class="qp-step-h">Step 1 — Initialise variable-to-check messages</div>`;
        h += `<div class="qp-formula">m<sub>v→c</sub><sup>(0)</sup> = L<sub>v</sub><sup>(0)</sup></div>`;
        E.forEach((e) => {
          h += line(`m ${vName(e.v)}→${cName(e.c)} = L${sub(e.v + 1)}⁽⁰⁾ = <span class="qp-pos">${fmt(it0.m[K(e.c, e.v)])}</span>`);
        });
        h += `<div class="qp-note">Each variable broadcasts its prior belief to every incident check (blue packets).</div>`;
      } else if (step === 2) {
        h += `<div class="qp-step-h">Step 2 — Check → Variable: the “Min” step</div>`;
        h += `<div class="qp-formula">μ<sub>c→v</sub> = α·(−1)<sup>s<sub>c</sub></sup>·∏<sub>v′≠v</sub> sgn(m<sub>v′→c</sub>)·min<sub>v′≠v</sub> |m<sub>v′→c</sub>|</div>`;
        E.forEach((e) => {
          const others = E.filter((o) => o.c === e.c && o.v !== e.v);
          const ins = others.map((o) => it0.m[K(o.c, o.v)]);
          const sgnTxt = ins.map((x) => `sgn(${fmt(x)})`).join("·");
          const minTxt = `min{${ins.map((x) => Math.abs(x).toFixed(2)).join(", ")}}`;
          const mu = it1.mu[K(e.c, e.v)];
          const f = { dir: "cv", c: e.c, v: e.v };
          h += line(
            `μ ${cName(e.c)}→${vName(e.v)} = ${alpha.toFixed(1)}·(−1)${s[e.c] ? "¹" : "⁰"}·${sgnTxt}·${minTxt} = <span class="${signCls(mu)}">${fmt(mu)}</span>`,
            f,
            sameFocus(f, focus) ? "focus" : ""
          );
        });
        h += `<div class="qp-note">Blue = inputs (extrinsic: the target's own message is excluded); bold = output. Click a line to trace it. c₁ is unsatisfied, so its sign flips: it tells v₁ “flip!” with strength α·|m v₂→c₁| = 2.00. With degree-2 checks the min runs over the single other neighbour; on a degree-k check it picks the weakest of k−1 inputs.</div>`;
      } else if (step === 3) {
        h += `<div class="qp-step-h">Step 3 — Variable → Check: the “Sum” step</div>`;
        h += `<div class="qp-formula">m<sub>v→c</sub><sup>(1)</sup> = L<sub>v</sub><sup>(0)</sup> + Σ<sub>c′∈N(v)∖c</sub> μ<sub>c′→v</sub><sup>(1)</sup></div>`;
        E.forEach((e) => {
          const others = E.filter((o) => o.v === e.v && o.c !== e.c);
          const terms = others.map((o) => `μ ${cName(o.c)}→${vName(o.v)} (${fmt(it1.mu[K(o.c, o.v)])})`);
          const mm = it1.m[K(e.c, e.v)];
          const f = { dir: "vc", c: e.c, v: e.v };
          h += line(
            `m ${vName(e.v)}→${cName(e.c)} = ${fmt(L0[e.v])}${terms.length ? " + " + terms.join(" + ") : " + (no other check)"} = <span class="${signCls(mm)}">${fmt(mm)}</span>`,
            f,
            sameFocus(f, focus) ? "focus" : ""
          );
        });
        h += `<div class="qp-note">Each variable adds the prior to the check messages from all <em>other</em> checks, so c₁ never hears its own −0.64 echoed back through v₂.</div>`;
      } else {
        h += `<div class="qp-step-h">Step 4 — Posterior LLR and hard decision</div>`;
        h += `<div class="qp-formula">L<sub>v</sub><sup>(1)</sup> = L<sub>v</sub><sup>(0)</sup> + Σ<sub>c∈N(v)</sub> μ<sub>c→v</sub><sup>(1)</sup>, &nbsp; ê<sub>v</sub> = 1 iff L<sub>v</sub><sup>(1)</sup> &lt; 0</div>`;
        L0.forEach((l, v) => {
          const inc = E.filter((e) => e.v === v);
          const terms = inc.map((e) => fmt(it1.mu[K(e.c, e.v)])).join(" ");
          const L = it1.L[v];
          const f = { dir: "post", v };
          h += line(
            `L${sub(v + 1)}⁽¹⁾ = ${fmt(l)} ${terms} = <span class="${signCls(L)}">${fmt(L)}</span> ⇒ ê${sub(v + 1)} = ${ehat[v]}`,
            f,
            sameFocus(f, focus) ? "focus" : ""
          );
        });
        const ok = synd.every((x, i) => x === s[i]);
        h += line(`H ê = (${synd.join(", ")}) ${ok ? "=" : "≠"} s = (${s.join(", ")}) ${ok ? "✓ converged in 1 iteration" : "✗"}`, null, ok ? "ok" : "bad");
        h += `<div class="qp-note">On a tree every message is extrinsic, so one flooding pass already yields the exact answer ê = (1, 0, 0).</div>`;
      }
      sCard.innerHTML = h;
      sCard.querySelectorAll(".qp-line.qp-click").forEach((ln) => {
        ln.addEventListener("click", () => {
          try {
            focus = JSON.parse(ln.getAttribute("data-f"));
          } catch (err) {
            focus = null;
          }
          renderGraph(true);
          renderSide();
        });
      });
    }

    const CAPTIONS = [
      "<strong>Step 0:</strong> v₁ has the weakest prior (+0.80) and touches the only unsatisfied check c₁ (red). c₂ is satisfied (green).",
      "<strong>Step 1:</strong> Every variable sends its prior L⁽⁰⁾ along each Tanner edge to its checks.",
      "<strong>Step 2 (Min):</strong> Each check replies with the sign that would fix its parity and the magnitude of its <em>weakest other</em> input. c₁ pulls v₁ to −2.00.",
      "<strong>Step 3 (Sum):</strong> Each variable sums its prior with the messages from its other checks and passes the result on.",
      "<strong>Step 4:</strong> v₁ ends negative (L = −1.20 ⇒ ê₁ = 1); v₂, v₃ stay positive, so H ê = s and BP stops."
    ];

    function show(st, animate) {
      if (cancelAnim) {
        cancelAnim();
        cancelAnim = null;
      }
      step = st;
      focus = DEFAULT_FOCUS[st] ? Object.assign({}, DEFAULT_FOCUS[st]) : null;
      gHead.innerHTML = `<span>Tanner graph · ${STEP_LABELS[st]}</span><span style="color:${C.blue};">checks c₁, c₂ · variables v₁, v₂, v₃</span>`;
      caption.innerHTML = CAPTIONS[st];
      renderSide();
      const packets = animate ? packetsForStep(st) : [];
      if (!packets.length) {
        gOver.innerHTML = "";
        renderGraph(true);
        return;
      }
      renderGraph(false);
      cancelAnim = tween(
        950,
        (t) => {
          gOver.innerHTML = t >= 1 ? "" : packets.map((pk) => svgPacket(lerp(pk.x1, pk.x2, t), lerp(pk.y1, pk.y2, t), pk.txt, pk.color)).join("");
        },
        () => {
          gOver.innerHTML = "";
          renderGraph(true);
        }
      );
    }

    const stepper = createStepper({
      labels: STEP_LABELS,
      host: root,
      intervalMs: 3200,
      onChange: (i, prev) => show(i, prev !== -1 && i !== prev)
    });
    bar.appendChild(stepper.controls);
    pillBar.appendChild(el("span", "qp-title", "Steps:"));
    pillBar.appendChild(stepper.pills);
    stepper.go(Math.max(0, Math.min(4, Number(p.step) || 0)));

    return {
      cleanup() {
        stepper.stop();
        if (cancelAnim) cancelAnim();
      }
    };
  }

  /* ==========================================================================
     2. SYMMETRIC 4-CYCLE OSCILLATION ("bp_4cycle_oscillation" / "bp_4cycle")
        H = [[1,1],[1,1]] (c1, c2 both touch v1, v2), s = (1, 1),
        symmetric priors L0 = (+2, +2). Parallel flooding min-sum.
     ========================================================================== */
  function mountBp4CycleOscillation(container, props) {
    ensureStyles();
    const p = props || {};
    const H = [
      [1, 1],
      [1, 1]
    ];
    const L0 = [2.0, 2.0];
    const s = [1, 1];
    const T_MAX = 6;
    let alpha = p.alpha === 0.8 || p.alpha === "0.8" ? 0.8 : 1.0;
    let run = minsumRun(H, L0, s, alpha, T_MAX);

    // Diamond layout: no crossing edges (cycle v1 - c1 - v2 - c2 - v1)
    const PC = [
      [190, 50],
      [190, 254]
    ];
    const PV = [
      [52, 152],
      [328, 152]
    ];
    const CENTER = [190, 152];
    const ITER_LABELS = [
      "Iter t=0 (Init)",
      "Iter t=1 (Overshoot to (1,1))",
      "Iter t=2 (Snap back to (0,0))",
      "Iter t=3 (Overshoot to (1,1))",
      "Iter t=4 (Locked Limit Cycle)"
    ];

    container.innerHTML = "";
    const root = el("div", "qp-root");
    const bar = el("div", "qp-bar");
    bar.appendChild(el("span", "qp-title", "Why Flooding Min-Sum Fails on a Symmetric 4-Cycle"));
    const pillBar = el("div", "qp-bar");
    const alphaGroup = el("div", "qp-bar-group");
    const body = el("div", "qp-body");
    const gCard = el("div", "qp-card qp-card-half");
    const gHead = el("div", "qp-card-h");
    const gWrap = el("div", "qp-svg-wrap");
    gWrap.innerHTML = `<svg class="qp-svg" viewBox="0 0 380 300" style="max-height:320px;" xmlns="http://www.w3.org/2000/svg"><defs><marker id="qp4c-arr-r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${C.red}"/></marker><marker id="qp4c-arr-g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${C.green}"/></marker></defs><g class="qp-base"></g><g class="qp-overlay"></g></svg>`;
    gCard.appendChild(gHead);
    gCard.appendChild(gWrap);
    const pCard = el("div", "qp-card qp-card-half");
    const pHead = el("div", "qp-card-h");
    const pWrap = el("div", "qp-svg-wrap");
    pCard.appendChild(pHead);
    pCard.appendChild(pWrap);
    body.appendChild(gCard);
    body.appendChild(pCard);
    const caption = el("div", "qp-caption");
    root.appendChild(bar);
    root.appendChild(pillBar);
    root.appendChild(body);
    root.appendChild(caption);
    container.appendChild(root);

    const svg = gWrap.querySelector("svg");
    const gBase = svg.querySelector(".qp-base");
    const gOver = svg.querySelector(".qp-overlay");
    let t = 0;
    let cancelAnim = null;

    function edgeGeom(c, v) {
      const [cx, cy] = PC[c];
      const [vx, vy] = PV[v];
      const dx = vx - cx;
      const dy = vy - cy;
      const len = Math.hypot(dx, dy);
      const ux = dx / len;
      const uy = dy / len;
      const mx = (cx + vx) / 2;
      const my = (cy + vy) / 2;
      let ox = mx - CENTER[0];
      let oy = my - CENTER[1];
      const ol = Math.hypot(ox, oy) || 1;
      ox /= ol;
      oy /= ol;
      // labels sit inside the diamond (toward the centre) so they never collide with node tags
      return { x1: cx + ux * 20, y1: cy + uy * 20, x2: vx - ux * 21, y2: vy - uy * 21, lx: mx - ox * 34, ly: my - oy * 34 };
    }

    function renderGraph(showLabels) {
      const it = run.iters[t];
      const K = run.key;
      let h = "";
      run.edges.forEach((e) => {
        const g = edgeGeom(e.c, e.v);
        const mu = it.mu ? it.mu[K(e.c, e.v)] : null;
        const col = mu === null ? "#475569" : mu < 0 ? C.red : Math.abs(mu) < 5e-9 ? C.grey : C.green;
        const marker = mu === null || Math.abs(mu) < 5e-9 ? "" : ` marker-end="url(#qp4c-arr-${mu < 0 ? "r" : "g"})"`;
        h += `<line x1="${g.x1.toFixed(1)}" y1="${g.y1.toFixed(1)}" x2="${g.x2.toFixed(1)}" y2="${g.y2.toFixed(1)}" stroke="${col}" stroke-width="3" stroke-linecap="round"${marker}/>`;
        if (showLabels) {
          const mPrev = run.iters[Math.max(0, t - 1)].m[K(e.c, e.v)];
          const lines = mu === null ? [`m=${fmt(it.m[K(e.c, e.v)])}`] : [`μ=${fmt(mu)}`, `m=${fmt(it.m[K(e.c, e.v)])}`];
          const colors = mu === null ? [C.blue] : [col === C.grey ? C.muted : col, C.blue];
          h += svgTag(g.lx, g.ly, lines, { color: col === C.grey ? C.grey : mu === null ? C.blue : col, colors, fs: 9.6 });
          void mPrev;
        }
      });
      const parity = (it.ehat[0] + it.ehat[1]) % 2;
      PC.forEach(([x, y], c) => {
        h += svgCheck(x, y, `c${sub(c + 1)}`, { fill: "#FEE2E2", stroke: C.red, text: "#991B1B", size: 36 });
        const ty = c === 0 ? y - 33 : y + 34;
        h += svgTag(x, ty, `s=1 · (Hê)=${parity} ✗`, { color: C.red, fs: 9.2 });
      });
      PV.forEach(([x, y], v) => {
        const L = it.L[v];
        const isOne = it.ehat[v] === 1;
        h += svgVar(x, y, `v${sub(v + 1)}`, {
          fill: isOne ? "#FEE2E2" : "#EFF6FF",
          stroke: isOne ? C.red : C.blue,
          text: isOne ? "#991B1B" : "#1E3A8A",
          r: 20
        });
        h += svgTag(x, y + 40, [`L⁽${t}⁾=${fmt(L)}`, `ê=${it.ehat[v]}`], {
          color: isOne ? C.red : C.blue,
          bg: isOne ? "#FEF2F2" : "#EFF6FF",
          fs: 9.4
        });
        h += svgText(x, y - 30, `L⁽⁰⁾=${fmt(L0[v], 1)}`, { fs: 9.5, color: C.muted });
      });
      h += svgText(190, 151, `ê⁽${t}⁾ = (${it.ehat.join(", ")})`, { fs: 12.5, weight: 700, color: it.ehat[0] ? C.red : C.blue, font: "'Space Mono',monospace" });
      h += svgText(190, 165, `Hê = (0,0) ≠ s = (1,1)`, { fs: 9, weight: 700, color: C.red, font: "'Space Mono',monospace" });
      gBase.innerHTML = h;
    }

    function renderPlot() {
      const W = 380;
      const Hh = 262;
      const padL = 44;
      const padR = 14;
      const padT = 26;
      const padB = 58;
      const plotW = W - padL - padR;
      const plotH = Hh - padT - padB;
      const maxAbs = Math.max(...run.iters.map((it) => Math.max(Math.abs(it.L[0]), Math.abs(it.L[1])))) * 1.18;
      const xOf = (k) => padL + 12 + (plotW - 24) * (k / T_MAX);
      const yOf = (L) => padT + plotH / 2 - (L / maxAbs) * (plotH / 2);
      const showUpTo = t >= 4 ? T_MAX : t;
      let h = `<svg class="qp-svg" viewBox="0 0 ${W} ${Hh}" style="max-height:300px;" xmlns="http://www.w3.org/2000/svg">`;
      h += `<rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH / 2}" fill="#EFF6FF"/>`;
      h += `<rect x="${padL}" y="${padT + plotH / 2}" width="${plotW}" height="${plotH / 2}" fill="#FEF2F2"/>`;
      h += svgText(padL + plotW - 4, padT + 12, "L &gt; 0 ⇒ ê = 0", { anchor: "end", fs: 9.5, color: C.blue, weight: 700 });
      h += svgText(padL + plotW - 4, padT + plotH - 5, "L &lt; 0 ⇒ ê = 1", { anchor: "end", fs: 9.5, color: C.red, weight: 700 });
      h += svgLine(padL, yOf(0), padL + plotW, yOf(0), { color: C.ink, w: 1.4, dash: "5,4" });
      h += `<rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="none" stroke="#475569" stroke-width="1.1"/>`;
      [maxAbs / 1.18, maxAbs / 2.36, 0, -maxAbs / 2.36, -maxAbs / 1.18].forEach((v) => {
        h += svgText(padL - 5, yOf(v) + 3.5, fmt(v, 1), { anchor: "end", fs: 9, font: "'Space Mono',monospace", color: "#475569" });
      });
      h += `<text x="12" y="${padT + plotH / 2}" text-anchor="middle" transform="rotate(-90 12 ${padT + plotH / 2})" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#334155">Posterior L⁽ᵗ⁾</text>`;
      // Current iteration highlight
      h += `<rect x="${xOf(t) - 13}" y="${padT}" width="26" height="${plotH}" fill="#FDE68A" opacity="0.45"/>`;
      const series = [
        { v: 0, color: C.red, dash: "", dy: 0, w: 3 },
        { v: 1, color: C.blue, dash: "6,4", dy: 0, w: 2 }
      ];
      series.forEach((sr) => {
        const pts = run.iters.slice(0, showUpTo + 1).map((it, k) => [xOf(k), yOf(it.L[sr.v]) + sr.dy]);
        if (pts.length > 1) {
          h += `<polyline points="${pts.map((q) => q[0].toFixed(1) + "," + q[1].toFixed(1)).join(" ")}" fill="none" stroke="${sr.color}" stroke-width="${sr.w}"${sr.dash ? ` stroke-dasharray="${sr.dash}"` : ""} opacity="0.95"/>`;
        }
      });
      run.iters.forEach((it, k) => {
        const x = xOf(k);
        const vis = k <= showUpTo;
        const future = k > 4;
        if (vis) {
          const y = yOf(it.L[0]);
          const col = it.L[0] < 0 ? C.red : C.blue;
          h += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${k === t ? 6.5 : 4.8}" fill="${col}" stroke="#FFFFFF" stroke-width="1.5" opacity="${future ? 0.7 : 1}"/>`;
          if (k === t || (t >= 4 && k <= 4)) {
            h += svgText(x, it.L[0] < 0 ? y + 16 : y - 9, fmt(it.L[0]), { fs: 9, font: "'Space Mono',monospace", color: col, weight: 700 });
          }
        }
        h += svgText(x, padT + plotH + 13, `t=${k}`, { fs: 9.3, font: "'Space Mono',monospace", color: k === t ? C.ink : "#475569", weight: 700 });
        h += svgText(x, padT + plotH + 27, vis ? `(${it.ehat.join(",")})` : "·", {
          fs: 9,
          font: "'Space Mono',monospace",
          color: vis ? (it.ehat[0] ? C.red : C.blue) : C.grey,
          weight: 700,
          opacity: future ? 0.75 : 1
        });
        h += svgText(x, padT + plotH + 41, vis ? "✗" : "", { fs: 10, color: C.red, weight: 700 });
      });
      h += svgText(padL - 4, padT + plotH + 27, "ê:", { anchor: "end", fs: 9, font: "'Space Mono',monospace", color: "#334155", weight: 700 });
      h += svgText(padL - 4, padT + plotH + 41, "Hê=s?", { anchor: "end", fs: 8.5, font: "'Space Mono',monospace", color: "#334155", weight: 700 });
      // Legend
      h += `<line x1="${padL + 6}" y1="${padT - 11}" x2="${padL + 26}" y2="${padT - 11}" stroke="${C.red}" stroke-width="3"/>`;
      h += svgText(padL + 30, padT - 7.5, "L_v₁⁽ᵗ⁾", { anchor: "start", fs: 9.5, color: C.red, weight: 700 });
      h += `<line x1="${padL + 82}" y1="${padT - 11}" x2="${padL + 102}" y2="${padT - 11}" stroke="${C.blue}" stroke-width="2" stroke-dasharray="6,4"/>`;
      h += svgText(padL + 106, padT - 7.5, "L_v₂⁽ᵗ⁾ (identical: lockstep)", { anchor: "start", fs: 9.5, color: C.blue, weight: 700 });
      h += `</svg>`;
      pWrap.innerHTML = h;
    }

    function captionFor(k) {
      const it = run.iters[k];
      const a = alpha.toFixed(1);
      if (k === 0) {
        return `<strong>t=0:</strong> True error is e=(1,0) or (0,1) — both give s=(1,1). Symmetric priors L⁽⁰⁾=+2.00 ⇒ every V→C message m=+2.00 and ê=(0,0), which violates both checks.`;
      }
      const prev = run.iters[k - 1];
      const mPrev = prev.m[run.key(0, 0)];
      const mu = it.mu[run.key(0, 0)];
      let txt = `<strong>t=${k}:</strong> Both checks see the <em>same</em> input m=${fmt(mPrev)} from the other variable and send μ = ${a}·(−1)·|${fmt(mPrev)}| = ${fmt(mu)} to <em>both</em> v₁ and v₂ ⇒ L = 2.00 ${fmt(mu)} ${fmt(mu)} = ${fmt(it.L[0])} ⇒ ê=(${it.ehat.join(",")}).`;
      if (k === 1) txt += " Both flip together (overshoot) — but 1⊕1 = 0, so neither check is fixed.";
      if (k === 2) txt += " The V→C echo m = L⁽⁰⁾ + μ from the other check shrank, so both snap back to (0,0).";
      if (k >= 4) {
        txt += alpha === 1
          ? " With plain min-sum (α=1) the swing is an exact ±2 limit cycle: (0,0) ↔ (1,1) forever."
          : " With α=0.8 the swing is damped, but symmetry is never broken: ê⁽ᵗ⁾ ∈ {(0,0),(1,1)} for every t, so H ê ≠ s forever.";
      }
      return txt;
    }

    function show(k, animate) {
      if (cancelAnim) {
        cancelAnim();
        cancelAnim = null;
      }
      t = k;
      gHead.innerHTML = `<span>4-cycle v₁–c₁–v₂–c₂–v₁ · ${ITER_LABELS[k]}</span><span style="color:${C.red};">s = (1, 1)</span>`;
      pHead.innerHTML = `<span>Trajectory L_v₁⁽ᵗ⁾, L_v₂⁽ᵗ⁾ · t = 0..${T_MAX}</span><span style="color:${C.muted};">α = ${alpha.toFixed(1)}</span>`;
      caption.innerHTML = captionFor(k);
      renderPlot();
      const it = run.iters[k];
      if (!animate || !it.mu) {
        gOver.innerHTML = "";
        renderGraph(true);
        return;
      }
      renderGraph(false);
      const packets = run.edges.map((e) => {
        const g = edgeGeom(e.c, e.v);
        const mu = it.mu[run.key(e.c, e.v)];
        return { x1: g.x1, y1: g.y1, x2: g.x2, y2: g.y2, txt: fmt(mu), color: mu < 0 ? C.red : Math.abs(mu) < 5e-9 ? C.grey : C.green };
      });
      cancelAnim = tween(
        900,
        (u) => {
          gOver.innerHTML = u >= 1 ? "" : packets.map((pk) => svgPacket(lerp(pk.x1, pk.x2, u), lerp(pk.y1, pk.y2, u), pk.txt, pk.color)).join("");
        },
        () => {
          gOver.innerHTML = "";
          renderGraph(true);
        }
      );
    }

    const stepper = createStepper({
      labels: ITER_LABELS,
      host: root,
      intervalMs: 2600,
      onChange: (i, prev) => show(i, prev !== -1 && i !== prev)
    });
    bar.appendChild(stepper.controls);

    function renderAlphaGroup() {
      alphaGroup.innerHTML = "";
      alphaGroup.appendChild(el("span", "qp-title", "Normalization:"));
      [
        { a: 1.0, lbl: "α = 1.0 (plain min-sum)" },
        { a: 0.8, lbl: "α = 0.8 (normalized)" }
      ].forEach((opt) => {
        const b = btn(opt.lbl, "qp-pill" + (alpha === opt.a ? " active" : ""));
        b.onclick = () => {
          alpha = opt.a;
          run = minsumRun(H, L0, s, alpha, T_MAX);
          renderAlphaGroup();
          show(t, false);
        };
        alphaGroup.appendChild(b);
      });
    }
    renderAlphaGroup();
    pillBar.appendChild(stepper.pills);
    pillBar.appendChild(alphaGroup);
    stepper.go(Math.max(0, Math.min(4, Number(p.iter) || 0)));

    return {
      cleanup() {
        stepper.stop();
        if (cancelAnim) cancelAnim();
      }
    };
  }

  /* ==========================================================================
     3. CYCLE BREAKING: ROW-XOR & BICLIQUE FACTOR ("gari_transform")
        Tab 1: GF(2) Row-XOR  R_c2 <- R_c2 xor R_c1  (7 ones, 3 four-cycles -> 4 ones, 0)
        Tab 2: Biclique K_{3,4} factor z_k = v1^v2^v3^v4 with defining check c_z
               (12 ones, 18 four-cycles -> 8 ones = 7 star edges + 1 link, 0)
     ========================================================================== */
  function mountGariTransform(container, props) {
    ensureStyles();
    const p = props || {};
    let tab = String(p.tab || p.mode || "").toLowerCase().includes("biclique") || p.tab === 2 || p.tab === "2" ? 1 : 0;

    const H1 = [
      [1, 1, 1, 0],
      [1, 1, 1, 1]
    ];
    const H1p = [
      [1, 1, 1, 0],
      [0, 0, 0, 1]
    ];
    const H2 = [
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1]
    ];
    const H2ext = [
      [0, 0, 0, 0, 1],
      [0, 0, 0, 0, 1],
      [0, 0, 0, 0, 1],
      [1, 1, 1, 1, 1]
    ];

    const TABS = ["1. Row-XOR Elimination (GARI)", "2. Biclique Factor Synthesis (K₃,₄ → z_k)"];
    const VIEW_LABELS = [
      ["Before", "Step: R_c₂ ← R_c₂ ⊕ R_c₁", "After"],
      ["Before", "Step: factor z_k = v₁⊕v₂⊕v₃⊕v₄", "After"]
    ];

    container.innerHTML = "";
    const root = el("div", "qp-root");
    const tabBar = el("div", "qp-bar");
    const tabs = el("div", "qp-tabs");
    tabBar.appendChild(tabs);
    const viewBar = el("div", "qp-bar");
    const subtitle = el("div", "qp-caption");
    const body = el("div", "qp-body");
    const gCard = el("div", "qp-card qp-card-graph");
    const sCard = el("div", "qp-card qp-card-side");
    body.appendChild(gCard);
    body.appendChild(sCard);
    const metrics = el("div", "qp-metrics");
    root.appendChild(tabBar);
    root.appendChild(viewBar);
    root.appendChild(subtitle);
    root.appendChild(body);
    root.appendChild(metrics);
    container.appendChild(root);

    let view = 0;
    let stepper = null;

    /* ---------------- Tab 1 graph ---------------- */
    function graphRowXor(v) {
      const PC = [
        [140, 52],
        [270, 52]
      ];
      const PV = [
        [60, 206],
        [165, 206],
        [270, 206],
        [375, 206]
      ];
      let h = "";
      const edge = (c, j, o) => {
        const [cx, cy] = PC[c];
        const [vx, vy] = PV[j];
        const dx = vx - cx;
        const dy = vy - cy;
        const len = Math.hypot(dx, dy);
        h += svgLine(cx + (dx / len) * 19, cy + (dy / len) * 19, vx - (dx / len) * 18, vy - (dy / len) * 18, o);
        return [(cx + vx) / 2, (cy + vy) / 2];
      };
      for (let j = 0; j < 3; j++) {
        edge(0, j, { color: v === 0 ? C.red : C.blue, w: v === 0 ? 3 : 2.6 });
      }
      for (let j = 0; j < 3; j++) {
        if (v === 0) edge(1, j, { color: C.red, w: 3 });
        if (v === 1) {
          const [mx, my] = edge(1, j, { color: C.red, w: 2.6, dash: "6,5", opacity: 0.55 });
          h += `<circle cx="${mx.toFixed(1)}" cy="${my.toFixed(1)}" r="8" fill="#FEF7E0" stroke="${C.gold}" stroke-width="1.5"/>`;
          h += svgText(mx.toFixed(1), (my + 4.5).toFixed(1), "⊕", { fs: 13, color: "#8D5B00", weight: 700 });
        }
      }
      if (v === 1) h += svgTag(365, 100, "⊕ deletes 3 edges", { color: "#8D5B00", fs: 8.6, bg: "#FEF7E0" });
      edge(1, 3, { color: v === 2 ? C.green : "#475569", w: v === 2 ? 3.2 : 2.6 });

      const c2label = v === 0 ? "c₂" : "c₂′";
      h += svgCheck(PC[0][0], PC[0][1], "c₁", { fill: "#DBEAFE", stroke: C.blue, text: "#1E3A8A" });
      h += svgCheck(PC[1][0], PC[1][1], c2label, {
        fill: v === 2 ? "#DCFCE7" : v === 1 ? "#FEF7E0" : "#DBEAFE",
        stroke: v === 2 ? C.green : v === 1 ? C.gold : C.blue,
        text: "#0F172A",
        glow: v === 1 ? C.gold : null,
        fs: 14
      });
      h += svgTag(PC[0][0], PC[0][1] - 32, "s₁", { color: C.blue, fs: 9.5 });
      h += svgTag(PC[1][0], PC[1][1] - 32, v === 0 ? "s₂" : "s₂′ = s₂ ⊕ s₁", { color: v === 0 ? C.blue : C.green, fs: 9.5 });
      PV.forEach(([x, y], j) => {
        const inCycle = j < 3 && v === 0;
        h += svgVar(x, y, `v${sub(j + 1)}`, {
          fill: inCycle ? "#FEE2E2" : v === 2 ? "#DCFCE7" : "#FFFFFF",
          stroke: inCycle ? C.red : v === 2 ? C.green : C.ink
        });
      });
      if (v === 0) {
        h += svgText(205, 244, "c₁, c₂ share v₁, v₂, v₃ ⇒ C(3,2) = 3 four-cycles (red)", { fs: 10.5, color: C.red, weight: 700 });
      } else if (v === 1) {
        h += svgText(205, 244, "Adding row c₁ to row c₂ (mod 2) deletes the 3 shared edges", { fs: 10.5, color: "#8D5B00", weight: 700 });
      } else {
        h += svgText(205, 244, "Tree: c₁ keeps v₁, v₂, v₃; c₂′ now only checks v₄ — 0 four-cycles", { fs: 10.5, color: C.green, weight: 700 });
      }
      return h;
    }

    /* ---------------- Tab 2 graph ---------------- */
    function graphBiclique(v) {
      const PC = [
        [110, 40],
        [205, 40],
        [300, 40]
      ];
      const PV = [
        [62, 222],
        [158, 222],
        [252, 222],
        [348, 222]
      ];
      const Z = [205, 104];
      const CZ = [205, 162];
      let h = "";
      const seg = (a, b, o, ra, rb) => {
        const dx = b[0] - a[0];
        const dy = b[1] - a[1];
        const len = Math.hypot(dx, dy);
        h += svgLine(a[0] + (dx / len) * (ra || 19), a[1] + (dy / len) * (ra || 19), b[0] - (dx / len) * (rb || 18), b[1] - (dy / len) * (rb || 18), o);
      };
      if (v < 2) {
        PC.forEach((c) =>
          PV.forEach((x) => seg(c, x, { color: C.red, w: v === 0 ? 2 : 1.6, opacity: v === 0 ? 0.9 : 0.25, dash: v === 1 ? "5,5" : "" }))
        );
      }
      if (v >= 1) {
        const col = v === 1 ? C.gold : null;
        PC.forEach((c) => seg(c, Z, { color: col || C.green, w: 3 }, 19, 18));
        seg(Z, CZ, { color: C.gold, w: 3 }, 18, 18);
        PV.forEach((x) => seg(CZ, x, { color: col || C.blue, w: 3 }, 18, 18));
      }
      PC.forEach(([x, y], i) => {
        h += svgCheck(x, y, `c${sub(i + 1)}`, {
          fill: v === 2 ? "#DCFCE7" : "#FEE2E2",
          stroke: v === 2 ? C.green : C.red,
          text: v === 2 ? "#14532D" : "#991B1B"
        });
      });
      PV.forEach(([x, y], j) => {
        h += svgVar(x, y, `v${sub(j + 1)}`, {
          fill: v === 2 ? "#EFF6FF" : "#FEE2E2",
          stroke: v === 2 ? C.blue : C.red,
          text: v === 2 ? "#1E3A8A" : "#991B1B"
        });
      });
      if (v >= 1) {
        h += svgVar(Z[0], Z[1], "z", { fill: "#FEF7E0", stroke: C.gold, text: "#8D5B00", glow: v === 1 ? C.gold : null });
        h += svgCheck(CZ[0], CZ[1], "c_z", { fill: "#FEF7E0", stroke: C.gold, text: "#8D5B00", fs: 12, glow: v === 1 ? C.gold : null });
        h += svgTag(Z[0] + 72, Z[1], "new variable z_k", { color: "#8D5B00", fs: 9 });
        h += svgTag(CZ[0] + 95, CZ[1] - 14, "c_z: v₁⊕v₂⊕v₃⊕v₄⊕z_k = 0", { color: "#8D5B00", fs: 9 });
      }
      if (v === 0) {
        h += svgText(205, 258, "K₃,₄: 12 edges, C(3,2)·C(4,2) = 18 four-cycles", { fs: 10.5, color: C.red, weight: 700 });
      } else if (v === 1) {
        h += svgText(205, 258, "v₁⊕v₂⊕v₃⊕v₄ appears in every row c₁, c₂, c₃ → name it z_k", { fs: 10.5, color: "#8D5B00", weight: 700 });
      } else {
        h += svgText(205, 258, "Star / tree: 3 + 1 + 4 = 8 edges, 0 four-cycles", { fs: 10.5, color: C.green, weight: 700 });
      }
      return h;
    }

    /* ---------------- Side (matrix) panels ---------------- */
    function sideRowXor(v) {
      let h = "";
      if (v === 0) {
        h += matrixHtml(H1, ["c₁", "c₂"], ["v₁", "v₂", "v₃", "v₄"], (r, c) => (c < 3 ? "red" : ""), "H (7 ones · three 4-cycles)", C.red);
        h += `<div class="qp-note">Rows c₁ and c₂ overlap on three columns: every pair of those columns closes a length-4 loop v–c₁–v′–c₂–v in the Tanner graph.</div>`;
      } else if (v === 1) {
        h += `<div class="qp-mat-wrap"><div class="qp-mat-title" style="color:#8D5B00;">GF(2) row operation on H and s</div><table class="qp-mat">`;
        h += `<tr><td class="lbl"></td><td class="op"></td><th>v₁</th><th>v₂</th><th>v₃</th><th>v₄</th><th>s</th></tr>`;
        h += `<tr><td class="lbl">R_c₂</td><td class="op"></td><td class="one red">1</td><td class="one red">1</td><td class="one red">1</td><td class="one">1</td><td class="one">s₂</td></tr>`;
        h += `<tr><td class="lbl">R_c₁</td><td class="op">⊕</td><td class="one blue">1</td><td class="one blue">1</td><td class="one blue">1</td><td>0</td><td class="one">s₁</td></tr>`;
        h += `<tr class="sep"><td class="lbl">R_c₂′</td><td class="op">=</td><td class="green">0</td><td class="green">0</td><td class="green">0</td><td class="one green">1</td><td class="one green">s₂⊕s₁</td></tr>`;
        h += `</table></div>`;
        h += `<div class="qp-note">1 ⊕ 1 = 0 on the shared columns: the “1 1 1” in row c₂ becomes “0 0 0”. The syndrome bit is XOR-ed the same way, so H′e = s′ ⇔ He = s (row operations are invertible).</div>`;
      } else {
        h += matrixHtml(H1p, ["c₁", "c₂′"], ["v₁", "v₂", "v₃", "v₄"], (r, c) => (r === 1 ? (c < 3 ? "green" : "blue") : ""), "H′ (4 ones · zero 4-cycles)", C.green);
        h += `<div class="qp-note">Same code, same solutions, sparser rows: no new variables or checks, and the Tanner graph is now a tree.</div>`;
      }
      return h;
    }

    function sideBiclique(v) {
      let h = "";
      if (v === 0) {
        h += matrixHtml(H2, ["c₁", "c₂", "c₃"], ["v₁", "v₂", "v₃", "v₄"], () => "red", "H block (3×4 all-ones · 12 ones · 18 4-cycles)", C.red);
        h += `<div class="qp-note">Multi-detector faults (e.g. Y = iXZ, hook errors) make a dense all-ones block: every check pair × variable pair is a 4-cycle.</div>`;
      } else if (v === 1) {
        h += matrixHtml(
          [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]
          ],
          ["c₁", "c₂", "c₃", "c_z"],
          ["v₁", "v₂", "v₃", "v₄", "z_k"],
          (r, c) => (r < 3 && c < 4 ? "red strike" : "gold"),
          "Substitute v₁⊕v₂⊕v₃⊕v₄ → z_k in rows c₁..c₃",
          "#8D5B00"
        );
        h += `<div class="qp-note">New column z_k and new defining row c_z (gold). Each struck block entry is replaced by a single 1 in column z_k.</div>`;
      } else {
        h += matrixHtml(
          H2ext,
          ["c₁", "c₂", "c₃", "c_z"],
          ["v₁", "v₂", "v₃", "v₄", "z_k"],
          (r, c) => (r < 3 ? (c === 4 ? "green" : "") : c === 4 ? "gold" : "blue"),
          "H_ext (8 ones · zero 4-cycles)",
          C.green
        );
        h += `<div class="qp-note">c₁..c₃ now read z_k only (plus their other neighbours outside the block, unchanged); c_z ties z_k to v₁..v₄.</div>`;
      }
      return h;
    }

    const SUBTITLES = [
      "<strong>Row-XOR (GF(2) elimination):</strong> when two checks share k ≥ 2 fault columns, replacing R_c₂ ← R_c₂ ⊕ R_c₁ cancels the shared ones and removes all C(k,2) four-cycles with <em>zero</em> extra nodes. This is the right tool for small overlaps such as a single 4-cycle.",
      "<strong>Why factor only large bicliques?</strong> A K_a,b block costs a·b ones; the factored star costs a+b edges through z_k (+1 link z_k–c_z). For K₂,₂: a·b = 4 = a+b, so factoring cannot save ones (it adds the link: 5) — use Row-XOR instead. For K₃,₄: 12 ones → 7 star edges (+1 link = 8 ones) and 18 → 0 four-cycles."
    ];

    function renderMetrics() {
      let items;
      if (tab === 0) {
        const Hm = view === 2 ? H1p : H1;
        items = [
          { k: "Ones in H", v: view === 1 ? "7 → 4" : String(countOnes(Hm)), c: view === 2 ? C.green : view === 0 ? C.red : C.ink },
          { k: "4-cycles", v: view === 1 ? "3 → 0" : String(count4Cycles(Hm)), c: view === 2 ? C.green : view === 0 ? C.red : C.ink },
          { k: "New nodes", v: "0", c: C.ink },
          { k: "Solutions", v: "unchanged", c: C.blue }
        ];
      } else {
        const Hm = view === 2 ? H2ext : H2;
        items = [
          { k: "Ones in H", v: view === 1 ? "12 → 8" : String(countOnes(Hm)), c: view === 2 ? C.green : view === 0 ? C.red : C.ink },
          { k: "4-cycles", v: view === 1 ? "18 → 0" : String(count4Cycles(Hm)), c: view === 2 ? C.green : view === 0 ? C.red : C.ink },
          { k: "Star edges", v: view === 0 ? "12 (dense)" : "a+b = 7 (+1)", c: C.ink },
          { k: "New nodes", v: view === 0 ? "0" : "+1 var, +1 check", c: "#8D5B00" }
        ];
      }
      metrics.innerHTML = items
        .map((m) => `<div class="qp-metric"><div class="qp-metric-k">${m.k}</div><div class="qp-metric-v" style="color:${m.c};">${m.v}</div></div>`)
        .join("");
    }

    function render() {
      const vb = tab === 0 ? "0 0 420 256" : "0 0 420 270";
      const label = VIEW_LABELS[tab][view];
      gCard.innerHTML = `<div class="qp-card-h"><span>Tanner graph · ${label}</span><span style="color:${C.blue};">squares = checks · circles = faults</span></div>` +
        `<div class="qp-svg-wrap"><svg class="qp-svg" viewBox="${vb}" style="max-height:320px;" xmlns="http://www.w3.org/2000/svg">${tab === 0 ? graphRowXor(view) : graphBiclique(view)}</svg></div>`;
      sCard.innerHTML = `<div class="qp-card-h"><span>Parity-check matrix · ${label}</span></div><div class="qp-calc">${tab === 0 ? sideRowXor(view) : sideBiclique(view)}</div>`;
      subtitle.innerHTML = SUBTITLES[tab];
      renderMetrics();
    }

    function buildViewBar() {
      if (stepper) stepper.stop();
      viewBar.innerHTML = "";
      stepper = createStepper({
        labels: VIEW_LABELS[tab],
        host: root,
        intervalMs: 2600,
        onChange: (i) => {
          view = i;
          render();
        }
      });
      viewBar.appendChild(stepper.pills);
      viewBar.appendChild(stepper.controls);
      stepper.go(0);
    }

    function renderTabs() {
      tabs.innerHTML = "";
      TABS.forEach((lbl, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "qp-tab" + (i === tab ? " active" : "");
        b.textContent = lbl;
        b.onclick = () => {
          if (tab === i) return;
          tab = i;
          renderTabs();
          buildViewBar();
        };
        tabs.appendChild(b);
      });
    }

    renderTabs();
    buildViewBar();
    if (p.view === "after" || p.stage === 2) stepper.go(2);

    return {
      cleanup() {
        if (stepper) stepper.stop();
      }
    };
  }

  /* ==========================================================================
     4. MULTI-PASS 4-STAGE TANNER GRAPH TRANSFORMATION ("multipass_funnel")
        Stage 1 raw DEM -> Stage 2 duplicate-symptom merge -> Stage 3 X/Z
        sector shattering -> Stage 4 soft (75%/25%) cross-pass prior injection
        (Lambda <- min(Lambda, (Lambda + 3 Lambda_rew)/4), as in
        multi_pass_bp_decoder.cc with --bp-reweight=soft).
     ========================================================================== */
  function mountMultipassFunnel(container, props) {
    ensureStyles();
    const p = props || {};
    const CHK = [
      { id: "c1X", x: 55, y: 54, label: "c₁ˣ", sector: "X" },
      { id: "c2X", x: 150, y: 54, label: "c₂ˣ", sector: "X" },
      { id: "c1Z", x: 270, y: 54, label: "c₁ᶻ", sector: "Z" },
      { id: "c2Z", x: 365, y: 54, label: "c₂ᶻ", sector: "Z" }
    ];
    const CHK_BY_ID = {};
    CHK.forEach((c) => {
      CHK_BY_ID[c.id] = c;
    });
    const LAM = Math.log(999); // p = 1e-3
    const LAM_REW = Math.log(0.7 / 0.3); // illustrative conditional P = 0.3
    const LAM_EFF = Math.min(LAM, (LAM + 3 * LAM_REW) / 4);

    const STAGES = [
      {
        label: "Stage 1: Raw Monolithic DEM",
        vars: [
          { id: "f1a", x: 38, label: "f₁a", kind: "dup" },
          { id: "f1b", x: 100, label: "f₁b", kind: "dup" },
          { id: "fY", x: 210, label: "f_Y", kind: "y" },
          { id: "f2a", x: 320, label: "f₂a", kind: "dup" },
          { id: "f2b", x: 382, label: "f₂b", kind: "dup" }
        ],
        edges: [
          ["c1X", "f1a"], ["c2X", "f1a"], ["c1X", "f1b"], ["c2X", "f1b"],
          ["c2X", "fY"], ["c1Z", "fY"],
          ["c1Z", "f2a"], ["c2Z", "f2a"], ["c1Z", "f2b"], ["c2Z", "f2b"]
        ],
        H: [
          [1, 1, 0, 0, 0],
          [1, 1, 1, 0, 0],
          [0, 0, 1, 1, 1],
          [0, 0, 0, 1, 1]
        ],
        cols: ["f₁a", "f₁b", "f_Y", "f₂a", "f₂b"],
        cellCls: (r, c, v) => (!v ? "" : c === 2 ? "pink" : "red"),
        matTitle: "H_raw · duplicate columns + Y-bridge",
        comps: 1,
        lines: [
          "Parallel faults f₁a, f₁b (and f₂a, f₂b) flip the <em>same</em> detectors → identical columns of H, each pair a trivial 4-cycle.",
          "f_Y (a Y = iXZ fault) touches both an X-check and a Z-check, coupling the two sectors into one component.",
          "Real scale: Color code d=11 has 2,852 raw fault columns."
        ]
      },
      {
        label: "Stage 2: Duplicate-Symptom Merge",
        vars: [
          { id: "f1", x: 70, label: "f₁*", kind: "merged", from: ["f1a", "f1b"] },
          { id: "fY", x: 210, label: "f_Y", kind: "y" },
          { id: "f2", x: 350, label: "f₂*", kind: "merged", from: ["f2a", "f2b"] }
        ],
        edges: [["c1X", "f1"], ["c2X", "f1"], ["c2X", "fY"], ["c1Z", "fY"], ["c1Z", "f2"], ["c2Z", "f2"]],
        H: [
          [1, 0, 0],
          [1, 1, 0],
          [0, 1, 1],
          [0, 0, 1]
        ],
        cols: ["f₁*", "f_Y", "f₂*"],
        cellCls: (r, c, v) => (!v ? "" : c === 1 ? "pink" : "blue"),
        matTitle: "H_merged · unique columns only",
        comps: 1,
        lines: [
          "Identical columns coalesce into one variable with the exact XOR prior p* = p<sub>a</sub>(1−p<sub>b</sub>) + p<sub>b</sub>(1−p<sub>a</sub>) (--bp-merge-duplicates).",
          "Duplicate columns halve and the trivial 4-cycles vanish; the X and Z sectors are still bridged by f_Y.",
          "Real scale: Color d=11 collapses 2,852 → 1,599 fault columns."
        ]
      },
      {
        label: "Stage 3: X/Z Sector Shattering",
        vars: [
          { id: "f1", x: 70, label: "f₁*", kind: "xs", from: ["f1a", "f1b"] },
          { id: "fYX", x: 160, label: "f_Yˣ", kind: "xs", from: ["fY"] },
          { id: "fYZ", x: 260, label: "f_Yᶻ", kind: "zs", from: ["fY"] },
          { id: "f2", x: 350, label: "f₂*", kind: "zs", from: ["f2a", "f2b"] }
        ],
        edges: [["c1X", "f1"], ["c2X", "f1"], ["c2X", "fYX"], ["c1Z", "fYZ"], ["c1Z", "f2"], ["c2Z", "f2"]],
        H: [
          [1, 0, 0, 0],
          [1, 1, 0, 0],
          [0, 0, 1, 1],
          [0, 0, 0, 1]
        ],
        cols: ["f₁*", "f_Yˣ", "f_Yᶻ", "f₂*"],
        cellCls: (r, c, v) => (!v ? "" : c < 2 ? "blue" : "orange"),
        matTitle: "Block-diagonal H_X ⊕ H_Z",
        comps: 2,
        sectors: true,
        lines: [
          "Projecting each fault onto one check basis splits f_Y into f_Yˣ and f_Yᶻ; H becomes block-diagonal (--multipass).",
          "Two disconnected bipartite graphs: Pass 0 (Z sector) and Pass 1 (X sector) are decoded separately — no cross-basis 4-cycles.",
          "Real scale: d=23 shrinks 434,705 → 50,259 columns per pass (8.6×) after merge + split."
        ]
      },
      {
        label: "Stage 4: Soft (75%/25%) Cross-Pass Reweighting",
        vars: [
          { id: "f1", x: 70, label: "f₁*", kind: "xs", from: ["f1a", "f1b"] },
          { id: "fYX", x: 160, label: "f_Yˣ", kind: "target", from: ["fY"] },
          { id: "fYZ", x: 260, label: "f_Yᶻ", kind: "fired", from: ["fY"] },
          { id: "f2", x: 350, label: "f₂*", kind: "zdone", from: ["f2a", "f2b"] }
        ],
        edges: [["c1X", "f1"], ["c2X", "f1"], ["c2X", "fYX"], ["c1Z", "fYZ"], ["c1Z", "f2"], ["c2Z", "f2"]],
        H: [
          [1, 0, 0, 0],
          [1, 1, 0, 0],
          [0, 0, 1, 1],
          [0, 0, 0, 1]
        ],
        cols: ["f₁*", "f_Yˣ", "f_Yᶻ", "f₂*"],
        cellCls: (r, c, v) => (c === 1 ? (v ? "gold" : "") : !v ? "" : c < 2 ? "blue" : "green"),
        matTitle: "Same H_X ⊕ H_Z · updated prior on f_Yˣ",
        comps: 2,
        sectors: true,
        reweight: true,
        lines: [
          "Pass 0 decodes the Z sector first; its hard decision fires f_Yᶻ (ê = 1).",
          `A reweight rule links f_Yᶻ → f_Yˣ: Pass 1 starts from Λ ← min(Λ, (Λ + 3·Λ_rew)/4) = ${LAM.toFixed(2)} → ${LAM_EFF.toFixed(2)} (75% conditional, 25% channel prior; illustrative P(Yˣ | Yᶻ) = 0.3).`,
          "Soft blending carries the Y correlation across passes without hard-pinning; it cuts LER 3.0×–14.5× in the 4-circuit ablation (--bp-reweight=soft)."
        ]
      }
    ];

    const Y_VAR = 214;
    const VAR_STYLE = {
      dup: { fill: "#FEE2E2", stroke: C.red, text: "#991B1B" },
      y: { fill: "#FCE7F3", stroke: C.pink, text: "#9D174D" },
      merged: { fill: "#DBEAFE", stroke: C.blue, text: "#1E3A8A" },
      xs: { fill: "#DBEAFE", stroke: C.blue, text: "#1E3A8A" },
      zs: { fill: "#FEF3C7", stroke: C.orange, text: "#92400E" },
      fired: { fill: "#FEE2E2", stroke: C.red, text: "#991B1B" },
      target: { fill: "#FEF7E0", stroke: C.gold, text: "#8D5B00" },
      zdone: { fill: "#DCFCE7", stroke: C.green, text: "#14532D" }
    };

    container.innerHTML = "";
    const root = el("div", "qp-root");
    const bar = el("div", "qp-bar");
    bar.appendChild(el("span", "qp-title", "Multi-Pass DEM → Tanner Graph Transformation"));
    const pillBar = el("div", "qp-bar");
    const body = el("div", "qp-body");
    const gCard = el("div", "qp-card qp-card-graph");
    const gHead = el("div", "qp-card-h");
    const gWrap = el("div", "qp-svg-wrap");
    gWrap.innerHTML = `<svg class="qp-svg" viewBox="0 0 420 286" style="max-height:330px;" xmlns="http://www.w3.org/2000/svg"><defs><marker id="qpmp-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${C.gold}"/></marker></defs><g class="qp-base"></g></svg>`;
    gCard.appendChild(gHead);
    gCard.appendChild(gWrap);
    const sCard = el("div", "qp-card qp-card-side");
    body.appendChild(gCard);
    body.appendChild(sCard);
    const metrics = el("div", "qp-metrics");
    root.appendChild(bar);
    root.appendChild(pillBar);
    root.appendChild(body);
    root.appendChild(metrics);
    container.appendChild(root);
    const gBase = gWrap.querySelector(".qp-base");

    let stage = 0;
    let cancelAnim = null;

    /** Where does node n of the new stage come from in the old stage? */
    function sourcesFor(n, oldStage) {
      const oldVars = oldStage.vars;
      const byId = oldVars.filter((o) => o.id === n.id);
      if (byId.length) return byId;
      const fromList = n.from || [];
      const viaFrom = oldVars.filter((o) => fromList.includes(o.id));
      if (viaFrom.length) return viaFrom;
      const viaRev = oldVars.filter((o) => (o.from || []).includes(n.id) || (o.from || []).some((f) => fromList.includes(f)));
      return viaRev;
    }

    function drawFrame(st, oldSt, t) {
      const S = STAGES[st];
      const O = oldSt === null ? null : STAGES[oldSt];
      let h = "";
      // Sector boxes
      const secOp = S.sectors ? (O && !O.sectors ? t : 1) : O && O.sectors ? 1 - t : 0;
      if (secOp > 0.01) {
        h += `<rect x="10" y="22" width="190" height="232" rx="10" fill="#EFF6FF" stroke="${C.blue}" stroke-width="1.4" stroke-dasharray="6,4" opacity="${secOp}"/>`;
        h += `<rect x="220" y="22" width="190" height="232" rx="10" fill="#FFFBEB" stroke="${C.orange}" stroke-width="1.4" stroke-dasharray="6,4" opacity="${secOp}"/>`;
        h += svgText(105, 272, S.reweight ? "Pass 1 · X sector (reweighted priors)" : "Pass 1 · X sector", { fs: 9.5, color: C.blue, weight: 700, opacity: secOp });
        h += svgText(315, 272, S.reweight ? "Pass 0 · Z sector (decoded first)" : "Pass 0 · Z sector", { fs: 9.5, color: C.orange, weight: 700, opacity: secOp });
      } else {
        h += svgText(210, 272, st === 0 ? "One connected component: X and Z checks bridged by f_Y" : "Still one component: f_Y bridges the X and Z sectors", { fs: 9.8, color: st === 0 ? C.red : C.pink, weight: 700 });
      }

      // Interpolated variable positions
      const pos = {};
      const ghosts = [];
      S.vars.forEach((n) => {
        const srcs = O ? sourcesFor(n, O) : [];
        if (!srcs.length) {
          pos[n.id] = { x: n.x, op: O ? t : 1 };
          return;
        }
        pos[n.id] = { x: lerp(srcs[0].x, n.x, t), op: 1 };
        srcs.slice(1).forEach((sN) => {
          ghosts.push({ x: lerp(sN.x, n.x, t), label: sN.label, kind: sN.kind, op: 1 - t });
        });
      });

      // Edges
      S.edges.forEach(([cid, vid]) => {
        const c = CHK_BY_ID[cid];
        const v = pos[vid];
        const node = S.vars.find((q) => q.id === vid);
        let color = c.sector === "X" ? C.blue : C.orange;
        let w = 2.4;
        if (st === 0) color = node.kind === "y" ? C.pink : C.red;
        if (st === 1 && node.kind === "y") color = C.pink;
        if (S.reweight && c.sector === "Z") color = C.green;
        if (S.reweight && vid === "fYX") {
          color = C.gold;
          w = 3.2;
        }
        const dx = v.x - c.x;
        const dy = Y_VAR - c.y;
        const len = Math.hypot(dx, dy);
        h += svgLine(c.x + (dx / len) * 19, c.y + (dy / len) * 19, v.x - (dx / len) * 18, Y_VAR - (dy / len) * 18, { color, w, opacity: v.op });
      });
      ghosts.forEach((g) => {
        h += svgVar(g.x, Y_VAR, g.label, Object.assign({ r: 17, fs: 12.5, opacity: g.op }, VAR_STYLE[g.kind]));
      });

      // Checks
      CHK.forEach((c) => {
        const isX = c.sector === "X";
        const done = S.reweight && !isX;
        h += svgCheck(c.x, c.y, c.label, {
          fill: done ? "#DCFCE7" : isX ? "#DBEAFE" : "#FEF3C7",
          stroke: done ? C.green : isX ? C.blue : C.orange,
          text: "#0F172A",
          fs: 13,
          size: 34
        });
      });

      // Variables
      S.vars.forEach((n) => {
        const ps = pos[n.id];
        const wide = n.label.length > 3;
        h += svgVar(ps.x, Y_VAR, n.label, Object.assign({ r: 17, rx: wide ? 22 : 0, fs: 12.5, opacity: ps.op }, VAR_STYLE[n.kind]));
      });

      // Stage-specific annotations
      const annOp = O ? t : 1;
      if (st === 0) {
        h += svgTag(69, 252, "identical columns", { color: C.red, fs: 8.6, opacity: annOp });
        h += svgTag(351, 252, "identical columns", { color: C.red, fs: 8.6, opacity: annOp });
      } else if (st === 1) {
        h += svgTag(70, 248, "p* = p_a ⊕ p_b", { color: C.blue, fs: 8.8, opacity: annOp });
        h += svgTag(350, 248, "p* = p_a ⊕ p_b", { color: C.blue, fs: 8.8, opacity: annOp });
      } else if (st === 2) {
        h += svgLine(210, 30, 210, 246, { color: C.teal, w: 1.6, dash: "5,4", opacity: annOp });
        h += svgTag(210, 140, "0 cross-basis edges", { color: C.teal, fs: 8.2, opacity: annOp, bg: "#F0FDFA" });
      } else if (st === 3) {
        h += `<path d="M 260 ${Y_VAR + 20} Q 210 ${Y_VAR + 56} 168 ${Y_VAR + 22}" fill="none" stroke="${C.gold}" stroke-width="2.4" stroke-dasharray="6,4" marker-end="url(#qpmp-arr)" opacity="${annOp}"/>`;
        h += svgTag(210, 118, "soft 75/25 prior", { color: "#8D5B00", fs: 8.6, opacity: annOp, bg: "#FEF7E0" });
        h += svgTag(160, Y_VAR - 34, `Λ ${LAM.toFixed(2)} → ${LAM_EFF.toFixed(2)}`, { color: "#8D5B00", fs: 8.8, opacity: annOp });
        h += svgTag(260, Y_VAR - 34, "ê = 1 (Pass 0)", { color: C.red, fs: 8.8, opacity: annOp });
      }
      gBase.innerHTML = h;
    }

    function renderSide() {
      const S = STAGES[stage];
      let h = `<div class="qp-card-h"><span>Parity-check matrix · ${S.label.split(":")[0]}</span></div><div class="qp-calc">`;
      h += matrixHtml(S.H, CHK.map((c) => c.label), S.cols, S.cellCls, S.matTitle, stage === 3 ? "#8D5B00" : stage === 2 ? C.teal : stage === 1 ? C.blue : C.red);
      if (S.reweight) {
        h += `<div class="qp-formula">Λ<sub>cur</sub>[f_Yˣ] ← min( Λ<sub>cur</sub>, ⌊(Λ<sub>cur</sub> + 3·Λ<sub>rew</sub>(f_Yˣ | f_Yᶻ=1)) / 4⌋ )</div>`;
      }
      S.lines.forEach((ln, i) => {
        h += `<div class="qp-line${i === S.lines.length - 1 ? " ok" : ""}">${ln}</div>`;
      });
      h += `</div>`;
      sCard.innerHTML = h;
      const cyc = count4Cycles(S.H);
      const items = [
        { k: "Fault columns", v: String(S.H[0].length) },
        { k: "Ones in H", v: String(countOnes(S.H)) },
        { k: "4-cycles", v: String(cyc), c: cyc ? C.red : C.green },
        { k: "Components", v: S.comps === 1 ? "1 (coupled)" : "2 (X | Z)", c: S.comps === 1 ? C.red : C.green },
        { k: "Priors updated", v: S.reweight ? "1 (f_Yˣ)" : "0", c: S.reweight ? "#8D5B00" : C.ink }
      ];
      metrics.innerHTML = items
        .map((m) => `<div class="qp-metric"><div class="qp-metric-k">${m.k}</div><div class="qp-metric-v" style="color:${m.c || C.ink};">${m.v}</div></div>`)
        .join("");
    }

    function show(st, prev) {
      if (cancelAnim) {
        cancelAnim();
        cancelAnim = null;
      }
      const oldSt = prev === -1 || prev === st ? null : prev;
      stage = st;
      gHead.innerHTML = `<span>Tanner graph · ${STAGES[st].label}</span><span style="color:${C.blue};">squares = checks · circles = faults</span>`;
      renderSide();
      if (oldSt === null) {
        drawFrame(st, null, 1);
        return;
      }
      cancelAnim = tween(1100, (t) => drawFrame(st, oldSt, t), null);
    }

    const stepper = createStepper({
      labels: STAGES.map((s) => s.label),
      host: root,
      intervalMs: 3400,
      onChange: show
    });
    bar.appendChild(stepper.controls);
    pillBar.appendChild(stepper.pills);
    stepper.go(Math.max(0, Math.min(3, (Number(p.stage) || 1) - 1)));

    return {
      cleanup() {
        stepper.stop();
        if (cancelAnim) cancelAnim();
      }
    };
  }

  /* ==========================================================================
     Dispatcher + registration
     ========================================================================== */
  const VIEWS = {
    bp_minsum_stepper: mountBpMinsumStepper,
    bp_mechanics: mountBpMinsumStepper,
    bp_4cycle_oscillation: mountBp4CycleOscillation,
    bp_4cycle: mountBp4CycleOscillation,
    gari_transform: mountGariTransform,
    multipass_funnel: mountMultipassFunnel
  };

  function mountQecPrimitives(container, props) {
    const v = String((props && (props.view || props.mode || props.preset)) || "bp_minsum_stepper").toLowerCase();
    const fn = VIEWS[v] || mountBpMinsumStepper;
    return fn(container, props);
  }

  window.TesseractViz = window.TesseractViz || {};
  Object.keys(VIEWS).forEach((id) => {
    window.TesseractViz[id] = VIEWS[id];
  });
  window.TesseractViz["qec_primitives"] = mountQecPrimitives;
  window.TesseractQecPrimitives = { views: VIEWS, mount: mountQecPrimitives, minsumRun, count4Cycles, countOnes };

  function registerAll() {
    if (window.SlideEngine && typeof window.SlideEngine.registerAnimation === "function") {
      Object.keys(VIEWS).forEach((id) => window.SlideEngine.registerAnimation(id, VIEWS[id]));
      window.SlideEngine.registerAnimation("qec_primitives", mountQecPrimitives);
    }
  }
  registerAll();
  document.addEventListener("DOMContentLoaded", registerAll);
})();
