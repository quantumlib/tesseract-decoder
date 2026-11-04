/* ==========================================================================
   Animation Module: bp_osd_isd_pipeline
   Clean, Spacious Static 2D Mechanism Diagrams (with CLI Flags) +
   Multi-Circuit 4-Family Ablation SVG Charts Separated into Four Visuals:
     - "bp_comp"  / "ablation_bp_comp"  (Slide 8):  BP Computational Optimizations (3 Steps A1..A3, 1 Plot: Speedup & ms/shot)
     - "bp_algo"  / "ablation_bp_algo"  (Slide 9):  BP Algorithmic Optimizations (3 Steps B1..B3, Dual Plot: BP Conv % & LER/round)
     - "osd_comp" / "ablation_osd_comp" (Slide 11): OSD Computational Optimizations (3 Steps C1..C3, 1 Plot: Log Speedup, 0.0% LER Loss)
     - "osd_algo" / "ablation_osd_algo" (Slide 12): OSD Algorithmic Optimizations (4 Steps D0..D3, LER/round & Error Reduction)
   Zero continuous animation, zero toggle buttons, zero Tesseract-Hybrid references.
   ========================================================================== */

(function () {
  /* ------------------------------------------------------------------------
     4-Circuit Flag-Verified Ablation Dataset (Exact Match to ablation_4circuits_results.json)
     Circuits:
       - cc_d11:     Superdense Color Code d=11, r=11, p=0.001
       - bb_d6:      Bivariate Bicycle [[72, 12, 6]], r=6, p=0.001
       - mitten_d10: Mitten Code [[150, 30, 10]], r=10, p=0.001
       - hyp_iter40: Hyperbolic Octagon iter=40 [[288, 12, 8]], R=16
     ------------------------------------------------------------------------ */
  const ABLATION_DATA = {
    circuits: [
      { id: "cc_d11", label: "Color Code d=11", sub: "d=11, r=11, p=0.001" },
      { id: "bb_d6", label: "BB ⟦72, 12, 6⟧", sub: "r=6, p=0.001" },
      { id: "mitten_d10", label: "Mitten ⟦150, 30, 10⟧", sub: "r=10, p=0.001" },
      { id: "hyp_iter40", label: "Hyperbolic iter=40", sub: "⟦288, 12, 8⟧, R=16" }
    ],
    // Progression A: BP Computational (3 Steps A1..A3: Speedup vs A1 & BP ms/shot; A2 has 0.0% LER change)
    bp_comp: {
      steps: [
        { key: "A1", name: "A1: Scalar Monolithic BP (1.00×)", flag: "--no-multipass --no-bp-simd-batch", color: "#64748B" },
        { key: "A2", name: "A2: +16-Lane AVX2 SIMD Batching (0% LER Δ)", flag: "--bp-simd-batch", color: "#1967D2" },
        { key: "A3", name: "A3: +Multi-Pass DEM/Tanner Compression", flag: "--multipass (default)", color: "#1E8E3E" }
      ],
      byCircuit: {
        cc_d11: [
          { speedup: 1.00, ms: 166.55 },
          { speedup: 13.53, ms: 12.31 },
          { speedup: 101.31, ms: 1.64 }
        ],
        bb_d6: [
          { speedup: 1.00, ms: 93.18 },
          { speedup: 12.24, ms: 7.62 },
          { speedup: 98.56, ms: 0.95 }
        ],
        mitten_d10: [
          { speedup: 1.00, ms: 1026.72 },
          { speedup: 15.28, ms: 67.21 },
          { speedup: 135.76, ms: 7.56 }
        ],
        hyp_iter40: [
          { speedup: 1.00, ms: 634.90 },
          { speedup: 13.75, ms: 46.17 },
          { speedup: 77.94, ms: 8.15 }
        ]
      }
    },
    // Progression B: BP Algorithmic (3 Steps B1..B3: Primary Pure-BP Convergence % AND End-to-End LER Reduction vs B1)
    bp_algo: {
      steps: [
        { key: "B1", name: "B1: Flooding + Unweighted", flag: "--no-bp-serial --bp-reweight=none", color: "#64748B" },
        { key: "B2", name: "B2: +Serial Check Schedule", flag: "--bp-serial --bp-reweight=none", color: "#1967D2" },
        { key: "B3", name: "B3: +Soft-Power Edge Reweighting", flag: "--bp-serial --bp-reweight=soft (default)", color: "#1E8E3E" }
      ],
      byCircuit: {
        cc_d11: [
          { conv: 36.60, ler_gain: 1.00, ler_top: "1.00×", ler_str: "3.33e-5" },
          { conv: 41.49, ler_gain: 1.71, ler_top: "1.71×", ler_str: "1.94e-5" },
          { conv: 44.49, ler_gain: 4.00, ler_top: "4.00×", ler_str: "8.32e-6" }
        ],
        bb_d6: [
          { conv: 75.24, ler_gain: 1.00, ler_top: "1.00×", ler_str: "2.44e-4" },
          { conv: 82.84, ler_gain: 1.20, ler_top: "1.20×", ler_str: "2.04e-4" },
          { conv: 82.89, ler_gain: 3.00, ler_top: "3.00×", ler_str: "8.14e-5" }
        ],
        mitten_d10: [
          { conv: 22.18, ler_gain: 1.00, ler_top: "1.00×", ler_str: "1.77e-4" },
          { conv: 90.52, ler_gain: 1.76, ler_top: "1.76×", ler_str: "1.01e-4" },
          { conv: 91.89, ler_gain: 14.52, ler_top: "14.52×", ler_str: "1.22e-5" }
        ],
        hyp_iter40: [
          { conv: 1.61, ler_gain: 1.00, ler_top: "1.00×", ler_str: "1.24e-3" },
          { conv: 6.15, ler_gain: 1.00, ler_top: "1.00×", ler_str: "1.24e-3" },
          { conv: 8.37, ler_gain: 3.69, ler_top: "3.69×", ler_str: "3.37e-4" }
        ]
      }
    },
    // Progression C: OSD Computational (3 Steps C1..C3: Isolated OSD Speedup vs C1 & ms/OSD-shot; 0.0% LER Change)
    osd_comp: {
      steps: [
        { key: "C1", name: "C1: Full RREF + Full Sort (1.00×)", flag: "--no-osd-early-exit --no-osd-col-truncation --no-osd-partial-sort", color: "#64748B" },
        { key: "C2", name: "C2: +Dynamic Zero-Remainder Early Exit (0% LER Δ)", flag: "--osd-early-exit", color: "#1967D2" },
        { key: "C3", name: "C3: +Horizon-Bounded Partial Sort (K=1.2M, nth_element)", flag: "--osd-col-truncation --osd-partial-sort (default)", color: "#1E8E3E" }
      ],
      byCircuit: {
        cc_d11: [
          { speedup: 1.00, ms: 98.96 },
          { speedup: 7.77, ms: 12.73 },
          { speedup: 14.89, ms: 6.64 }
        ],
        bb_d6: [
          { speedup: 1.00, ms: 16.51 },
          { speedup: 1.96, ms: 8.42 },
          { speedup: 2.49, ms: 6.64 }
        ],
        mitten_d10: [
          { speedup: 1.00, ms: 192.47 },
          { speedup: 3.83, ms: 50.32 },
          { speedup: 4.98, ms: 38.69 }
        ],
        hyp_iter40: [
          { speedup: 1.00, ms: 1314.35 },
          { speedup: 16.01, ms: 82.10 },
          { speedup: 22.35, ms: 58.80 }
        ]
      }
    },
    // Progression D: OSD Algorithmic (4 Steps D0..D3: Cumulative LER Reduction vs D0 Pure BP, LER/round, & Total ms/shot)
    osd_algo: {
      steps: [
        { key: "D0", name: "D0: Pure BP (No OSD)", flag: "--osd-order -1", color: "#64748B" },
        { key: "D1", name: "D1: +Naive OSD-0 (Final Iter L^(T))", flag: "--no-osd-min-syndrome --osd-order 0", color: "#1967D2" },
        { key: "D2", name: "D2: +Trajectory Seed Diversification (L^(t*) + 2 Relay Legs)", flag: "--osd-min-syndrome --osd-relay", color: "#00796B" },
        { key: "D3", name: "D3: +Higher-Order OSD(O10/W2) & α-ISD", flag: "--osd-alpha-isd --osd-order 10 (default)", color: "#1E8E3E" }
      ],
      byCircuit: {
        cc_d11: [
          { ler_gain: 1.00, d1_gain: 0.00, ler_str: "1.13e-2", ms: 1.68 },
          { ler_gain: 407.61, d1_gain: 1.00, ler_str: "2.78e-5", ms: 1.75 },
          { ler_gain: 582.36, d1_gain: 1.43, ler_str: "1.94e-5", ms: 3.29 },
          { ler_gain: 1358.98, d1_gain: 3.33, ler_str: "8.32e-6", ms: 6.02 }
        ],
        bb_d6: [
          { ler_gain: 1.00, d1_gain: 0.29, ler_str: "2.85e-4", ms: 0.91 },
          { ler_gain: 3.50, d1_gain: 1.00, ler_str: "8.14e-5", ms: 0.98 },
          { ler_gain: 3.50, d1_gain: 1.00, ler_str: "8.14e-5", ms: 1.69 },
          { ler_gain: 3.50, d1_gain: 1.00, ler_str: "8.14e-5", ms: 2.15 }
        ],
        mitten_d10: [
          { ler_gain: 1.00, d1_gain: 0.58, ler_str: "2.11e-4", ms: 7.67 },
          { ler_gain: 1.73, d1_gain: 1.00, ler_str: "1.22e-4", ms: 7.55 },
          { ler_gain: 9.87, d1_gain: 5.72, ler_str: "2.14e-5", ms: 9.78 },
          { ler_gain: 17.28, d1_gain: 10.01, ler_str: "1.22e-5", ms: 11.34 }
        ],
        hyp_iter40: [
          { ler_gain: 1.00, d1_gain: 0.02, ler_str: "4.76e-2", ms: 8.17 },
          { ler_gain: 50.44, d1_gain: 1.00, ler_str: "9.44e-4", ms: 8.79 },
          { ler_gain: 71.85, d1_gain: 1.42, ler_str: "6.63e-4", ms: 28.68 },
          { ler_gain: 141.12, d1_gain: 2.80, ler_str: "3.37e-4", ms: 81.79 }
        ]
      }
    }
  };

  function ensureSharedStyles() {
    if (document.getElementById("tesseract-comp-algo-styles")) return;
    const style = document.createElement("style");
    style.id = "tesseract-comp-algo-styles";
    style.textContent = `
      :root {
        --qai-cream: #FBF9F5;
        --qai-cream-card: #F4F1EA;
        --qai-hairline: #DFDCD4;
        --qai-charcoal: #1E2022;
        --qai-dark-grey: #2D3134;
        --qai-muted: #5F6368;
        --qai-blue: #4285F4;
        --qai-deep-blue: #1967D2;
        --qai-gari-blue: #0018F0;
        --qai-gold: #FBBC04;
        --qai-orange: #D97706;
        --qai-green: #1E8E3E;
        --qai-red: #D93025;
        --qai-teal: #00796B;
        --qai-white: #FFFFFF;
        --qai-badge-blue: #E8F0FE;
        --qai-badge-gold: #FEF7E0;
        --qai-badge-green: #E6F4EA;
        --qai-badge-red: #FCE8E6;
      }
      .ca-root {
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        gap: 9px;
        font-family: "DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: var(--qai-charcoal);
        box-sizing: border-box;
      }
      .ca-card {
        flex: 1;
        background: #FFFFFF;
        border: 1px solid var(--qai-hairline);
        border-radius: 10px;
        padding: 11px 15px;
        display: flex;
        flex-direction: column;
        gap: 7px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
      }
      .ca-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 6px;
        padding-bottom: 5px;
        border-bottom: 1px solid var(--qai-hairline);
      }
      .ca-card-title {
        font-family: "Space Mono", monospace;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
      }
      .ca-badge {
        font-family: "Space Mono", monospace;
        font-size: 0.66rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
      }
      .ca-svg-wrap {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
      }
      .ca-svg-wrap svg {
        width: 100%;
        height: auto;
        overflow: visible;
      }
      .ca-legend-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
        justify-content: center;
        font-size: 0.70rem;
        color: var(--qai-charcoal);
        padding-top: 2px;
      }
      .ca-legend-item {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-weight: 600;
      }
      .ca-legend-swatch {
        width: 11px;
        height: 11px;
        border-radius: 2px;
        border: 1px solid rgba(0,0,0,0.2);
      }
      .ca-flag-pill {
        font-family: "Space Mono", monospace;
        font-size: 0.63rem;
        background: #F1F5F9;
        color: #334155;
        padding: 1px 5px;
        border-radius: 3px;
        border: 1px solid #CBD5E1;
      }
    `;
    document.head.appendChild(style);
  }

  function renderMath(container) {
    if (window.renderMathInElement) {
      try {
        window.renderMathInElement(container, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false }
          ],
          throwOnError: false
        });
      } catch (e) {}
    }
  }

  function fmtMs(ms) {
    if (ms >= 100) return `${ms.toFixed(0)}ms`;
    if (ms >= 10) return `${ms.toFixed(1)}ms`;
    return `${ms.toFixed(2)}ms`;
  }

  /* ------------------------------------------------------------------------
     Helper: Render Single-Panel Multi-Circuit SVG Grouped Bar Chart
     ------------------------------------------------------------------------ */
  function buildSingleGroupedChartSvg(opts) {
    const circuits = ABLATION_DATA.circuits;
    const steps = opts.steps;
    const byCircuit = opts.byCircuit;
    const logScale = Boolean(opts.logScale);
    const yLabel = opts.yLabel || "Speedup (×)";
    const maxVal = opts.maxVal || 10.0;
    const minVal = opts.minVal || 0.7;

    const W = 860, H = 172;
    const padL = 56, padR = 16, padT = 24, padB = 34;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    function yPos(val) {
      if (logScale) {
        const minLog = Math.log10(minVal);
        const maxLog = Math.log10(maxVal * 1.45);
        const vLog = Math.log10(Math.max(minVal, val));
        return padT + plotH * (1 - (vLog - minLog) / (maxLog - minLog));
      }
      return padT + plotH * (1 - Math.min(val, maxVal * 1.18) / (maxVal * 1.18));
    }

    const yBase = padT + plotH;
    let svg = `<svg viewBox="0 0 ${W} ${H}" style="max-height:178px;" xmlns="http://www.w3.org/2000/svg">`;

    // Horizontal gridlines
    const ticks = opts.ticks || (logScale ? [1, 10, 100] : [0, 2, 4, 6, 8, 10]);
    for (const t of ticks) {
      const y = yPos(t);
      if (y < padT - 4 || y > yBase + 2) continue;
      svg += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="3,3" />`;
      svg += `<text x="${padL - 6}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" font-family="'Space Mono',monospace" font-size="9" fill="#64748B">${t}×</text>`;
    }

    // Y-axis title
    svg += `<text x="14" y="${(padT + plotH / 2).toFixed(1)}" text-anchor="middle" transform="rotate(-90 14 ${(padT + plotH / 2).toFixed(1)})" font-family="'Space Mono',monospace" font-size="9.0" font-weight="700" fill="#334155">${yLabel}</text>`;

    // Baseline axis line
    svg += `<line x1="${padL}" y1="${yBase}" x2="${W - padR}" y2="${yBase}" stroke="#94A3B8" stroke-width="1.3" />`;

    const groupW = plotW / circuits.length;
    const barW = Math.min(36, (groupW - 36) / steps.length);

    circuits.forEach((c, cIdx) => {
      const gCenter = padL + groupW * (cIdx + 0.5);
      const rows = byCircuit[c.id] || [];

      rows.forEach((row, sIdx) => {
        const val = opts.getValue(row);
        const topLabel = opts.getTopLabel(row);
        const subLabel = opts.getSubLabel ? opts.getSubLabel(row) : "";
        const x = gCenter + (sIdx - (steps.length - 1) / 2) * (barW + 4) - barW / 2;
        const y = yPos(val);
        const h = Math.max(2, yBase - y);
        const color = steps[sIdx].color;
        const isFinal = sIdx === steps.length - 1;

        svg += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" rx="2.5" fill="${color}" stroke="#1E2022" stroke-width="${isFinal ? 1.1 : 0.6}" />`;
        svg += `<text x="${(x + barW / 2).toFixed(1)}" y="${(y - (subLabel ? 11 : 4)).toFixed(1)}" text-anchor="middle" font-family="'Space Mono',monospace" font-size="7.8" font-weight="${isFinal ? 700 : 600}" fill="${isFinal ? '#14532D' : '#1E2022'}">${topLabel}</text>`;
        if (subLabel) {
          svg += `<text x="${(x + barW / 2).toFixed(1)}" y="${(y - 2.5).toFixed(1)}" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="7.3" font-weight="600" fill="#5F6368">${subLabel}</text>`;
        }
      });

      // X-axis group label
      svg += `<text x="${gCenter.toFixed(1)}" y="${yBase + 14}" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#1E2022">${c.label}</text>`;
      svg += `<text x="${gCenter.toFixed(1)}" y="${yBase + 26}" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="8.2" fill="#64748B">${c.sub}</text>`;
    });

    svg += `</svg>`;
    return svg;
  }

  function buildLegendHtml(steps) {
    return `<div class="ca-legend-bar">` +
      steps.map(s => `
        <span class="ca-legend-item">
          <span class="ca-legend-swatch" style="background:${s.color};"></span>
          <span>${s.name}</span>
          <code class="ca-flag-pill">${s.flag}</code>
        </span>
      `).join("") +
      `</div>`;
  }

  /* ==========================================================================
     1. BP COMPUTATIONAL OPTIMIZATIONS ("bp_comp" / "ablation_bp_comp")
     ========================================================================== */
  function renderBpComp(container) {
    const d = ABLATION_DATA.bp_comp;
    const allSpeedups = Object.values(d.byCircuit).flat().map(r => r.speedup);
    const maxSp = Math.max(...allSpeedups, 150.0);

    const chartSvg = buildSingleGroupedChartSvg({
      steps: d.steps,
      byCircuit: d.byCircuit,
      logScale: true,
      minVal: 0.7,
      yLabel: "BP Speedup (×, log)",
      maxVal: maxSp,
      ticks: [1, 10, 50, 100],
      getValue: r => r.speedup,
      getTopLabel: r => `${r.speedup >= 10 ? r.speedup.toFixed(1) : r.speedup.toFixed(2)}×`,
      getSubLabel: r => fmtMs(r.ms)
    });

    container.innerHTML = `
      <div class="ca-root">
        <!-- Top Card: 3-Step BP Computational Pipeline (A1 -> A2 -> A3) -->
        <div class="ca-card" style="flex:0.95;">
          <div class="ca-card-header">
            <span class="ca-card-title" style="color:var(--qai-deep-blue);">
              1. BP Computational Pipeline: Scalar Monolithic &rarr; 16-Lane AVX2 SIMD Batching &rarr; Multi-Pass DEM/Tanner Compression
            </span>
            <span class="ca-badge" style="background:var(--qai-badge-green); color:var(--qai-green);">
              16-Lane SIMD: 0.0% LER Change &nbsp;&middot;&nbsp; Multi-Pass: Shrinks Graph &amp; Cuts LER
            </span>
          </div>
          <div class="ca-svg-wrap">
            <svg viewBox="0 0 860 154" style="max-height:158px;" xmlns="http://www.w3.org/2000/svg">
              <!-- Box 1: A1 Scalar Monolithic BP -->
              <rect x="8" y="4" width="262" height="145" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="139" y="23" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.2" font-weight="700" fill="#475569">
                A1. SCALAR MONOLITHIC BP
              </text>
              <rect x="20" y="31" width="238" height="24" rx="4" fill="#F1F5F9" stroke="#64748B" stroke-width="1.3" />
              <text x="139" y="47" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.0" font-weight="700" fill="#334155">
                --no-multipass --no-bp-simd-batch
              </text>
              <text x="139" y="73" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10.5" font-weight="600" fill="#1E2022">
                Single-shot scalar min-sum on full DEM
              </text>
              <text x="139" y="92" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#5F6368">
                93.2 ms &ndash; 1,026.7 ms per shot across 4 codes
              </text>
              <text x="139" y="110" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#5F6368">
                Uncompressed cross-basis Tanner graph
              </text>
              <text x="139" y="136" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.8" font-weight="700" fill="#475569">
                1.00&times; Throughput Baseline Reference
              </text>

              <!-- Arrow 1 -> 2 -->
              <line x1="273" y1="76" x2="293" y2="76" stroke="#1967D2" stroke-width="2.2" />
              <polygon points="296,76 288,71 288,81" fill="#1967D2" />

              <!-- Box 2: A2 16-Lane AVX2 SIMD Batching -->
              <rect x="299" y="4" width="262" height="145" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="430" y="23" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.2" font-weight="700" fill="#1967D2">
                A2. +16-LANE AVX2 SIMD BATCHING
              </text>
              <rect x="317" y="31" width="226" height="24" rx="4" fill="#E8F0FE" stroke="#1967D2" stroke-width="1.3" />
              <text x="430" y="47" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#1967D2">
                Flag: --bp-simd-batch (16 &times; int32 = 64B)
              </text>
              <text x="430" y="73" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10.5" font-weight="600" fill="#1E2022">
                Flat cache-aligned bidirectional CSR arrays
              </text>
              <text x="430" y="92" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#5F6368">
                12.2&times;&ndash;15.3&times; faster than scalar across all 4 codes
              </text>
              <text x="430" y="110" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#5F6368">
                On-the-fly Q8 fixed-point min-sum updates
              </text>
              <text x="430" y="136" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.8" font-weight="700" fill="#1967D2">
                0.0% LER Change (Pure Throughput)
              </text>

              <!-- Arrow 2 -> 3 -->
              <line x1="564" y1="76" x2="584" y2="76" stroke="#1967D2" stroke-width="2.2" />
              <polygon points="587,76 579,71 579,81" fill="#1967D2" />

              <!-- Box 3: A3 Multi-Pass DEM/Tanner Compression -->
              <rect x="590" y="4" width="262" height="145" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="721" y="23" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.0" font-weight="700" fill="#1E8E3E">
                A3. +MULTI-PASS GRAPH COMPRESSION
              </text>
              <rect x="604" y="31" width="234" height="24" rx="4" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="1.3" />
              <text x="721" y="47" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.3" font-weight="700" fill="#14532D">
                Flags: --multipass --bp-merge-duplicates
              </text>
              <text x="721" y="73" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10.5" font-weight="600" fill="#1E2022">
                Single-basis projection merges duplicate cols
              </text>
              <text x="721" y="92" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#5F6368">
                77.9&times;&ndash;135.8&times; cumulative BP speedup vs. A1!
              </text>
              <text x="721" y="110" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#5F6368">
                5.7&times;&ndash;8.9&times; additional step speedup over A2
              </text>
              <text x="721" y="136" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.8" font-weight="700" fill="#1E8E3E">
                Boosts Speed AND Breaks Cross-Basis 4-Cycles
              </text>
            </svg>
          </div>
        </div>

        <!-- Bottom Card: 1 Plot Only (Multi-Circuit BP Speedup & ms/shot) -->
        <div class="ca-card" style="flex:1.05;">
          <div class="ca-card-header">
            <span class="ca-card-title" style="color:var(--qai-green);">
              2. Multi-Circuit BP Computational Speedup Across 4 Code Families (Log Scale, with BP ms/shot)
            </span>
            <span class="ca-badge" style="background:var(--qai-badge-blue); color:var(--qai-deep-blue);">
              77.9&times;&ndash;135.8&times; Cumulative BP Speedup &nbsp;&middot;&nbsp; A2 (SIMD): 0.0% LER Change
            </span>
          </div>
          <div class="ca-svg-wrap">
            ${chartSvg}
          </div>
          ${buildLegendHtml(d.steps)}
        </div>
      </div>
    `;
    renderMath(container);
  }

  /* ==========================================================================
     2. BP ALGORITHMIC OPTIMIZATIONS ("bp_algo" / "ablation_bp_algo")
     ========================================================================== */
  function renderBpAlgo(container) {
    const d = ABLATION_DATA.bp_algo;
    const circuits = ABLATION_DATA.circuits;
    const steps = d.steps;

    // Build dual-panel SVG: Left = Primary Pure-BP Convergence Rate (%), Right = LER Reduction Factor (x, log)
    const W = 860, H = 172;
    const padT = 24, padB = 34;
    const plotH = H - padT - padB;
    const yBase = padT + plotH;

    // Left panel geometry (x: 46 .. 412)
    const L1 = 46, R1 = 412, W1 = R1 - L1;
    // Right panel geometry (x: 472 .. 848)
    const L2 = 472, R2 = 848, W2 = R2 - L2;

    const minLog2 = Math.log10(0.6);
    const maxLog2 = Math.log10(25.0);
    function yPos2(val) {
      const vLog = Math.log10(Math.max(0.6, val));
      return padT + plotH * (1 - (vLog - minLog2) / (maxLog2 - minLog2));
    }

    let dualSvg = `<svg viewBox="0 0 ${W} ${H}" style="max-height:178px;" xmlns="http://www.w3.org/2000/svg">`;

    // Left Panel: Pure-BP Convergence (%)
    dualSvg += `<text x="${(L1 + R1) / 2}" y="12" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.3" font-weight="700" fill="#1967D2">(A) PRIMARY PURE-BP CONVERGENCE RATE (%) &rarr; CUTS OSD LOAD</text>`;
    for (const t of [0, 25, 50, 75, 100]) {
      const y = padT + plotH * (1 - t / 112);
      dualSvg += `<line x1="${L1}" y1="${y.toFixed(1)}" x2="${R1}" y2="${y.toFixed(1)}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="3,3" />`;
      dualSvg += `<text x="${L1 - 5}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-family="'Space Mono',monospace" font-size="8.5" fill="#64748B">${t}%</text>`;
    }
    dualSvg += `<line x1="${L1}" y1="${yBase}" x2="${R1}" y2="${yBase}" stroke="#94A3B8" stroke-width="1.3" />`;

    const gW1 = W1 / circuits.length;
    const bW1 = Math.min(22, (gW1 - 18) / steps.length);
    circuits.forEach((c, cIdx) => {
      const gCenter = L1 + gW1 * (cIdx + 0.5);
      const rows = d.byCircuit[c.id] || [];
      rows.forEach((row, sIdx) => {
        const x = gCenter + (sIdx - (steps.length - 1) / 2) * (bW1 + 3) - bW1 / 2;
        const y = padT + plotH * (1 - row.conv / 112);
        const h = Math.max(2, yBase - y);
        const isFinal = sIdx === steps.length - 1;
        dualSvg += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bW1.toFixed(1)}" height="${h.toFixed(1)}" rx="2" fill="${steps[sIdx].color}" stroke="#1E2022" stroke-width="${isFinal ? 1.0 : 0.5}" />`;
        dualSvg += `<text x="${(x + bW1 / 2).toFixed(1)}" y="${(y - 3).toFixed(1)}" text-anchor="middle" font-family="'Space Mono',monospace" font-size="7.2" font-weight="${isFinal ? 700 : 600}" fill="${isFinal ? '#14532D' : '#1E2022'}">${row.conv.toFixed(1)}%</text>`;
      });
      dualSvg += `<text x="${gCenter.toFixed(1)}" y="${yBase + 14}" text-anchor="middle" font-family="'Space Mono',monospace" font-size="8.5" font-weight="700" fill="#1E2022">${c.label}</text>`;
      dualSvg += `<text x="${gCenter.toFixed(1)}" y="${yBase + 25}" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="7.6" fill="#64748B">${c.sub}</text>`;
    });

    // Right Panel: End-to-End LER Reduction Factor (x, log) & LER/round
    dualSvg += `<text x="${(L2 + R2) / 2}" y="12" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.3" font-weight="700" fill="#1E8E3E">(B) LER REDUCTION FACTOR vs. B1 (&times;, LOG) &amp; LER/ROUND</text>`;
    for (const t of [1, 2, 5, 10, 15]) {
      const y = yPos2(t);
      dualSvg += `<line x1="${L2}" y1="${y.toFixed(1)}" x2="${R2}" y2="${y.toFixed(1)}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="3,3" />`;
      dualSvg += `<text x="${L2 - 5}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-family="'Space Mono',monospace" font-size="8.5" fill="#64748B">${t}×</text>`;
    }
    dualSvg += `<line x1="${L2}" y1="${yBase}" x2="${R2}" y2="${yBase}" stroke="#94A3B8" stroke-width="1.3" />`;

    const gW2 = W2 / circuits.length;
    const bW2 = Math.min(22, (gW2 - 18) / steps.length);
    circuits.forEach((c, cIdx) => {
      const gCenter = L2 + gW2 * (cIdx + 0.5);
      const rows = d.byCircuit[c.id] || [];
      rows.forEach((row, sIdx) => {
        const x = gCenter + (sIdx - (steps.length - 1) / 2) * (bW2 + 3) - bW2 / 2;
        const y = yPos2(row.ler_gain);
        const h = Math.max(2, yBase - y);
        const isFinal = sIdx === steps.length - 1;
        dualSvg += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bW2.toFixed(1)}" height="${h.toFixed(1)}" rx="2" fill="${steps[sIdx].color}" stroke="#1E2022" stroke-width="${isFinal ? 1.0 : 0.5}" />`;
        dualSvg += `<text x="${(x + bW2 / 2).toFixed(1)}" y="${(y - 10).toFixed(1)}" text-anchor="middle" font-family="'Space Mono',monospace" font-size="7.1" font-weight="${isFinal ? 700 : 600}" fill="${isFinal ? '#14532D' : '#1E2022'}">${row.ler_top}</text>`;
        dualSvg += `<text x="${(x + bW2 / 2).toFixed(1)}" y="${(y - 2).toFixed(1)}" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="6.8" fill="#5F6368">${row.ler_str}</text>`;
      });
      dualSvg += `<text x="${gCenter.toFixed(1)}" y="${yBase + 14}" text-anchor="middle" font-family="'Space Mono',monospace" font-size="8.5" font-weight="700" fill="#1E2022">${c.label}</text>`;
      dualSvg += `<text x="${gCenter.toFixed(1)}" y="${yBase + 25}" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="7.6" fill="#64748B">${c.sub}</text>`;
    });

    dualSvg += `</svg>`;

    container.innerHTML = `
      <div class="ca-root">
        <!-- Top Card: Parallel Flooding vs. Layered Serial Schedule + Soft-Power Edge Reweighting -->
        <div class="ca-card" style="flex:0.95;">
          <div class="ca-card-header">
            <span class="ca-card-title" style="color:var(--qai-green);">
              1. Breaking 4-Cycle Symmetry: Layered Serial Check Schedule &amp; Soft-Power Edge Reweighting
            </span>
            <span class="ca-badge" style="background:var(--qai-badge-green); color:var(--qai-green);">
              Affects Both Convergence (+69.7 pp on Mitten) &amp; Logical Accuracy (Up to 14.5&times; Lower LER)
            </span>
          </div>
          <div class="ca-svg-wrap">
            <svg viewBox="0 0 860 156" style="max-height:160px;" xmlns="http://www.w3.org/2000/svg">
              <!-- Panel 1: B1 Parallel Flooding + Unweighted (Oscillates) -->
              <rect x="8" y="4" width="268" height="148" rx="8" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="1.4" />
              <text x="142" y="22" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.6" font-weight="700" fill="#D93025">
                B1. FLOODING + UNWEIGHTED
              </text>
              <text x="142" y="36" text-anchor="middle" font-family="'Space Mono',monospace" font-size="8.6" font-weight="700" fill="#991B1B">
                --no-bp-serial --bp-reweight=none
              </text>
              <line x1="92" y1="68" x2="142" y2="112" stroke="#D93025" stroke-width="2.4" />
              <line x1="192" y1="68" x2="142" y2="112" stroke="#D93025" stroke-width="2.4" />
              <circle cx="92" cy="56" r="14" fill="#FEE2E2" stroke="#D93025" stroke-width="2" />
              <text x="92" y="61" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="13" fill="#991B1B">e₁=1</text>
              <circle cx="192" cy="56" r="14" fill="#FEE2E2" stroke="#D93025" stroke-width="2" />
              <text x="192" y="61" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="13" fill="#991B1B">e₂=1</text>
              <rect x="128" y="104" width="28" height="24" fill="#D93025" stroke="#1E2022" stroke-width="1.6" />
              <text x="142" y="120" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#FFFFFF">c=1</text>
              <text x="142" y="143" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" font-weight="600" fill="#991B1B">
                Frozen iteration scores oscillate (0,0)&harr;(1,1)
              </text>

              <!-- Panel 2: B2 +Serial Check Schedule (Converges) -->
              <rect x="292" y="4" width="276" height="148" rx="8" fill="#F0FDF4" stroke="#86EFAC" stroke-width="1.5" />
              <text x="430" y="22" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.6" font-weight="700" fill="#14532D">
                B2. +SERIAL CHECK SCHEDULE
              </text>
              <text x="430" y="36" text-anchor="middle" font-family="'Space Mono',monospace" font-size="8.6" font-weight="700" fill="#1E8E3E">
                --bp-serial --bp-reweight=none
              </text>
              <line x1="378" y1="68" x2="430" y2="112" stroke="#1E8E3E" stroke-width="2.6" />
              <line x1="482" y1="68" x2="430" y2="112" stroke="#94A3B8" stroke-width="1.8" stroke-dasharray="4,3" />
              <circle cx="378" cy="56" r="14" fill="#1E8E3E" stroke="#1E2022" stroke-width="2" />
              <text x="378" y="61" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="13" fill="#FFFFFF">e₁=1</text>
              <circle cx="482" cy="56" r="14" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
              <text x="482" y="61" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="13" fill="#1E2022">e₂=0</text>
              <rect x="416" y="104" width="28" height="24" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="2" />
              <text x="430" y="120" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#14532D">c&check;</text>
              <text x="430" y="143" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" font-weight="600" fill="#14532D">
                +68.3 pp convergence on Mitten d=10 (22.2% &rarr; 90.5%)
              </text>

              <!-- Panel 3: B3 +Soft-Power Edge Reweighting -->
              <rect x="584" y="4" width="268" height="148" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="718" y="22" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.6" font-weight="700" fill="#1967D2">
                B3. +SOFT-POWER EDGE REWEIGHTING
              </text>
              <text x="718" y="36" text-anchor="middle" font-family="'Space Mono',monospace" font-size="8.6" font-weight="700" fill="#1967D2">
                --bp-serial --bp-reweight=soft (Default)
              </text>
              <rect x="600" y="44" width="236" height="40" rx="5" fill="#E8F0FE" stroke="#1967D2" stroke-width="1.3" />
              <text x="718" y="60" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#1967D2">
                Pass 0 Active Fault (e&#770;₀ = 1)
              </text>
              <text x="718" y="75" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.6" fill="#1E2022">
                Propagates X-Z Y-correlations to Pass 1
              </text>
              <rect x="600" y="92" width="236" height="48" rx="5" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="1.4" />
              <text x="718" y="109" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#14532D">
                75% Conditional + 25% Channel Prior
              </text>
              <text x="718" y="126" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.6" font-weight="600" fill="#14532D">
                3.0&times;&ndash;14.5&times; lower LER across all 4 families!
              </text>
            </svg>
          </div>
        </div>

        <!-- Bottom Card: Dual Multi-Circuit Chart (Convergence % & LER Reduction) -->
        <div class="ca-card" style="flex:1.05;">
          <div class="ca-card-header">
            <span class="ca-card-title" style="color:var(--qai-deep-blue);">
              2. Multi-Circuit BP Algorithmic Ablation Across 4 Quantum Code Families
            </span>
            <span class="ca-badge" style="background:var(--qai-badge-blue); color:var(--qai-deep-blue);">
              Dual Effect: Higher Pure-BP Convergence + 3.0&times;&ndash;14.5&times; Lower End-to-End LER
            </span>
          </div>
          <div class="ca-svg-wrap">
            ${dualSvg}
          </div>
          ${buildLegendHtml(d.steps)}
        </div>
      </div>
    `;
    renderMath(container);
  }

  /* ==========================================================================
     3. OSD COMPUTATIONAL OPTIMIZATIONS ("osd_comp" / "ablation_osd_comp")
     ========================================================================== */
  function renderOsdComp(container) {
    const d = ABLATION_DATA.osd_comp;
    const allSpeedups = Object.values(d.byCircuit).flat().map(r => r.speedup);
    const maxSp = Math.max(...allSpeedups, 25.0);

    const chartSvg = buildSingleGroupedChartSvg({
      steps: d.steps,
      byCircuit: d.byCircuit,
      logScale: true,
      minVal: 0.7,
      yLabel: "OSD Speedup (×, log)",
      maxVal: maxSp,
      ticks: [1, 2, 5, 10, 20],
      getValue: r => r.speedup,
      getTopLabel: r => `${r.speedup.toFixed(2)}×`,
      getSubLabel: r => fmtMs(r.ms)
    });

    container.innerHTML = `
      <div class="ca-root">
        <!-- Top Card: Matrix Diagram of Zero-Remainder Early Exit + Horizon-Bounded Partial Sort -->
        <div class="ca-card" style="flex:0.95;">
          <div class="ca-card-header">
            <span class="ca-card-title" style="color:var(--qai-deep-blue);">
              1. Exact Gauss&ndash;Jordan RREF Acceleration: Dynamic Zero-Remainder Early Exit + Horizon-Bounded Partial Sort
            </span>
            <span class="ca-badge" style="background:var(--qai-badge-green); color:var(--qai-green);">
              0.0% LER Change Across All 4 Circuits (Bit-for-Bit Identical Codewords)
            </span>
          </div>
          <div class="ca-svg-wrap">
            <svg viewBox="0 0 860 156" style="max-height:160px;" xmlns="http://www.w3.org/2000/svg">
              <!-- Left Visual: Matrix Truncation & Early Exit -->
              <rect x="8" y="4" width="440" height="148" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="228" y="22" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.8" font-weight="700" fill="#1967D2">
                ORDERED PARITY MATRIX H (M DETECTOR ROWS &times; N FAULT COLS)
              </text>

              <!-- Full Column Bar -->
              <rect x="22" y="30" width="412" height="22" rx="4" fill="#F1F5F9" stroke="#94A3B8" stroke-width="1.2" />
              <rect x="22" y="30" width="166" height="22" rx="4" fill="#DBEAFE" stroke="#1967D2" stroke-width="1.5" />
              <text x="105" y="45" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9" font-weight="700" fill="#1967D2">
                Horizon K = 1.2M Cols
              </text>
              <text x="310" y="45" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.3" font-weight="600" fill="#64748B">
                Remaining N &minus; 1.2M columns skipped completely!
              </text>

              <!-- Truncated Submatrix -->
              <rect x="22" y="58" width="166" height="84" rx="5" fill="#FFFFFF" stroke="#1967D2" stroke-width="1.5" />
              <rect x="22" y="58" width="76" height="84" rx="4" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="2" />
              <line x1="28" y1="64" x2="90" y2="136" stroke="#1E8E3E" stroke-width="2.4" />
              <text x="60" y="94" text-anchor="middle" font-family="'Space Mono',monospace" font-size="8.8" font-weight="700" fill="#14532D">
                Pivots Span
              </text>
              <text x="60" y="108" text-anchor="middle" font-family="'Space Mono',monospace" font-size="8.8" font-weight="700" fill="#14532D">
                s_res!
              </text>

              <!-- Early Exit Cutoff Line -->
              <line x1="98" y1="54" x2="98" y2="146" stroke="#D93025" stroke-width="2.4" stroke-dasharray="4,3" />
              <text x="143" y="96" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.3" font-weight="700" fill="#D93025">
                HALT RREF!
              </text>
              <text x="143" y="111" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="8.8" fill="#64748B">
                s_res = 0 reached
              </text>

              <!-- Explanation Callout inside Left Box -->
              <rect x="200" y="58" width="234" height="84" rx="5" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.2" />
              <text x="210" y="76" font-family="'Space Mono',monospace" font-size="9.2" font-weight="700" fill="#14532D">
                Why Early Exit Is Exact:
              </text>
              <text x="210" y="92" font-family="'DM Sans',sans-serif" font-size="9.6" fill="#334155">
                OSD-0 only needs pivots that span the residual
              </text>
              <text x="210" y="107" font-family="'DM Sans',sans-serif" font-size="9.6" fill="#334155">
                syndrome s_res = s &oplus; H e_hard. Once s_res is
              </text>
              <text x="210" y="122" font-family="'DM Sans',sans-serif" font-size="9.6" fill="#334155">
                reduced to 0, later pivots get 0 coefficient!
              </text>

              <!-- Right Box: 3 Progression C Steps with CLI Flags -->
              <rect x="462" y="4" width="390" height="148" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="657" y="22" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.8" font-weight="700" fill="#1E8E3E">
                PROGRESSION C: OSD COMPUTATIONAL STEPS
              </text>

              <rect x="476" y="30" width="362" height="35" rx="5" fill="#F1F5F9" stroke="#64748B" stroke-width="1.2" />
              <text x="486" y="45" font-family="'Space Mono',monospace" font-size="9.1" font-weight="700" fill="#334155">
                C1. Full RREF + Full Sort (1.00&times; Baseline):
              </text>
              <text x="486" y="58" font-family="'DM Sans',sans-serif" font-size="9.4" fill="#1E2022">
                Full O(N log N) sort + all N columns (16.5&ndash;1,314 ms/OSD-shot)
              </text>

              <rect x="476" y="71" width="362" height="35" rx="5" fill="#E8F0FE" stroke="#1967D2" stroke-width="1.2" />
              <text x="486" y="86" font-family="'Space Mono',monospace" font-size="9.1" font-weight="700" fill="#1967D2">
                C2. +Dynamic Zero-Remainder Early Exit (--osd-early-exit):
              </text>
              <text x="486" y="99" font-family="'DM Sans',sans-serif" font-size="9.4" fill="#1E2022">
                7.77&times; speedup on Color d=11 &amp; 16.01&times; on Hyperbolic iter=40 (0% LER &Delta;)
              </text>

              <rect x="476" y="112" width="362" height="34" rx="5" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="1.4" />
              <text x="486" y="127" font-family="'Space Mono',monospace" font-size="8.9" font-weight="700" fill="#14532D">
                C3. +Horizon-Bounded Partial Sort (K=1.2M, nth_element):
              </text>
              <text x="486" y="140" font-family="'DM Sans',sans-serif" font-size="9.4" font-weight="600" fill="#14532D">
                --osd-col-truncation --osd-partial-sort &rArr; up to 22.35&times; OSD speedup
              </text>
            </svg>
          </div>
        </div>

        <!-- Bottom Card: 1 Plot Only (Throughput-Only, Log-Scale Isolated OSD Speedup & ms/OSD-shot) -->
        <div class="ca-card" style="flex:1.05;">
          <div class="ca-card-header">
            <span class="ca-card-title" style="color:var(--qai-green);">
              2. Multi-Circuit Isolated OSD Speedup Across 4 Code Families (Log Scale, with ms/OSD-shot)
            </span>
            <span class="ca-badge" style="background:var(--qai-badge-green); color:var(--qai-green);">
              1 Plot Only (Throughput) &nbsp;&middot;&nbsp; 0.0% LER Change Across All Steps
            </span>
          </div>
          <div class="ca-svg-wrap">
            ${chartSvg}
          </div>
          ${buildLegendHtml(d.steps)}
        </div>
      </div>
    `;
    renderMath(container);
  }

  /* ==========================================================================
     4. OSD ALGORITHMIC OPTIMIZATIONS ("osd_algo" / "ablation_osd_algo")
     ========================================================================== */
  function renderOsdAlgo(container) {
    const d = ABLATION_DATA.osd_algo;
    const allGains = Object.values(d.byCircuit).flat().map(r => r.ler_gain);
    const maxGain = Math.max(...allGains, 1400.0);

    const chartSvg = buildSingleGroupedChartSvg({
      steps: d.steps,
      byCircuit: d.byCircuit,
      logScale: true,
      minVal: 0.7,
      yLabel: "LER Reduction vs D0 (×, log)",
      maxVal: maxGain,
      ticks: [1, 10, 100, 1000],
      getValue: r => r.ler_gain,
      getTopLabel: r => r.ler_gain >= 100 ? `${r.ler_gain.toFixed(0)}×` : (r.ler_gain >= 10 ? `${r.ler_gain.toFixed(1)}×` : `${r.ler_gain.toFixed(2)}×`),
      getSubLabel: r => `${r.ler_str} [${fmtMs(r.ms)}]`
    });

    container.innerHTML = `
      <div class="ca-root">
        <!-- Top Card: Trajectory Seed Diversification + Higher-Order OSD & Signed Alpha-ISD -->
        <div class="ca-card" style="flex:0.95;">
          <div class="ca-card-header">
            <span class="ca-card-title" style="color:var(--qai-green);">
              1. Trajectory Seed Diversification (L^(t*) + 2 Relay Legs) &amp; Higher-Order OSD(O10/W2) + &alpha;-ISD
            </span>
            <span class="ca-badge" style="background:var(--qai-badge-blue); color:var(--qai-deep-blue);">
              Up to 1,359&times; Lower LER vs. Pure BP D0 (Up to 10.01&times; vs. Naive OSD-0 D1)
            </span>
          </div>
          <div class="ca-svg-wrap">
            <svg viewBox="0 0 860 156" style="max-height:160px;" xmlns="http://www.w3.org/2000/svg">
              <!-- Panel 1: Progression D Algorithmic Steps -->
              <rect x="8" y="4" width="395" height="148" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="205" y="22" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.6" font-weight="700" fill="#1967D2">
                A. PROGRESSION D: OSD ALGORITHMIC STEPS
              </text>

              <rect x="22" y="30" width="367" height="35" rx="5" fill="#E8F0FE" stroke="#1967D2" stroke-width="1.3" />
              <text x="32" y="45" font-family="'Space Mono',monospace" font-size="9.0" font-weight="700" fill="#1967D2">
                D0 &rarr; D1: +Naive OSD-0 (Final Iteration L^(T)):
              </text>
              <text x="32" y="58" font-family="'DM Sans',sans-serif" font-size="9.4" fill="#1E2022">
                407.6&times; LER reduction on Color d=11 &amp; 50.4&times; on Hyperbolic iter=40
              </text>

              <rect x="22" y="71" width="367" height="35" rx="5" fill="#E0F2F1" stroke="#00796B" stroke-width="1.3" />
              <text x="32" y="86" font-family="'Space Mono',monospace" font-size="8.9" font-weight="700" fill="#00796B">
                D2: +Trajectory Seed Diversification (L^(t*) + 2 Relay Legs):
              </text>
              <text x="32" y="99" font-family="'DM Sans',sans-serif" font-size="9.4" fill="#1E2022">
                5.72&times; step LER reduction on Mitten d=10 (40 &rarr; 7 errs) &amp; 1.43&times; on CC d=11
              </text>

              <rect x="22" y="112" width="367" height="34" rx="5" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="1.3" />
              <text x="32" y="127" font-family="'Space Mono',monospace" font-size="9.0" font-weight="700" fill="#14532D">
                D3: +Higher-Order OSD(O10/W2) &amp; &alpha;-ISD (Default):
              </text>
              <text x="32" y="140" font-family="'DM Sans',sans-serif" font-size="9.4" fill="#1E2022">
                2.33&times; step LER reduction on Color d=11 &amp; 1.96&times; on Hyperbolic iter=40
              </text>

              <!-- Arrow A -> B -->
              <line x1="406" y1="78" x2="424" y2="78" stroke="#1967D2" stroke-width="2.2" />
              <polygon points="427,78 419,73 419,83" fill="#1967D2" />

              <!-- Panel 2: Why Signed Alpha-Blending Rescues False Positives -->
              <rect x="430" y="4" width="422" height="148" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="641" y="22" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.8" font-weight="700" fill="#14532D">
                B. HOW SIGNED &alpha;-BLENDING UN-PINS TRAPPED FALSE POSITIVES
              </text>

              <!-- Problem Box -->
              <rect x="444" y="30" width="394" height="54" rx="5" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="1.3" />
              <text x="456" y="47" font-family="'Space Mono',monospace" font-size="9.2" font-weight="700" fill="#D93025">
                Standard |L_v| Trap (Saturated False Positive):
              </text>
              <text x="456" y="63" font-family="'DM Sans',sans-serif" font-size="9.6" fill="#1E2022">
                Trapped false fault has L_v^(t*) = &minus;10 (wrong sign, large |L_v| = 10)
              </text>
              <text x="456" y="77" font-family="'DM Sans',sans-serif" font-size="9.4" font-weight="600" fill="#991B1B">
                &rArr; Standard sorting treats |L_v|=10 as "reliable" and locks e_v = 1 in B&#772;!
              </text>

              <!-- Solution Box -->
              <rect x="444" y="92" width="394" height="54" rx="5" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="1.5" />
              <text x="456" y="109" font-family="'Space Mono',monospace" font-size="9.2" font-weight="700" fill="#14532D">
                Signed &alpha;-Blend Fix (&Lambda;&#771;_v = (1&minus;&alpha;)L_v^(t*) + &alpha;&Lambda;_v^(0)):
              </text>
              <text x="456" y="125" font-family="'DM Sans',sans-serif" font-size="9.6" fill="#1E2022">
                Negative L_v^(t*) = &minus;10 cancels positive prior &Lambda;_v^(0) = +18 &rArr; &Lambda;&#771;_v &approx; 0!
              </text>
              <text x="456" y="139" font-family="'DM Sans',sans-serif" font-size="9.4" font-weight="700" fill="#14532D">
                &rArr; Pulls trapped variable into pivot basis B where RREF fixes it!
              </text>
            </svg>
          </div>
        </div>

        <!-- Bottom Card: Multi-Circuit OSD Algorithmic LER Reduction (D0..D3) & Total ms/shot -->
        <div class="ca-card" style="flex:1.05;">
          <div class="ca-card-header">
            <span class="ca-card-title" style="color:var(--qai-deep-blue);">
              2. Multi-Circuit OSD Algorithmic LER Reduction vs. D0 Pure BP Across 4 Code Families (D0..D3, Log Scale)
            </span>
            <span class="ca-badge" style="background:var(--qai-badge-green); color:var(--qai-green);">
              D1 &rarr; D3 Cuts LER by Additional 3.33&times; (CC d=11), 10.01&times; (Mitten d=10), &amp; 2.80&times; (iter=40)
            </span>
          </div>
          <div class="ca-svg-wrap">
            ${chartSvg}
          </div>
          ${buildLegendHtml(d.steps)}
        </div>
      </div>
    `;
    renderMath(container);
  }

  function mountBpComp(container) {
    ensureSharedStyles();
    renderBpComp(container);
    return { cleanup() {} };
  }

  function mountBpAlgo(container) {
    ensureSharedStyles();
    renderBpAlgo(container);
    return { cleanup() {} };
  }

  function mountOsdComp(container) {
    ensureSharedStyles();
    renderOsdComp(container);
    return { cleanup() {} };
  }

  function mountOsdAlgo(container) {
    ensureSharedStyles();
    renderOsdAlgo(container);
    return { cleanup() {} };
  }

  function mountBpOsdIsdPipeline(container, props) {
    props = props || {};
    const raw = String(props.preset || props.mode || props.view || props.plotType || props.id || "").toLowerCase();
    if (raw.includes("osd_algo")) return mountOsdAlgo(container, props);
    if (raw.includes("osd")) return mountOsdComp(container, props);
    if (raw.includes("bp_algo")) return mountBpAlgo(container, props);
    return mountBpComp(container, props);
  }

  window.TesseractViz = window.TesseractViz || {};
  window.TesseractViz["bp_comp"] = mountBpComp;
  window.TesseractViz["bp_algo"] = mountBpAlgo;
  window.TesseractViz["osd_comp"] = mountOsdComp;
  window.TesseractViz["osd_algo"] = mountOsdAlgo;
  window.TesseractViz["ablation_bp_comp"] = mountBpComp;
  window.TesseractViz["ablation_bp_algo"] = mountBpAlgo;
  window.TesseractViz["ablation_osd_comp"] = mountOsdComp;
  window.TesseractViz["ablation_osd_algo"] = mountOsdAlgo;
  window.TesseractViz["bp_comp_algo"] = mountBpComp;
  window.TesseractViz["osd_comp_algo"] = mountOsdComp;
  window.TesseractViz["bp_osd_isd_pipeline"] = mountBpOsdIsdPipeline;

  function registerAll() {
    if (window.SlideEngine && typeof window.SlideEngine.registerAnimation === "function") {
      window.SlideEngine.registerAnimation("bp_comp", mountBpComp);
      window.SlideEngine.registerAnimation("bp_algo", mountBpAlgo);
      window.SlideEngine.registerAnimation("osd_comp", mountOsdComp);
      window.SlideEngine.registerAnimation("osd_algo", mountOsdAlgo);
      window.SlideEngine.registerAnimation("ablation_bp_comp", mountBpComp);
      window.SlideEngine.registerAnimation("ablation_bp_algo", mountBpAlgo);
      window.SlideEngine.registerAnimation("ablation_osd_comp", mountOsdComp);
      window.SlideEngine.registerAnimation("ablation_osd_algo", mountOsdAlgo);
      window.SlideEngine.registerAnimation("bp_comp_algo", mountBpComp);
      window.SlideEngine.registerAnimation("osd_comp_algo", mountOsdComp);
      window.SlideEngine.registerAnimation("bp_osd_isd_pipeline", mountBpOsdIsdPipeline);
    }
  }
  registerAll();
  document.addEventListener("DOMContentLoaded", registerAll);
})();
