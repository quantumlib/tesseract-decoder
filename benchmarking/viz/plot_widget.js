/* ==========================================================================
   Tesseract-BP-OSD Interactive Plot & Telemetry Widget
   - Works standalone (in benchmarking/viz/explorer.html) AND embedded inside
     SlideEngine (`SlideEngine.registerAnimation("viz_plot", mountVizPlot)`).
   - Supports:
     1. Speed-Accuracy Tradeoff Explorer ([B] -> [G] -> [M] -> [O])
     2. Hyperbolic Octagon Cat-Flagged Explorer (AQ Baseline vs [O])
     3. Stage Shot-Flow & CPU Compute Breakdown Explorer (Stage-1 BP / Stage-1 OSD)
   - Interactive controls: toggle decoders, toggle distances/codes, switch X-axis
     (Log vs Linear), switch Y-axis (Per-Round Block LER vs Per-Logical LER),
     scroll-wheel zoom & drag pan, rich hover tooltip, and `</> Provenance` drawer.
   ========================================================================== */

(function () {
  "use strict";

  const DEC_ORDER = ["B", "G", "M", "O", "R", "T"];
  const HYP_DEC_ORDER = ["AQ", "O"];

  const SERIES_STYLE_7 = [
    { id: "B", label: "[B] Baseline Tesseract", short: "[B] Base", color: "#202124", shape: "h", dash: [6, 3], lw: 2.0 },
    { id: "G", label: "[G] Gari", short: "[G] Gari", color: "#1E8E3E", shape: "^", dash: [4, 3], lw: 2.0 },
    { id: "M", label: "[M] Multi-Pass Tesseract", short: "[M] MP-Tess", color: "#1967D2", shape: "D", dash: [7, 3, 2, 3], lw: 2.1 },
    { id: "R", label: "[R] Relay-BP", short: "[R] Relay-BP", color: "#E66100", shape: "v", dash: [3, 3], lw: 2.0 },
    { id: "T", label: "[T] Telescoping-BP", short: "[T] Telescoping-BP", color: "#7B3F00", shape: "p", dash: [5, 2], lw: 2.0 },
    { id: "O", label: "[O] Tesseract-BP-OSD", short: "[O] BP-OSD", color: "#008080", shape: "s", dash: [], lw: 2.6 }
  ];

  function hexToRgba(hex, alpha) {
    const clean = String(hex || "#1967D2").replace("#", "");
    const num = parseInt(clean.length === 3 ? clean.replace(/(.)/g, "$1$1") : clean, 16);
    const r = (num >> 16) & 255;
    const g = (num >> 8) & 255;
    const b = num & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  const STAGE_COLORS = {
    stage1_bp: { color: "#1E8E3E", label: "Pass 0+1: BP converged" },
    stage1_osd: { color: "#1967D2", label: "OSD post-processing" }
  };

  function formatSci(val, digits = 2) {
    if (val === null || val === undefined || isNaN(val)) return "—";
    if (val === 0) return "0";
    if (val >= 0.01 && val < 1000) return Number(val).toPrecision(digits + 1);
    const exp = Math.floor(Math.log10(Math.abs(val)));
    const mant = val / Math.pow(10, exp);
    return `${mant.toFixed(digits)}e${exp >= 0 ? "+" : ""}${exp}`;
  }

  function formatTime(sec) {
    if (sec === null || sec === undefined || isNaN(sec)) return "—";
    if (sec < 1e-3) return `${(sec * 1e6).toFixed(1)} µs`;
    if (sec < 1.0) return `${(sec * 1e3).toFixed(2)} ms`;
    return `${sec.toFixed(2)} s`;
  }

  function injectStylesOnce() {
    if (document.getElementById("tesseract-viz-widget-styles")) return;
    const style = document.createElement("style");
    style.id = "tesseract-viz-widget-styles";
    style.textContent = `
      .tviz-root {
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 100%;
        min-height: 0;
        background: #ffffff;
        border: none;
        border-radius: 0;
        overflow: hidden;
        font-family: "Google Sans", "DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #202124;
        position: relative;
        box-shadow: none;
      }
      .tviz-toolbar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
        padding: 6px 12px;
        background: #ffffff;
        border-bottom: 1px solid #ebecef;
        font-size: 0.72rem;
      }
      .tviz-toolbar-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 4px;
      }
      .tviz-group-label {
        font-family: "Google Sans", "Space Mono", sans-serif;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #5f6368;
        margin-right: 2px;
        letter-spacing: 0.04em;
      }
      .tviz-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #dadce0;
        background: #ffffff;
        color: #3c4043;
        font-size: 0.69rem;
        font-weight: 500;
        cursor: pointer;
        user-select: none;
        transition: all 0.12s ease;
        line-height: 1.2;
      }
      .tviz-pill:hover {
        border-color: #1967d2;
        color: #1967d2;
      }
      .tviz-pill.active {
        background: #1967d2;
        color: #ffffff;
        border-color: #1967d2;
      }
      .tviz-pill.dec-pill.active {
        background: #202124;
        color: #ffffff;
        border-color: #202124;
      }
      .tviz-pill.prov-btn {
        background: #f8f9fa;
        color: #3c4043;
        border-color: #dadce0;
        font-family: "Space Mono", monospace;
        font-size: 0.65rem;
      }
      .tviz-pill.prov-btn.active {
        background: #202124;
        color: #ffffff;
        border-color: #202124;
      }
      .tviz-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        border: 1.5px solid currentColor;
      }
      .tviz-canvas-wrap {
        flex: 1;
        position: relative;
        width: 100%;
        min-height: 0;
        background: #ffffff;
        overflow: hidden;
      }
      .tviz-canvas {
        width: 100%;
        height: 100%;
        display: block;
        cursor: crosshair;
      }
      .tviz-tooltip {
        position: absolute;
        pointer-events: none;
        background: rgba(15, 23, 42, 0.96);
        color: #f8fafc;
        padding: 9px 11px;
        border-radius: 7px;
        font-size: 0.72rem;
        line-height: 1.42;
        max-width: 340px;
        z-index: 30;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
        border: 1px solid rgba(148, 163, 184, 0.35);
        display: none;
      }
      .tviz-tooltip-title {
        font-weight: 700;
        font-size: 0.77rem;
        color: #38bdf8;
        margin-bottom: 4px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        padding-bottom: 3px;
      }
      .tviz-tooltip-grid {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 2px 8px;
      }
      .tviz-tooltip-key {
        color: #94a3b8;
        font-weight: 500;
      }
      .tviz-tooltip-val {
        color: #f8fafc;
        font-family: "Space Mono", monospace;
        font-size: 0.68rem;
        text-align: right;
      }
      .tviz-tooltip-cli {
        margin-top: 5px;
        padding-top: 4px;
        border-top: 1px solid rgba(148, 163, 184, 0.25);
        font-family: "Space Mono", monospace;
        font-size: 0.62rem;
        color: #a7f3d0;
        word-break: break-all;
      }
      .tviz-prov-drawer {
        position: absolute;
        inset: auto 10px 10px 10px;
        max-height: 62%;
        overflow-y: auto;
        background: rgba(15, 23, 42, 0.97);
        color: #e2e8f0;
        border-radius: 8px;
        padding: 12px 14px;
        z-index: 25;
        font-family: "Space Mono", monospace;
        font-size: 0.69rem;
        line-height: 1.48;
        border: 1px solid #334155;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
        display: none;
      }
      .tviz-prov-drawer.open {
        display: block;
      }
      .tviz-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 4px 12px;
        background: #ffffff;
        border-top: 1px solid #ebecef;
        font-size: 0.65rem;
        color: #5f6368;
        font-family: "Space Mono", monospace;
      }
    `;
    document.head.appendChild(style);
  }

  function resolveInitialState(props, dataset) {
    const p = props || {};
    const rawView = p.view || p.plotType || "tradeoff";
    let mode = "tradeoff";
    if (rawView === "cc_scaling" || p.preset === "cc_scaling") {
      mode = "cc_scaling";
    } else if (rawView === "bb_psweep" || p.preset === "bb_psweep") {
      mode = "bb_psweep";
    } else if (rawView === "hyperbolic" || rawView === "hyperbolic_ler" || p.preset === "hyperbolic") {
      mode = "hyperbolic";
    } else if (rawView === "stage_breakdown") {
      mode = "stage_breakdown";
    }

    const allCcTags = [
      "cc_d3",
      "cc_d5",
      "cc_d7",
      "cc_d9",
      "cc_d11",
      "cc_d13",
      "cc_d15",
      "cc_d17",
      "cc_d19",
      "cc_d21",
      "cc_d23"
    ].filter((c) => dataset.circuits[c]);

    const bbCode =
      p.code === "bb_d12_gross" || p.code === "gross_d12" || p.preset === "bb_gross"
        ? "bb_d12_gross"
        : "bb_d6";

    let activeCodes = [];
    if (mode === "cc_scaling") {
      activeCodes = allCcTags.slice();
    } else if (mode === "bb_psweep") {
      activeCodes = [bbCode];
    } else if (mode === "tradeoff") {
      if (p.preset === "bivariate_bicycle" || p.code === "bb") {
        activeCodes = Array.isArray(p.codes) ? p.codes.slice() : ["bb_d6", "bb_d12_gross"];
      } else if (p.preset === "mitten" || p.code === "mitten_d10" || p.code === "mitten") {
        activeCodes = ["mitten_d10"];
      } else if (Array.isArray(p.codes) && p.codes.length > 0) {
        activeCodes = p.codes.filter((c) => dataset.circuits[c]);
      } else if (Array.isArray(p.distances)) {
        activeCodes = p.distances.map((d) => `cc_d${d}`).filter((c) => dataset.circuits[c]);
      } else {
        activeCodes = ["cc_d5", "cc_d7", "cc_d9", "cc_d11"];
      }
    } else if (mode === "hyperbolic") {
      activeCodes = ["hyp_iter32", "hyp_iter40", "hyp_iter58", "hyp_iter109"];
    } else if (mode === "stage_breakdown") {
      if (p.preset === "all_families" || p.code === "all_families") {
        activeCodes = [
          "cc_d5",
          "cc_d7",
          "cc_d9",
          "cc_d11",
          "bb_d6",
          "bb_d12_gross",
          "mitten_d10",
          "hyp_iter32",
          "hyp_iter40",
          "hyp_iter58",
          "hyp_iter109"
        ];
      } else if (Array.isArray(p.distances)) {
        activeCodes = p.distances.map((d) => `cc_d${d}`).filter((c) => dataset.circuits[c]);
      } else {
        const d5to23 = allCcTags.filter((c) => c !== "cc_d3");
        activeCodes = d5to23.length > 0 ? d5to23 : ["cc_d5", "cc_d7", "cc_d9", "cc_d11"];
      }
    }

    const activeDecoders =
      mode === "hyperbolic"
        ? ["AQ", "O"]
        : Array.isArray(p.decoders)
        ? Array.from(new Set(p.decoders.filter((d) => DEC_ORDER.includes(d))))
        : mode === "cc_scaling" || mode === "bb_psweep"
        ? ["B", "G", "M", "O", "R", "T"]
        : ["B", "G", "M", "O"];

    const lockSinglePlot = p.showPlotSwitcher !== true;
    let plotTitle = "Speed–Accuracy Tradeoff · Superdense Color Codes (d = 5, 7, 9, 11 · up to 150k shots)";
    if (mode === "cc_scaling") {
      plotTitle = "Superdense Color Codes (d = 3 .. 23 · up to 150k shots) · LER & Runtime vs. Distance";
    } else if (mode === "bb_psweep") {
      plotTitle =
        bbCode === "bb_d12_gross"
          ? "Gross Code [[144, 12, 12]] (d = 12 · up to 50k shots/pt) · LER & Runtime vs. Physical Noise p"
          : "Bivariate Bicycle [[72, 12, 6]] (d = 6 · up to 100k shots/pt) · LER & Runtime vs. Physical Noise p";
    } else if (mode === "stage_breakdown") {
      plotTitle =
        p.preset === "all_families" || p.code === "all_families"
          ? "Stage Shot-Flow & CPU Time Breakdown · All Code Families (up to 150k shots)"
          : "Stage Shot-Flow & CPU Time Breakdown · Color Codes (d = 5 .. 23 · up to 150k shots)";
    } else if (mode === "hyperbolic") {
      plotTitle = "Cat-Flagged Hyperbolic Octagon Codes vs. AlphaQubit (102k shots iter=32..58, 24k iter=109, 1M AQ)";
    } else if (p.preset === "bivariate_bicycle" || p.code === "bb") {
      plotTitle = "Speed–Accuracy Tradeoff · Bivariate Bicycle & Gross Codes (50k–100k shots)";
    } else if (p.preset === "mitten" || p.code === "mitten_d10" || p.code === "mitten") {
      plotTitle = "Speed–Accuracy Tradeoff · High-Rate Mitten Code [[150, 30, 10]] (100k shots)";
    }

    return {
      mode,
      bbCode,
      lockSinglePlot,
      plotTitle,
      initialCodes: activeCodes.slice(),
      activeCodes,
      activeDecoders,
      xScale: p.xScale || "log",
      yNorm: p.norm === "per_logical" ? "per_logical" : "block",
      hypSubView: p.hypSubView || "dual",
      zoom: 1.0,
      panX: 0,
      panY: 0,
      showProvenance: false,
      hoveredItem: null
    };
  }

  function mountVizPlot(container, props) {
    if (!container) return;
    const pView = String((props && (props.view || props.plotType || props.preset)) || "").toLowerCase();
    if (
      (pView.includes("ablation_") || pView === "bp_comp" || pView === "bp_algo" || pView === "osd_comp" || pView === "osd_algo") &&
      window.TesseractViz &&
      typeof window.TesseractViz["bp_osd_isd_pipeline"] === "function"
    ) {
      return window.TesseractViz["bp_osd_isd_pipeline"](container, props);
    }
    injectStylesOnce();

    const dataset = window.TESSERACT_VIZ_DATASETS;
    if (!dataset || !Array.isArray(dataset.records)) {
      container.innerHTML = `<div style="padding:18px;color:#b91c1c;font-family:monospace;">Missing window.TESSERACT_VIZ_DATASETS (load datasets.js before plot_widget.js).</div>`;
      return;
    }

    const byKey = {};
    dataset.records.forEach((r) => {
      if (DEC_ORDER.includes(r.decoder_id) || HYP_DEC_ORDER.includes(r.decoder_id)) {
        byKey[`${r.tag}__${r.decoder_id}`] = r;
      }
    });

    const state = resolveInitialState(props, dataset);

    container.innerHTML = "";
    const root = document.createElement("div");
    root.className = "tviz-root";

    const toolbar = document.createElement("div");
    toolbar.className = "tviz-toolbar";
    root.appendChild(toolbar);

    const canvasWrap = document.createElement("div");
    canvasWrap.className = "tviz-canvas-wrap";
    const canvas = document.createElement("canvas");
    canvas.className = "tviz-canvas";
    canvasWrap.appendChild(canvas);

    const tooltip = document.createElement("div");
    tooltip.className = "tviz-tooltip";
    canvasWrap.appendChild(tooltip);

    const provDrawer = document.createElement("div");
    provDrawer.className = "tviz-prov-drawer";
    canvasWrap.appendChild(provDrawer);

    root.appendChild(canvasWrap);

    const footer = document.createElement("div");
    footer.className = "tviz-footer";
    footer.innerHTML = `
      <span>Hover any point/bar for exact shots, 1σ CI, stage breakdown &amp; CLI · Scroll to zoom, drag to pan</span>
      <span>commit <strong>${dataset.git_commit || "HEAD"}</strong></span>
    `;
    root.appendChild(footer);
    container.appendChild(root);

    let hitTargets = [];

    function renderToolbar() {
      toolbar.innerHTML = "";

      // Row 1: Single-Plot Header Badge (or Mode switcher if unlocked) + Axis toggles + Provenance
      const row1 = document.createElement("div");
      row1.className = "tviz-toolbar-row";

      if (state.lockSinglePlot) {
        const titleBadge = document.createElement("span");
        titleBadge.style.cssText =
          "font-family:'Space Mono',monospace; font-size:0.70rem; font-weight:700; padding:3px 9px; border-radius:5px; background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; display:inline-flex; align-items:center; gap:5px;";
        titleBadge.textContent = state.plotTitle;
        row1.appendChild(titleBadge);
      } else {
        const modeLbl = document.createElement("span");
        modeLbl.className = "tviz-group-label";
        modeLbl.textContent = "Plot:";
        row1.appendChild(modeLbl);

        [
          { id: "tradeoff", label: "Speed–Accuracy Tradeoff" },
          { id: "cc_scaling", label: "CC Scaling (d=3..23)" },
          { id: "bb_psweep", label: "BB p-Sweep" },
          { id: "hyperbolic", label: "Hyperbolic vs AQ" },
          { id: "stage_breakdown", label: "Stage Shot & CPU Breakdown" }
        ].forEach((m) => {
          const btn = document.createElement("button");
          btn.className = "tviz-pill" + (state.mode === m.id ? " active" : "");
          btn.textContent = m.label;
          btn.onclick = () => {
            state.mode = m.id;
            state.zoom = 1.0;
            state.panX = 0;
            state.panY = 0;
            if (m.id === "hyperbolic") {
              state.activeCodes = ["hyp_iter32", "hyp_iter40", "hyp_iter58", "hyp_iter109"];
              state.activeDecoders = ["AQ", "O"];
            } else if (m.id === "cc_scaling") {
              state.activeCodes = [
                "cc_d3",
                "cc_d5",
                "cc_d7",
                "cc_d9",
                "cc_d11",
                "cc_d13",
                "cc_d15",
                "cc_d17",
                "cc_d19",
                "cc_d21",
                "cc_d23"
              ];
              state.activeDecoders = ["B", "G", "M", "O", "R", "T"];
            } else if (m.id === "bb_psweep") {
              state.bbCode = state.bbCode || "bb_d6";
              state.activeCodes = [state.bbCode];
              state.activeDecoders = ["B", "G", "M", "O", "R", "T"];
            } else if (m.id === "stage_breakdown") {
              if (state.activeCodes.every((c) => c.startsWith("hyp_"))) {
                state.activeCodes = ["cc_d5", "cc_d7", "cc_d9", "cc_d11"];
              }
            } else if (m.id === "tradeoff") {
              if (state.activeCodes.every((c) => c.startsWith("hyp_")) || state.activeCodes.length > 5) {
                state.activeCodes = ["cc_d5", "cc_d7", "cc_d9", "cc_d11"];
              }
              state.activeDecoders = ["B", "G", "M", "O"];
            }
            renderToolbar();
            draw();
          };
          row1.appendChild(btn);
        });
      }

      const rightControls = document.createElement("div");
      rightControls.className = "tviz-toolbar-row";

      if (state.mode === "tradeoff") {
        const xBtn = document.createElement("button");
        xBtn.className = "tviz-pill";
        xBtn.textContent = state.xScale === "log" ? "X: Log" : "X: Linear";
        xBtn.title = "Toggle X-Axis between Log and Linear scale";
        xBtn.onclick = () => {
          state.xScale = state.xScale === "log" ? "linear" : "log";
          state.zoom = 1.0;
          state.panX = 0;
          renderToolbar();
          draw();
        };
        rightControls.appendChild(xBtn);
      }

      if (
        state.mode === "tradeoff" ||
        state.mode === "hyperbolic" ||
        state.mode === "cc_scaling" ||
        state.mode === "bb_psweep"
      ) {
        const yBtn = document.createElement("button");
        yBtn.className = "tviz-pill" + (state.yNorm === "per_logical" ? " active" : "");
        yBtn.textContent = state.yNorm === "per_logical" ? "Y: Per-Logical LER (÷k)" : "Y: Block LER/Rnd";
        yBtn.title = "Switch Y-axis between Per-Round Block LER and Per-Logical LER (normalized by k)";
        yBtn.onclick = () => {
          state.yNorm = state.yNorm === "per_logical" ? "block" : "per_logical";
          renderToolbar();
          draw();
        };
        rightControls.appendChild(yBtn);
      }

      if (state.zoom !== 1.0 || state.panX !== 0 || state.panY !== 0) {
        const rstBtn = document.createElement("button");
        rstBtn.className = "tviz-pill";
        rstBtn.textContent = "↺ Reset Zoom";
        rstBtn.onclick = () => {
          state.zoom = 1.0;
          state.panX = 0;
          state.panY = 0;
          renderToolbar();
          draw();
        };
        rightControls.appendChild(rstBtn);
      }

      const provBtn = document.createElement("button");
      provBtn.className = "tviz-pill prov-btn" + (state.showProvenance ? " active" : "");
      provBtn.textContent = "</> Provenance";
      provBtn.onclick = () => {
        state.showProvenance = !state.showProvenance;
        updateProvenanceDrawer();
        renderToolbar();
      };
      rightControls.appendChild(provBtn);

      toolbar.appendChild(row1);
      toolbar.appendChild(rightControls);

      // On stage_breakdown in single-plot mode, all bars and legend are on the canvas itself
      if (state.lockSinglePlot && state.mode === "stage_breakdown") {
        return;
      }

      // Row 2: Code Legend (for this slide's codes only) & Decoder Legend/Filter
      const row2 = document.createElement("div");
      row2.className = "tviz-toolbar-row";
      row2.style.width = "100%";
      row2.style.justifyContent = "space-between";
      row2.style.paddingTop = "3px";
      row2.style.borderTop = "1px dashed #e2e8f0";

      const codeGroup = document.createElement("div");
      codeGroup.className = "tviz-toolbar-row";

      const candidateCodes = state.lockSinglePlot
        ? state.mode === "cc_scaling"
          ? ["cc_d3", "cc_d7", "cc_d11", "cc_d15", "cc_d19", "cc_d23"]
          : state.initialCodes
        : state.mode === "bb_psweep"
        ? ["bb_d6", "bb_d12_gross"]
        : state.mode === "cc_scaling"
        ? ["cc_d3", "cc_d5", "cc_d7", "cc_d9", "cc_d11", "cc_d13", "cc_d15", "cc_d17", "cc_d19", "cc_d21", "cc_d23"]
        : state.mode === "hyperbolic"
        ? ["hyp_iter32", "hyp_iter40", "hyp_iter58", "hyp_iter109"]
        : state.mode === "stage_breakdown"
        ? [
            "cc_d5",
            "cc_d7",
            "cc_d9",
            "cc_d11",
            "cc_d13",
            "cc_d15",
            "cc_d17",
            "cc_d19",
            "cc_d21",
            "cc_d23",
            "bb_d6",
            "bb_d12_gross",
            "mitten_d10",
            "hyp_iter32",
            "hyp_iter40",
            "hyp_iter58",
            "hyp_iter109"
          ]
        : [
            "cc_d3",
            "cc_d5",
            "cc_d7",
            "cc_d9",
            "cc_d11",
            "cc_d13",
            "cc_d15",
            "cc_d17",
            "cc_d19",
            "cc_d21",
            "cc_d23",
            "bb_d6",
            "bb_d12_gross",
            "mitten_d10"
          ];

      const markerSymbol = (m) => {
        if (m === "^") return "▲";
        if (m === "v") return "▼";
        if (m === "<") return "◀";
        if (m === ">") return "▶";
        if (m === "s") return "■";
        if (m === "h") return "⬢";
        if (m === "D" || m === "d") return "◆";
        if (m === "*") return "★";
        if (m === "p") return "⬟";
        return "●";
      };

      if (state.mode !== "hyperbolic") {
        const codeLbl = document.createElement("span");
        codeLbl.className = "tviz-group-label";
        codeLbl.textContent =
          state.mode === "cc_scaling"
            ? "Distances:"
            : state.mode === "bb_psweep"
            ? "BB Code:"
            : state.lockSinglePlot && candidateCodes.length > 1
            ? "Distances:"
            : state.lockSinglePlot
            ? "Code:"
            : "Codes:";
        codeGroup.appendChild(codeLbl);

        if (state.mode === "cc_scaling" && state.lockSinglePlot) {
          const rangeBadge = document.createElement("span");
          rangeBadge.style.cssText =
            "font-family:'Space Mono',monospace; font-size:0.65rem; font-weight:600; padding:2px 8px; border-radius:4px; background:#fefce8; border:1px solid #fde047; color:#854d0e; display:inline-flex; align-items:center; gap:5px;";
          rangeBadge.textContent = "d = 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 (q = 13 .. 793)";
          codeGroup.appendChild(rangeBadge);
        } else {
          candidateCodes.forEach((cTag) => {
            const cm = dataset.circuits[cTag];
            if (!cm) return;
            const shortName = cTag
              .replace("cc_d", "CC d=")
              .replace("bb_d6", "BB [[72,12,6]] (d=6)")
              .replace("bb_d12_gross", "Gross [[144,12,12]] (d=12)")
              .replace("mitten_d10", "Mitten d=10")
              .replace("hyp_iter", "iter=");

            if (state.lockSinglePlot && candidateCodes.length <= 1) {
              const badge = document.createElement("span");
              badge.style.cssText =
                "font-family:'Space Mono',monospace; font-size:0.65rem; font-weight:600; padding:2px 7px; border-radius:4px; background:#ffffff; border:1px solid #dadce0; color:#202124; display:inline-flex; align-items:center; gap:4px;";
              const sym = document.createElement("span");
              sym.style.color = cm.fill_color === "#f5c242" ? "#d99b00" : cm.fill_color || "#1967d2";
              sym.textContent = markerSymbol(cm.marker);
              badge.appendChild(sym);
              const lbl = document.createElement("span");
              lbl.textContent = shortName;
              badge.appendChild(lbl);
              codeGroup.appendChild(badge);
            } else {
              const pill = document.createElement("button");
              const isActive =
                state.mode === "bb_psweep" ? state.bbCode === cTag : state.activeCodes.includes(cTag);
              pill.className = "tviz-pill" + (isActive ? " active" : "");
              const sym = document.createElement("span");
              sym.style.color = isActive
                ? "#ffffff"
                : cm.fill_color === "#f5c242"
                ? "#d99b00"
                : cm.fill_color || "#1967d2";
              sym.textContent = markerSymbol(cm.marker);
              pill.appendChild(sym);
              const lbl = document.createElement("span");
              lbl.textContent = shortName;
              pill.appendChild(lbl);
              pill.onclick = () => {
                if (state.mode === "bb_psweep") {
                  state.bbCode = cTag;
                  state.activeCodes = [cTag];
                } else if (isActive && state.activeCodes.length > 1) {
                  state.activeCodes = state.activeCodes.filter((x) => x !== cTag);
                } else if (!isActive) {
                  state.activeCodes.push(cTag);
                }
                renderToolbar();
                draw();
              };
              codeGroup.appendChild(pill);
            }
          });
        }
      }
      row2.appendChild(codeGroup);

      if (state.mode !== "stage_breakdown") {
        const decGroup = document.createElement("div");
        decGroup.className = "tviz-toolbar-row";
        const decLbl = document.createElement("span");
        decLbl.className = "tviz-group-label";
        decLbl.textContent = "Decoders:";
        decGroup.appendChild(decLbl);

        const decList = state.mode === "hyperbolic" ? HYP_DEC_ORDER : DEC_ORDER;
        decList.forEach((dId) => {
          const dm = dataset.decoders[dId];
          if (!dm) return;
          const isAct = state.activeDecoders.includes(dId);
          const pill = document.createElement("button");
          pill.className = "tviz-pill dec-pill" + (isAct ? " active" : "");
          const dot = document.createElement("span");
          dot.className = "tviz-dot";
          dot.style.borderColor = dm.ring_color;
          dot.style.background = isAct ? dm.ring_color : "transparent";
          pill.appendChild(dot);
          const txt = document.createElement("span");
          txt.textContent =
            dId === "B"
              ? "[B] Base"
              : dId === "G"
              ? "[G] Gari"
              : dId === "M"
              ? "[M] MP-Tess"
              : dId === "O"
              ? "[O] BP-OSD"
              : dId === "R"
              ? "[R] Relay"
              : dId === "T"
              ? "[T] Telescoping"
              : "AQ Baseline";
          pill.appendChild(txt);
          pill.onclick = () => {
            if (isAct && state.activeDecoders.length > 1) {
              state.activeDecoders = state.activeDecoders.filter((x) => x !== dId);
            } else if (!isAct) {
              state.activeDecoders.push(dId);
            }
            renderToolbar();
            draw();
          };
          decGroup.appendChild(pill);
        });
        row2.appendChild(decGroup);
      }

      toolbar.appendChild(row2);
    }

    function getPsweepRecordsForCode(codeTag) {
      const list = [];
      const seen = new Set();
      const rawPs = Array.isArray(dataset.psweep_records) ? dataset.psweep_records : [];
      rawPs.forEach((r) => {
        const bTag = r.base_tag || (String(r.tag || "").startsWith("bb_d12") || String(r.tag || "").includes("gross") ? "bb_d12_gross" : "bb_d6");
        if (bTag === codeTag) {
          const k = `${bTag}__${r.p}__${r.decoder_id}`;
          if (!seen.has(k)) {
            seen.add(k);
            list.push(r);
          }
        }
      });
      dataset.records.forEach((r) => {
        const bTag = r.base_tag || (String(r.tag || "").startsWith(codeTag) ? codeTag : r.tag);
        if (bTag === codeTag) {
          const pVal = r.p || 0.001;
          const k = `${bTag}__${pVal}__${r.decoder_id}`;
          if (!seen.has(k)) {
            seen.add(k);
            list.push(Object.assign({}, r, { p: pVal, base_tag: codeTag }));
          }
        }
      });
      return list;
    }

    function updateProvenanceDrawer() {
      if (!state.showProvenance) {
        provDrawer.classList.remove("open");
        return;
      }
      provDrawer.classList.add("open");
      const activeRecs = [];
      if (state.mode === "bb_psweep") {
        const codeTag = state.bbCode || state.activeCodes[0] || "bb_d6";
        getPsweepRecordsForCode(codeTag).forEach((r) => {
          if (state.activeDecoders.includes(r.decoder_id)) {
            activeRecs.push(r);
          }
        });
      } else {
        state.activeCodes.forEach((cTag) => {
          const decs = state.mode === "stage_breakdown" ? ["O"] : state.activeDecoders;
          decs.forEach((dId) => {
            const r = byKey[`${cTag}__${dId}`];
            if (r) activeRecs.push(r);
          });
        });
      }
      const sources = Array.from(new Set(activeRecs.map((r) => r.json_source)));
      const cmds = activeRecs.slice(0, 6).map((r) => `# ${r.tag} (${r.decoder_short}):\n${r.command_line}`);
      provDrawer.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <strong style="color:#4ade80;font-size:0.76rem;">&lt;/&gt; DATA &amp; FIGURE PROVENANCE (commit ${dataset.git_commit})</strong>
          <div style="display:flex;gap:6px;">
            <button id="tvizCopyCliBtn" style="background:#1e293b;color:#38bdf8;border:1px solid #38bdf8;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:0.65rem;">Copy CLI Commands</button>
            <button id="tvizCloseProvBtn" style="background:#334155;color:#ffffff;border:none;border-radius:4px;padding:2px 7px;cursor:pointer;font-size:0.65rem;">✕</button>
          </div>
        </div>
        <div style="color:#94a3b8;margin-bottom:4px;">Source JSON(s): <span style="color:#f8fafc;">${sources.join(", ") || "benchmarking/data/*.json"}</span></div>
        <div style="color:#94a3b8;margin-bottom:6px;">Static Exporter: <span style="color:#fde047;">python3 benchmarking/viz/export_static.py</span></div>
        <pre style="white-space:pre-wrap;word-break:break-all;background:#090d16;padding:8px;border-radius:5px;border:1px solid #1e293b;color:#a7f3d0;margin:0;">${cmds.join("\n\n")}</pre>
      `;
      const closeBtn = provDrawer.querySelector("#tvizCloseProvBtn");
      if (closeBtn) {
        closeBtn.onclick = () => {
          state.showProvenance = false;
          updateProvenanceDrawer();
          renderToolbar();
        };
      }
      const copyBtn = provDrawer.querySelector("#tvizCopyCliBtn");
      if (copyBtn) {
        copyBtn.onclick = () => {
          if (navigator.clipboard) {
            navigator.clipboard.writeText(cmds.join("\n\n"));
            copyBtn.textContent = "✓ Copied!";
            setTimeout(() => {
              copyBtn.textContent = "Copy CLI Commands";
            }, 1500);
          }
        };
      }
    }

    function drawMarkerShape(ctx, shape, x, y, r) {
      ctx.beginPath();
      if (shape === "^") {
        ctx.moveTo(x, y - r * 1.15);
        ctx.lineTo(x + r * 1.08, y + r * 0.9);
        ctx.lineTo(x - r * 1.08, y + r * 0.9);
        ctx.closePath();
      } else if (shape === "v") {
        ctx.moveTo(x, y + r * 1.15);
        ctx.lineTo(x + r * 1.08, y - r * 0.9);
        ctx.lineTo(x - r * 1.08, y - r * 0.9);
        ctx.closePath();
      } else if (shape === "<") {
        ctx.moveTo(x - r * 1.15, y);
        ctx.lineTo(x + r * 0.9, y - r * 1.08);
        ctx.lineTo(x + r * 0.9, y + r * 1.08);
        ctx.closePath();
      } else if (shape === ">") {
        ctx.moveTo(x + r * 1.15, y);
        ctx.lineTo(x - r * 0.9, y - r * 1.08);
        ctx.lineTo(x - r * 0.9, y + r * 1.08);
        ctx.closePath();
      } else if (shape === "s") {
        ctx.rect(x - r * 0.88, y - r * 0.88, r * 1.76, r * 1.76);
      } else if (shape === "D" || shape === "d") {
        const rx = shape === "d" ? r * 0.85 : r * 1.15;
        ctx.moveTo(x, y - r * 1.15);
        ctx.lineTo(x + rx, y);
        ctx.lineTo(x, y + r * 1.15);
        ctx.lineTo(x - rx, y);
        ctx.closePath();
      } else if (shape === "*") {
        const spikes = 5;
        const outer = r * 1.22;
        const inner = r * 0.52;
        for (let i = 0; i < spikes * 2; i++) {
          const rad = i % 2 === 0 ? outer : inner;
          const angle = (i * Math.PI) / spikes - Math.PI / 2;
          const px = x + Math.cos(angle) * rad;
          const py = y + Math.sin(angle) * rad;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
      } else if (shape === "p" || shape === "h") {
        const sides = shape === "h" ? 6 : 5;
        for (let i = 0; i < sides; i++) {
          const angle = (i * 2 * Math.PI) / sides - Math.PI / 2;
          const px = x + Math.cos(angle) * r * 1.1;
          const py = y + Math.sin(angle) * r * 1.1;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
      } else {
        ctx.arc(x, y, r, 0, Math.PI * 2);
      }
    }

    function drawArrowhead(ctx, x0, y0, x1, y1, color, lw) {
      const dx = x1 - x0;
      const dy = y1 - y0;
      const len = Math.hypot(dx, dy);
      if (len < 22) return;
      const ux = dx / len;
      const uy = dy / len;
      const tipX = x1 - ux * 11;
      const tipY = y1 - uy * 11;
      const startX = x0 + ux * 11;
      const startY = y0 + uy * 11;

      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(tipX, tipY);
      ctx.stroke();

      const headLen = 7.5;
      const angle = Math.atan2(dy, dx);
      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(tipX - headLen * Math.cos(angle - Math.PI / 6), tipY - headLen * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(tipX - headLen * Math.cos(angle + Math.PI / 6), tipY - headLen * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function drawTradeoff(ctx, W, H) {
      const padL = 68;
      const padR = 24;
      const padT = 36;
      const padB = 46;
      const plotW = Math.max(100, W - padL - padR);
      const plotH = Math.max(100, H - padT - padB);

      const yKey = state.yNorm === "per_logical" ? "ler_per_logical" : "ler";
      const isZeroErr = (r) => !r || !(r.num_errors > 0) || Boolean(r.ler_upper_bound);
      // On multi-distance Color Code tradeoff, omit d>=11 0-error points so 256-shot upper bounds don't sit above d=5.
      // On single-code tradeoff plots (like Mitten d=10), keep [B] and [M] 95% Wilson upper-bound points with downward error bars.
      const shouldOmitTradeoffPoint = (cTag, r) => {
        if (!r || !(r.time_per_round > 0) || !(r[yKey] > 0)) return true;
        if (cTag.startsWith("cc_") && Number(r.d) >= 11 && isZeroErr(r)) return true;
        return false;
      };
      const pts = [];
      state.activeCodes.forEach((cTag) => {
        state.activeDecoders.forEach((dId) => {
          const r = byKey[`${cTag}__${dId}`];
          if (!shouldOmitTradeoffPoint(cTag, r)) {
            pts.push(r);
          }
        });
      });

      if (pts.length === 0) return;

      let minX = Math.min(...pts.map((p) => p.time_per_round));
      let maxX = Math.max(...pts.map((p) => p.time_per_round));
      let minY = Math.min(...pts.map((p) => (p.ler_lo > 0 ? p.ler_lo : p[yKey] * 0.35)));
      let maxY = Math.max(...pts.map((p) => (p.ler_hi > 0 ? p.ler_hi : p[yKey] * 1.8)));

      minX *= 0.45;
      maxX *= 2.4;
      minY *= 0.45;
      maxY *= 2.5;

      function mapX(val) {
        let t;
        if (state.xScale === "log") {
          t = (Math.log10(val) - Math.log10(minX)) / (Math.log10(maxX) - Math.log10(minX));
        } else {
          t = (val - 0) / (maxX - 0);
        }
        const centered = (t - 0.5) * state.zoom + 0.5;
        return padL + centered * plotW + state.panX;
      }

      function mapY(val) {
        const t = (Math.log10(val) - Math.log10(minY)) / (Math.log10(maxY) - Math.log10(minY));
        const centered = (t - 0.5) * state.zoom + 0.5;
        return padT + plotH - centered * plotH + state.panY;
      }

      // Title & Plot Description (includes shot counts)
      ctx.fillStyle = "#202124";
      ctx.font = "600 12.5px 'Google Sans', 'DM Sans', sans-serif";
      const chainLabel = ["B", "G", "M", "O"]
        .filter((d) => state.activeDecoders.includes(d))
        .map((d) => `[${d}]`)
        .join(" → ");
      const isMitten = state.activeCodes.length === 1 && state.activeCodes[0] === "mitten_d10";
      const shotsDesc = isMitten
        ? "100k shots ([O]); 95% UB for 0-error runs ([B] 200, [M] 20k)"
        : "100k–150k shots ([O])";
      const titleStr =
        state.yNorm === "per_logical"
          ? `Per-Round Per-Logical LER (÷k) vs. Decode CPU Time per Round (${chainLabel} · ${shotsDesc})`
          : `Logical Error Rate per Round vs. Decode CPU Time per Round (${chainLabel} · ${shotsDesc})`;
      ctx.fillText(titleStr, padL, 22);

      // Clip plot area
      ctx.save();
      ctx.beginPath();
      ctx.rect(padL, padT, plotW, plotH);
      ctx.clip();

      // Grid lines
      ctx.strokeStyle = "#ebecef";
      ctx.lineWidth = 1;
      for (let e = -7; e <= 2; e++) {
        const gx = mapX(Math.pow(10, e));
        if (gx >= padL && gx <= padL + plotW) {
          ctx.beginPath();
          ctx.moveTo(gx, padT);
          ctx.lineTo(gx, padT + plotH);
          ctx.stroke();
        }
        const gy = mapY(Math.pow(10, e));
        if (gy >= padT && gy <= padT + plotH) {
          ctx.beginPath();
          ctx.moveTo(padL, gy);
          ctx.lineTo(padL + plotW, gy);
          ctx.stroke();
        }
      }

      // Draw arrows & points per code
      state.activeCodes.forEach((cTag) => {
        const chainIds = ["B", "G", "M", "O"].filter((d) => {
          if (!state.activeDecoders.includes(d)) return false;
          const rec = byKey[`${cTag}__${d}`];
          return !shouldOmitTradeoffPoint(cTag, rec);
        });
        const chainColors = { B: "#202124", G: "#1967D2", M: "#008080", O: "#008080" };
        for (let i = 0; i < chainIds.length - 1; i++) {
          const r0 = byKey[`${cTag}__${chainIds[i]}`];
          const r1 = byKey[`${cTag}__${chainIds[i + 1]}`];
          if (r0 && r1) {
            drawArrowhead(
              ctx,
              mapX(r0.time_per_round),
              mapY(r0[yKey]),
              mapX(r1.time_per_round),
              mapY(r1[yKey]),
              chainColors[chainIds[i]] || "#5f6368",
              1.9
            );
          }
        }

        // Draw points
        state.activeDecoders.forEach((dId) => {
          const r = byKey[`${cTag}__${dId}`];
          if (shouldOmitTradeoffPoint(cTag, r)) return;
          const cx = mapX(r.time_per_round);
          const cy = mapY(r[yKey]);
          const ubPoint = isZeroErr(r);
          const clampBottom =
            ubPoint ||
            (r.num_errors <= 6 && r.num_shots >= 5000 && r[yKey] <= minY * 25.0) ||
            r[yKey] - r.ler_err_minus <= minY;
          const yLo = mapY(clampBottom ? minY : Math.max(minY, r[yKey] - r.ler_err_minus));
          const yHi = ubPoint ? cy : mapY(r[yKey] + r.ler_err_plus);

          // Error bar (dashed downward to axis floor if 0-error 95% upper bound)
          ctx.save();
          ctx.strokeStyle = r.fill_color === "#f5c242" ? "#d99b00" : r.fill_color;
          ctx.lineWidth = 1.4;
          if (ubPoint) {
            ctx.setLineDash([4, 3]);
          }
          ctx.beginPath();
          ctx.moveTo(cx, yHi);
          ctx.lineTo(cx, yLo);
          ctx.stroke();
          ctx.restore();

          // Outer decoder ring
          ctx.strokeStyle = r.ring_color;
          ctx.lineWidth = 2.6;
          drawMarkerShape(ctx, r.marker, cx, cy, 8.8);
          ctx.stroke();

          // Inner code marker
          ctx.fillStyle = r.fill_color;
          ctx.strokeStyle = "#202124";
          ctx.lineWidth = 1.0;
          drawMarkerShape(ctx, r.marker, cx, cy, 5.6);
          ctx.fill();
          ctx.stroke();

          // Badge letter
          const bx = cx + 10;
          const by = cy - 10;
          ctx.fillStyle = r.ring_color;
          ctx.beginPath();
          ctx.arc(bx, by, 7.2, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.1;
          ctx.stroke();
          ctx.fillStyle = "#ffffff";
          ctx.font = "700 8.5px 'Google Sans', sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(dId, bx, by + 0.5);

          hitTargets.push({
            x: cx,
            y: cy,
            r: 13,
            record: r
          });
        });
      });

      ctx.restore();

      // Axes frame & ticks
      ctx.strokeStyle = "#202124";
      ctx.lineWidth = 1.25;
      ctx.strokeRect(padL, padT, plotW, plotH);

      ctx.fillStyle = "#3c4043";
      ctx.font = "500 10px 'Google Sans', 'Space Mono', sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let e = -6; e <= 1; e++) {
        const v = Math.pow(10, e);
        const gx = mapX(v);
        if (gx >= padL + 15 && gx <= padL + plotW - 15) {
          ctx.beginPath();
          ctx.moveTo(gx, padT + plotH);
          ctx.lineTo(gx, padT + plotH + 4);
          ctx.stroke();
          ctx.fillText(`10^${e}`, gx, padT + plotH + 6);
        }
      }

      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let e = -7; e <= -1; e++) {
        const v = Math.pow(10, e);
        const gy = mapY(v);
        if (gy >= padT + 10 && gy <= padT + plotH - 10) {
          ctx.beginPath();
          ctx.moveTo(padL - 4, gy);
          ctx.lineTo(padL, gy);
          ctx.stroke();
          ctx.fillText(`10^${e}`, padL - 6, gy);
        }
      }

      ctx.fillStyle = "#202124";
      ctx.font = "500 11.5px 'Google Sans', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Decode CPU Time per Round (seconds)", padL + plotW / 2, H - 12);

      ctx.save();
      ctx.translate(15, padT + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(
        state.yNorm === "per_logical" ? "Per-Round Per-Logical LER" : "Logical Error Rate per Round",
        0,
        0
      );
      ctx.restore();
    }

    function drawHyperbolic(ctx, W, H) {
      const padL = 68;
      const padR = 26;
      const padT = 38;
      const padB = 52;
      const plotW = Math.max(120, W - padL - padR);
      const plotH = Math.max(120, H - padT - padB);

      const tags = state.activeCodes.filter((c) => c.startsWith("hyp_"));
      if (tags.length === 0) return;

      const isLogical = state.yNorm === "per_logical";
      const yKey = isLogical ? "ler_per_logical" : "ler";
      const sigKey = isLogical ? "ler_per_logical_1sigma" : "ler_1sigma";

      const minY = isLogical ? 6e-8 : 6e-6;
      const maxY = isLogical ? 1.2e-3 : 1.0e-2;

      function mapY(val) {
        const t = (Math.log10(Math.max(minY, val)) - Math.log10(minY)) / (Math.log10(maxY) - Math.log10(minY));
        return padT + plotH - t * plotH;
      }

      ctx.fillStyle = "#202124";
      ctx.font = "600 12.5px 'Google Sans', 'DM Sans', sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(
        isLogical
          ? "Hyperbolic Octagon Cat-Flagged Circuits: Per-Logical LER (±1σ · 102k shots iter≤58, 24k iter=109, 1M AQ)"
          : "Hyperbolic Octagon Cat-Flagged Circuits: Block LER vs. AlphaQubit (±1σ · 102k shots iter≤58, 24k iter=109, 1M AQ)",
        padL,
        22
      );

      // Horizontal & vertical gridlines (Quantum AI template style)
      ctx.strokeStyle = "#ebecef";
      ctx.lineWidth = 1;
      for (let e = -7; e <= -2; e++) {
        const gy = mapY(Math.pow(10, e));
        if (gy >= padT && gy <= padT + plotH) {
          ctx.beginPath();
          ctx.moveTo(padL, gy);
          ctx.lineTo(padL + plotW, gy);
          ctx.stroke();

          ctx.fillStyle = "#3c4043";
          ctx.font = "500 10px 'Google Sans', 'Space Mono', sans-serif";
          ctx.textAlign = "right";
          ctx.textBaseline = "middle";
          ctx.fillText(`10^${e}`, padL - 7, gy);
        }
      }

      const groupW = plotW / tags.length;
      tags.forEach((t, tIdx) => {
        const gx = padL + groupW * (tIdx + 0.5);
        ctx.strokeStyle = "#f1f3f4";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(gx, padT);
        ctx.lineTo(gx, padT + plotH);
        ctx.stroke();
      });

      const decSeries = [
        { id: "AQ", label: "AQ Baseline (AlphaQubit)", color: "#202124", shape: "D", dash: [6, 4], jitter: 0 },
        { id: "O", label: "[O] Tesseract-BP-OSD (Ours)", color: "#008080", shape: "s", dash: [], jitter: 0 }
      ].filter((s) => state.activeDecoders.includes(s.id));

      // First pass: draw connecting lines for each decoder series
      decSeries.forEach((s) => {
        const pts = [];
        tags.forEach((t, tIdx) => {
          const rec = byKey[`${t}__${s.id}`];
          if (!rec) return;
          const px = padL + groupW * (tIdx + 0.5) + s.jitter;
          const py = mapY(rec[yKey]);
          pts.push({ x: px, y: py, rec, t });
        });

        if (pts.length > 1) {
          ctx.save();
          ctx.strokeStyle = s.color;
          ctx.lineWidth = s.id === "O" ? 2.5 : 2.1;
          ctx.setLineDash(s.dash);
          ctx.beginPath();
          pts.forEach((pt, idx) => {
            if (idx === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
          });
          ctx.stroke();
          ctx.restore();
        }
      });

      // Second pass: draw ±1σ error bars with caps, markers, and concise ratio annotations
      decSeries.forEach((s) => {
        tags.forEach((t, tIdx) => {
          const rec = byKey[`${t}__${s.id}`];
          if (!rec) return;
          const px = padL + groupW * (tIdx + 0.5) + s.jitter;
          const val = rec[yKey];
          const py = mapY(val);
          const yTop = mapY(val + rec[sigKey]);
          const yBot = mapY(Math.max(minY, val - rec[sigKey]));

          // Error bar + horizontal caps
          ctx.strokeStyle = s.color;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(px, yTop);
          ctx.lineTo(px, yBot);
          ctx.moveTo(px - 4.5, yTop);
          ctx.lineTo(px + 4.5, yTop);
          ctx.moveTo(px - 4.5, yBot);
          ctx.lineTo(px + 4.5, yBot);
          ctx.stroke();

          // Marker with white halo
          ctx.fillStyle = "#ffffff";
          drawMarkerShape(ctx, s.shape, px, py, 6.8);
          ctx.fill();

          ctx.fillStyle = s.color;
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.2;
          drawMarkerShape(ctx, s.shape, px, py, 5.4);
          ctx.fill();
          ctx.stroke();

          hitTargets.push({
            x: px,
            y: py,
            r: 14,
            isRect: false,
            record: rec
          });
        });
      });

      // Outer plot frame
      ctx.strokeStyle = "#202124";
      ctx.lineWidth = 1.25;
      ctx.strokeRect(padL, padT, plotW, plotH);

      // In-canvas dashed-border legend box (Google Quantum AI Slide 49 style)
      const legX = padL + 14;
      const legY = padT + plotH - (decSeries.length * 19 + 14);
      const legW = 212;
      const legH = decSeries.length * 19 + 8;
      ctx.save();
      ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
      ctx.fillRect(legX, legY, legW, legH);
      ctx.strokeStyle = "#9aa0a6";
      ctx.lineWidth = 1.1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(legX, legY, legW, legH);
      ctx.setLineDash([]);

      decSeries.forEach((s, idx) => {
        const ry = legY + 13 + idx * 19;
        ctx.save();
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2.0;
        ctx.setLineDash(s.dash);
        ctx.beginPath();
        ctx.moveTo(legX + 10, ry);
        ctx.lineTo(legX + 32, ry);
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = s.color;
        drawMarkerShape(ctx, s.shape, legX + 21, ry, 4.2);
        ctx.fill();

        ctx.fillStyle = "#202124";
        ctx.font = "500 10px 'Google Sans', sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(s.label, legX + 40, ry);
      });
      ctx.restore();

      // X-axis tick marks & labels
      tags.forEach((t, tIdx) => {
        const cm = dataset.circuits[t];
        const cx = padL + groupW * (tIdx + 0.5);
        ctx.strokeStyle = "#202124";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(cx, padT + plotH);
        ctx.lineTo(cx, padT + plotH + 4);
        ctx.stroke();

        ctx.fillStyle = "#202124";
        ctx.font = "600 10.5px 'Google Sans', sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(`iter=${cm.iter} (n=${cm.n}, k=${cm.k})`, cx, padT + plotH + 8);
      });

      // Y-axis label
      ctx.save();
      ctx.fillStyle = "#202124";
      ctx.font = "500 11.5px 'Google Sans', sans-serif";
      ctx.textAlign = "center";
      ctx.translate(15, padT + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(
        isLogical ? "Per-Round Per-Logical LER (÷k)" : "Per-Round Block LER",
        0,
        0
      );
      ctx.restore();
    }

    function drawStageBreakdown(ctx, W, H) {
      const padL = 50;
      const padR = 16;
      const padT = 36;
      const padB = 56;
      const gapMid = 40;
      const panelW = Math.max(100, (W - padL - padR - gapMid) / 2);
      const plotH = Math.max(100, H - padT - padB);

      const tags = state.activeCodes;
      if (tags.length === 0) return;
      const allCc = tags.every((t) => t.startsWith("cc_d"));

      const panels = [
        {
          x0: padL,
          title: "(a) % of Shots Resolved per Stage",
          suffix: "_shots_pct"
        },
        {
          x0: padL + panelW + gapMid,
          title: "(b) % of Total Decode CPU Time per Stage",
          suffix: "_cpu_pct"
        }
      ];

      const stageKeys = ["stage1_bp", "stage1_osd"];

      panels.forEach((p) => {
        ctx.fillStyle = "#202124";
        ctx.font = "600 11.5px 'Google Sans', sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(p.title, p.x0, 22);

        // Y grid (0..100%)
        [0, 20, 40, 60, 80, 100].forEach((pct) => {
          const gy = padT + plotH - (pct / 100) * plotH;
          ctx.strokeStyle = "#ebecef";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(p.x0, gy);
          ctx.lineTo(p.x0 + panelW, gy);
          ctx.stroke();

          ctx.fillStyle = "#3c4043";
          ctx.font = "500 9.5px 'Google Sans', 'Space Mono', sans-serif";
          ctx.textAlign = "right";
          ctx.textBaseline = "middle";
          ctx.fillText(`${pct}%`, p.x0 - 5, gy);
        });

        const slotW = panelW / tags.length;
        const barW = Math.min(34, slotW * 0.68);

        tags.forEach((t, tIdx) => {
          const rec = byKey[`${t}__O`];
          if (!rec || !rec.telemetry_ok) return;
          const cx = p.x0 + slotW * (tIdx + 0.5);
          let curBottom = 0;

          stageKeys.forEach((sKey) => {
            const val = rec[`${sKey}${p.suffix}`] || 0;
            const yTop = padT + plotH - ((curBottom + val) / 100) * plotH;
            const bh = Math.max(1.5, (val / 100) * plotH);

            ctx.fillStyle = STAGE_COLORS[sKey].color;
            ctx.fillRect(cx - barW / 2, yTop, barW, bh);
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 0.7;
            ctx.strokeRect(cx - barW / 2, yTop, barW, bh);

            if (val >= 12 && barW >= 18) {
              ctx.fillStyle = "#ffffff";
              ctx.font = `700 ${tags.length > 7 ? "7.5px" : "8.5px"} 'Google Sans', sans-serif`;
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillText(`${Math.round(val)}%`, cx, yTop + bh / 2);
            }

            curBottom += val;
          });

          hitTargets.push({
            x: cx,
            y: padT + plotH / 2,
            w: barW,
            h: plotH,
            isRect: true,
            record: rec
          });

          // X label
          const shortName = allCc
            ? t.replace("cc_d", "d=")
            : t
                .replace("cc_d", "CC d=")
                .replace("bb_d6", "BB d=6")
                .replace("bb_d12_gross", "Gross")
                .replace("mitten_d10", "Mitten")
                .replace("hyp_iter", "i=");
          ctx.fillStyle = "#202124";
          ctx.font = `600 ${tags.length > 8 ? "8.5px" : "9px"} 'Google Sans', sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillText(shortName, cx, padT + plotH + 6);
        });

        ctx.strokeStyle = "#202124";
        ctx.lineWidth = 1.2;
        ctx.strokeRect(p.x0, padT, panelW, plotH);
      });

      // Legend at bottom with dashed border (Quantum AI template style)
      let lx = padL;
      const ly = H - 14;
      stageKeys.forEach((sKey) => {
        const st = STAGE_COLORS[sKey];
        ctx.fillStyle = st.color;
        ctx.fillRect(lx, ly - 8, 10, 10);
        ctx.fillStyle = "#202124";
        ctx.font = "500 9.5px 'Google Sans', sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(st.label, lx + 14, ly - 3);
        lx += ctx.measureText(st.label).width + 24;
      });
    }

    /**
     * Renders a publication-grade 2-panel vertically stacked scientific chart
     * (Top = Logical Error Rate per Round, Bottom = Decode CPU Time per Round)
     * with shared X-axis, shaded ±1σ error bands (ctx.fill()), ±1σ error bars
     * with horizontal caps, upper-bound ▽ triangles where num_errors == 0,
     * and distinct markers for all 7 decoders [B], [G], [M], [O], [C], [R], [T].
     */
    function drawStackedDualPanel(ctx, W, H, cfg) {
      const padL = 66;
      const padR = 20;
      const padT = 22;
      const gapMid = 32;
      const padB = 38;
      const plotW = Math.max(160, W - padL - padR);
      const subH = Math.max(92, (H - padT - padB - gapMid) / 2);
      const topY0 = padT;
      const botY0 = padT + subH + gapMid;

      const isLogical = state.yNorm === "per_logical";
      const yKey = isLogical ? "ler_per_logical" : "ler";
      const sigKey = isLogical ? "ler_per_logical_1sigma" : "ler_1sigma";
      const loKey = isLogical ? "ler_per_logical_lo" : "ler_lo";
      const hiKey = isLogical ? "ler_per_logical_hi" : "ler_hi";

      const activeSeries = SERIES_STYLE_7.filter((s) => state.activeDecoders.includes(s.id));

      // Gather series data for Top (LER) and Bottom (Time)
      const topSeriesPts = {};
      const botSeriesPts = {};
      const allLerVals = [];
      const allTimeVals = [];

      activeSeries.forEach((s) => {
        topSeriesPts[s.id] = [];
        botSeriesPts[s.id] = [];
        cfg.getRecordsForDecoder(s.id).forEach((rec) => {
          const xVal = cfg.getX(rec);
          if (xVal === null || xVal === undefined || isNaN(xVal)) return;
          const xPx = cfg.mapX(xVal, padL, plotW);

          // Top Subplot (LER): NEVER plot points where num_errors == 0
          const lerVal = rec[yKey];
          const hasErrors = rec.num_errors > 0 && !rec.ler_upper_bound && lerVal && lerVal > 0;
          if (hasErrors) {
            let sig = rec[sigKey] || 0;
            if (!sig && rec[hiKey] > 0 && rec[loKey] >= 0) {
              sig = (rec[hiKey] - rec[loKey]) * 0.5;
            }
            const yHi = lerVal + (sig > 0 ? sig : lerVal * 0.25);
            const rawLo = lerVal - (sig > 0 ? sig : lerVal * 0.22);
            const lowErrCandidate = rec.num_errors <= 6 && rec.num_shots >= 5000;
            allLerVals.push(lerVal, yHi);
            if (!lowErrCandidate && rawLo > 0) {
              allLerVals.push(rawLo);
            }
            topSeriesPts[s.id].push({
              x: xPx,
              xVal,
              val: lerVal,
              hi: yHi,
              rawLo: rawLo > 0 ? rawLo : (rec[loKey] > 0 ? rec[loKey] : 0),
              lowErrCandidate,
              rec
            });
          }

          // Bottom Subplot (Runtime): KEEP all measured points (even when num_errors == 0)
          const tVal = rec.time_per_round;
          if (tVal && tVal > 0) {
            const tSig = rec.time_1sigma > 0 ? rec.time_1sigma : tVal * 0.12;
            const tHi = tVal + tSig;
            const tLo = Math.max(tVal * 0.25, tVal - tSig);
            allTimeVals.push(tLo, tHi, tVal);
            botSeriesPts[s.id].push({
              x: xPx,
              xVal,
              val: tVal,
              hi: tHi,
              lo: tLo,
              rec
            });
          }
        });
        topSeriesPts[s.id].sort((a, b) => a.xVal - b.xVal);
        botSeriesPts[s.id].sort((a, b) => a.xVal - b.xVal);
      });

      const minLer = allLerVals.length > 0 ? Math.min(...allLerVals) * 0.48 : 1e-7;
      const maxLer = allLerVals.length > 0 ? Math.max(...allLerVals) * 2.3 : 1e-1;
      const minTime = allTimeVals.length > 0 ? Math.min(...allTimeVals) * 0.45 : 1e-6;
      const maxTime = allTimeVals.length > 0 ? Math.max(...allTimeVals) * 2.4 : 1.0;

      // Clamp lower error bar to the bottom of the log axis for low-error-count points near the floor
      activeSeries.forEach((s) => {
        (topSeriesPts[s.id] || []).forEach((pt) => {
          const clampBottom =
            (pt.lowErrCandidate && pt.val <= minLer * 25.0) || pt.rawLo <= minLer;
          pt.lo = clampBottom ? minLer : pt.rawLo;
        });
      });

      function mapYTop(v) {
        const clamped = Math.max(minLer, Math.min(maxLer, v));
        const t = (Math.log10(clamped) - Math.log10(minLer)) / (Math.log10(maxLer) - Math.log10(minLer));
        return topY0 + subH - t * subH;
      }

      function mapYBot(v) {
        const clamped = Math.max(minTime, Math.min(maxTime, v));
        const t = (Math.log10(clamped) - Math.log10(minTime)) / (Math.log10(maxTime) - Math.log10(minTime));
        return botY0 + subH - t * subH;
      }

      // -----------------------------------------------------------------------
      // SUBPLOT (a) — TOP: Logical Error Rate per Round
      // -----------------------------------------------------------------------
      ctx.fillStyle = "#202124";
      ctx.font = "600 11.5px 'Google Sans', 'DM Sans', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(cfg.topTitle(isLogical), padL, topY0 - 4);

      // Horizontal log gridlines & Y ticks for Top Subplot
      const minLerExp = Math.floor(Math.log10(minLer));
      const maxLerExp = Math.ceil(Math.log10(maxLer));
      for (let e = minLerExp; e <= maxLerExp; e++) {
        const gy = mapYTop(Math.pow(10, e));
        if (gy >= topY0 - 1 && gy <= topY0 + subH + 1) {
          ctx.strokeStyle = "#ebecef";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(padL, gy);
          ctx.lineTo(padL + plotW, gy);
          ctx.stroke();

          ctx.fillStyle = "#3c4043";
          ctx.font = "500 9.5px 'Google Sans', 'Space Mono', sans-serif";
          ctx.textAlign = "right";
          ctx.textBaseline = "middle";
          ctx.fillText(`10^${e}`, padL - 6, gy);
        }
      }

      // Vertical shared X gridlines in Top Subplot
      cfg.xTicks.forEach((tk) => {
        const gx = cfg.mapX(tk.val, padL, plotW);
        if (gx >= padL && gx <= padL + plotW) {
          ctx.strokeStyle = "#f1f3f4";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(gx, topY0);
          ctx.lineTo(gx, topY0 + subH);
          ctx.stroke();
        }
      });

      ctx.save();
      ctx.beginPath();
      ctx.rect(padL, topY0, plotW, subH);
      ctx.clip();

      // Pass 1 (Top): Shaded ±1σ error bands (ctx.fill())
      activeSeries.forEach((s) => {
        const pts = topSeriesPts[s.id];
        if (!pts || pts.length === 0) return;
        ctx.save();
        ctx.beginPath();
        if (pts.length === 1) {
          const p0 = pts[0];
          ctx.rect(p0.x - 8, mapYTop(p0.hi), 16, Math.max(2, mapYTop(p0.lo) - mapYTop(p0.hi)));
        } else {
          pts.forEach((pt, idx) => {
            const py = mapYTop(pt.hi);
            if (idx === 0) ctx.moveTo(pt.x, py);
            else ctx.lineTo(pt.x, py);
          });
          for (let idx = pts.length - 1; idx >= 0; idx--) {
            ctx.lineTo(pts[idx].x, mapYTop(pts[idx].lo));
          }
          ctx.closePath();
        }
        ctx.fillStyle = hexToRgba(s.color, s.id === "O" ? 0.19 : 0.13);
        ctx.fill();
        ctx.restore();
      });

      // Pass 2 (Top): Connecting lines
      activeSeries.forEach((s) => {
        const pts = topSeriesPts[s.id];
        if (!pts || pts.length < 2) return;
        ctx.save();
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.lw;
        ctx.setLineDash(s.dash);
        ctx.beginPath();
        pts.forEach((pt, idx) => {
          const py = mapYTop(pt.val);
          if (idx === 0) ctx.moveTo(pt.x, py);
          else ctx.lineTo(pt.x, py);
        });
        ctx.stroke();
        ctx.restore();
      });

      // Pass 3 (Top): ±1σ error bars with horizontal caps, markers, and ▽ upper bounds
      activeSeries.forEach((s) => {
        const pts = topSeriesPts[s.id];
        if (!pts) return;
        pts.forEach((pt) => {
          const py = mapYTop(pt.val);
          const yTopPx = mapYTop(pt.hi);
          const yBotPx = mapYTop(pt.lo);

          if (pt.isUpper) {
            // Downward arrow + open downward upper-bound triangle ▽
            const arrowTip = Math.min(topY0 + subH - 3, py + 14);
            ctx.save();
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(pt.x, py);
            ctx.lineTo(pt.x, arrowTip);
            ctx.moveTo(pt.x - 3.5, arrowTip - 4);
            ctx.lineTo(pt.x, arrowTip);
            ctx.lineTo(pt.x + 3.5, arrowTip - 4);
            ctx.stroke();

            ctx.fillStyle = "#ffffff";
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 1.9;
            drawMarkerShape(ctx, "v", pt.x, py, 5.4);
            ctx.fill();
            ctx.stroke();
            ctx.restore();
          } else {
            ctx.save();
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 1.35;
            ctx.beginPath();
            ctx.moveTo(pt.x, yTopPx);
            ctx.lineTo(pt.x, yBotPx);
            ctx.moveTo(pt.x - 3.6, yTopPx);
            ctx.lineTo(pt.x + 3.6, yTopPx);
            ctx.moveTo(pt.x - 3.6, yBotPx);
            ctx.lineTo(pt.x + 3.6, yBotPx);
            ctx.stroke();

            ctx.fillStyle = "#ffffff";
            drawMarkerShape(ctx, s.shape, pt.x, py, 5.8);
            ctx.fill();

            ctx.fillStyle = s.color;
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.0;
            drawMarkerShape(ctx, s.shape, pt.x, py, 4.6);
            ctx.fill();
            ctx.stroke();
            ctx.restore();
          }

          hitTargets.push({
            x: pt.x,
            y: py,
            r: 11,
            isRect: false,
            record: pt.rec
          });
        });
      });

      ctx.restore();

      // Top Subplot outer frame
      ctx.strokeStyle = "#202124";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(padL, topY0, plotW, subH);

      // Top Subplot Y-axis label
      ctx.save();
      ctx.fillStyle = "#202124";
      ctx.font = "500 10.5px 'Google Sans', sans-serif";
      ctx.textAlign = "center";
      ctx.translate(14, topY0 + subH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(isLogical ? "Logical LER / Rnd (÷k)" : "LER per Round", 0, 0);
      ctx.restore();

      // Compact multi-column legend box inside Top Subplot (Google Quantum AI template style)
      if (activeSeries.length > 0) {
        const cols = activeSeries.length > 4 ? 2 : 1;
        const rows = Math.ceil(activeSeries.length / cols);
        const colW = 128;
        const rowH = 14.5;
        const legW = cols * colW + 10;
        const legH = rows * rowH + 8;
        const legX = cfg.legendPos === "top-left" ? padL + 8 : padL + plotW - legW - 8;
        const legY = cfg.legendPos === "bottom-right" ? topY0 + subH - legH - 6 : topY0 + 6;

        ctx.save();
        ctx.fillStyle = "rgba(255, 255, 255, 0.93)";
        ctx.fillRect(legX, legY, legW, legH);
        ctx.strokeStyle = "#9aa0a6";
        ctx.lineWidth = 1.0;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(legX, legY, legW, legH);
        ctx.setLineDash([]);

        activeSeries.forEach((s, idx) => {
          const cIdx = Math.floor(idx / rows);
          const rIdx = idx % rows;
          const ix = legX + 6 + cIdx * colW;
          const iy = legY + 10 + rIdx * rowH;

          // Shaded ±1σ band swatch behind line
          ctx.fillStyle = hexToRgba(s.color, 0.22);
          ctx.fillRect(ix, iy - 4.5, 20, 9);

          ctx.save();
          ctx.strokeStyle = s.color;
          ctx.lineWidth = 1.8;
          ctx.setLineDash(s.dash);
          ctx.beginPath();
          ctx.moveTo(ix, iy);
          ctx.lineTo(ix + 20, iy);
          ctx.stroke();
          ctx.restore();

          ctx.fillStyle = s.color;
          drawMarkerShape(ctx, s.shape, ix + 10, iy, 3.6);
          ctx.fill();

          ctx.fillStyle = "#202124";
          ctx.font = "500 8.8px 'Google Sans', sans-serif";
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText(s.short, ix + 24, iy);
        });
        ctx.restore();
      }

      // -----------------------------------------------------------------------
      // SUBPLOT (b) — BOTTOM: Decode CPU Time per Round (seconds)
      // -----------------------------------------------------------------------
      ctx.fillStyle = "#202124";
      ctx.font = "600 11.5px 'Google Sans', 'DM Sans', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(cfg.botTitle, padL, botY0 - 4);

      const minTimeExp = Math.floor(Math.log10(minTime));
      const maxTimeExp = Math.ceil(Math.log10(maxTime));
      for (let e = minTimeExp; e <= maxTimeExp; e++) {
        const gy = mapYBot(Math.pow(10, e));
        if (gy >= botY0 - 1 && gy <= botY0 + subH + 1) {
          ctx.strokeStyle = "#ebecef";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(padL, gy);
          ctx.lineTo(padL + plotW, gy);
          ctx.stroke();

          ctx.fillStyle = "#3c4043";
          ctx.font = "500 9.5px 'Google Sans', 'Space Mono', sans-serif";
          ctx.textAlign = "right";
          ctx.textBaseline = "middle";
          ctx.fillText(`10^${e}`, padL - 6, gy);
        }
      }

      // Vertical shared X gridlines in Bottom Subplot
      cfg.xTicks.forEach((tk) => {
        const gx = cfg.mapX(tk.val, padL, plotW);
        if (gx >= padL && gx <= padL + plotW) {
          ctx.strokeStyle = "#f1f3f4";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(gx, botY0);
          ctx.lineTo(gx, botY0 + subH);
          ctx.stroke();
        }
      });

      ctx.save();
      ctx.beginPath();
      ctx.rect(padL, botY0, plotW, subH);
      ctx.clip();

      // Pass 1 (Bottom): Shaded ±1σ runtime bands (ctx.fill())
      activeSeries.forEach((s) => {
        const pts = botSeriesPts[s.id];
        if (!pts || pts.length === 0) return;
        ctx.save();
        ctx.beginPath();
        if (pts.length === 1) {
          const p0 = pts[0];
          ctx.rect(p0.x - 8, mapYBot(p0.hi), 16, Math.max(2, mapYBot(p0.lo) - mapYBot(p0.hi)));
        } else {
          pts.forEach((pt, idx) => {
            const py = mapYBot(pt.hi);
            if (idx === 0) ctx.moveTo(pt.x, py);
            else ctx.lineTo(pt.x, py);
          });
          for (let idx = pts.length - 1; idx >= 0; idx--) {
            ctx.lineTo(pts[idx].x, mapYBot(pts[idx].lo));
          }
          ctx.closePath();
        }
        ctx.fillStyle = hexToRgba(s.color, s.id === "O" ? 0.18 : 0.13);
        ctx.fill();
        ctx.restore();
      });

      // Pass 2 (Bottom): Connecting lines
      activeSeries.forEach((s) => {
        const pts = botSeriesPts[s.id];
        if (!pts || pts.length < 2) return;
        ctx.save();
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.lw;
        ctx.setLineDash(s.dash);
        ctx.beginPath();
        pts.forEach((pt, idx) => {
          const py = mapYBot(pt.val);
          if (idx === 0) ctx.moveTo(pt.x, py);
          else ctx.lineTo(pt.x, py);
        });
        ctx.stroke();
        ctx.restore();
      });

      // Pass 3 (Bottom): ±1σ runtime error bars with caps and markers
      activeSeries.forEach((s) => {
        const pts = botSeriesPts[s.id];
        if (!pts) return;
        pts.forEach((pt) => {
          const py = mapYBot(pt.val);
          const yTopPx = mapYBot(pt.hi);
          const yBotPx = mapYBot(pt.lo);

          ctx.save();
          ctx.strokeStyle = s.color;
          ctx.lineWidth = 1.3;
          ctx.beginPath();
          ctx.moveTo(pt.x, yTopPx);
          ctx.lineTo(pt.x, yBotPx);
          ctx.moveTo(pt.x - 3.2, yTopPx);
          ctx.lineTo(pt.x + 3.2, yTopPx);
          ctx.moveTo(pt.x - 3.2, yBotPx);
          ctx.lineTo(pt.x + 3.2, yBotPx);
          ctx.stroke();

          ctx.fillStyle = "#ffffff";
          drawMarkerShape(ctx, s.shape, pt.x, py, 5.5);
          ctx.fill();

          ctx.fillStyle = s.color;
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.0;
          drawMarkerShape(ctx, s.shape, pt.x, py, 4.4);
          ctx.fill();
          ctx.stroke();
          ctx.restore();

          hitTargets.push({
            x: pt.x,
            y: py,
            r: 11,
            isRect: false,
            record: pt.rec
          });
        });
      });

      ctx.restore();

      // Bottom Subplot outer frame
      ctx.strokeStyle = "#202124";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(padL, botY0, plotW, subH);

      // Bottom Subplot Y-axis label
      ctx.save();
      ctx.fillStyle = "#202124";
      ctx.font = "500 10.5px 'Google Sans', sans-serif";
      ctx.textAlign = "center";
      ctx.translate(14, botY0 + subH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("CPU Time / Round (s)", 0, 0);
      ctx.restore();

      // Shared X-axis tick marks & labels below Bottom Subplot
      cfg.xTicks.forEach((tk) => {
        const gx = cfg.mapX(tk.val, padL, plotW);
        if (gx >= padL - 2 && gx <= padL + plotW + 2) {
          ctx.strokeStyle = "#202124";
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(gx, botY0 + subH);
          ctx.lineTo(gx, botY0 + subH + 4);
          ctx.stroke();

          ctx.fillStyle = "#202124";
          ctx.font = "600 9.5px 'Google Sans', 'Space Mono', sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillText(tk.label, gx, botY0 + subH + 6);
        }
      });

      // Shared X-axis title
      ctx.fillStyle = "#202124";
      ctx.font = "500 11px 'Google Sans', sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(cfg.xLabel, padL + plotW / 2, H - 3);
    }

    function drawCcScaling(ctx, W, H) {
      const distances = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
      drawStackedDualPanel(ctx, W, H, {
        legendPos: "top-right",
        topTitle: (isLog) =>
          isLog
            ? "(a) Per-Logical LER per Round vs. Distance d (SI1000 p = 0.001 · up to 150k shots · ±1σ Bands)"
            : "(a) Logical Error Rate per Round vs. Distance d (SI1000 p = 0.001 · up to 150k shots · ±1σ Bands)",
        botTitle: "(b) Decode CPU Time per Round (seconds) vs. Distance d (Log Scale, ±1σ Bands)",
        xLabel: "Code Distance d (Superdense Color Code Z-Memory, r = d, q = 13 .. 793)",
        xTicks: distances.map((d) => ({ val: d, label: `d=${d}` })),
        getX: (rec) => Number(rec.d),
        mapX: (dVal, padL, plotW) => {
          const inset = 22;
          const t = (dVal - 3) / (23 - 3);
          return padL + inset + t * (plotW - 2 * inset);
        },
        getRecordsForDecoder: (decId) => {
          const out = [];
          distances.forEach((d) => {
            const r = byKey[`cc_d${d}__${decId}`];
            if (r) out.push(r);
          });
          return out;
        }
      });
    }

    function drawBbPsweep(ctx, W, H) {
      const codeTag = state.bbCode || state.activeCodes[0] || "bb_d6";
      const codeRecs = getPsweepRecordsForCode(codeTag);
      const presentP = Array.from(new Set(codeRecs.map((r) => Number(r.p || 0.001)))).sort((a, b) => a - b);
      const defaultP = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005];
      const tickPVals = Array.from(new Set([...defaultP, ...presentP])).sort((a, b) => a - b);
      const minP = Math.min(...tickPVals) * 0.85;
      const maxP = Math.max(...tickPVals) * 1.15;

      const codeTitle =
        codeTag === "bb_d12_gross"
          ? "Gross [[144, 12, 12]] (d = 12, up to 50k shots/pt)"
          : "BB [[72, 12, 6]] (d = 6, up to 100k shots/pt)";

      const fmtP = (pVal) => {
        if (Math.abs(pVal - 0.0001) < 1e-7) return "10⁻⁴";
        if (Math.abs(pVal - 0.0005) < 1e-7) return "5×10⁻⁴";
        if (Math.abs(pVal - 0.001) < 1e-7) return "10⁻³";
        if (Math.abs(pVal - 0.002) < 1e-7) return "2×10⁻³";
        if (Math.abs(pVal - 0.003) < 1e-7) return "3×10⁻³";
        if (Math.abs(pVal - 0.004) < 1e-7) return "4×10⁻³";
        if (Math.abs(pVal - 0.005) < 1e-7) return "5×10⁻³";
        return String(pVal);
      };

      drawStackedDualPanel(ctx, W, H, {
        legendPos: "top-left",
        topTitle: (isLog) =>
          isLog
            ? `(a) Per-Logical LER per Round vs. Physical Error Rate p — ${codeTitle} (±1σ Bands)`
            : `(a) Logical Error Rate per Round vs. Physical Error Rate p — ${codeTitle} (±1σ Bands)`,
        botTitle: `(b) Decode CPU Time per Round (seconds) vs. Physical Error Rate p — ${codeTitle}`,
        xLabel: "Physical Error Rate p (SI1000 Circuit-Level Noise, Log Scale)",
        xTicks: tickPVals.map((pv) => ({ val: pv, label: fmtP(pv) })),
        getX: (rec) => Number(rec.p || 0.001),
        mapX: (pVal, padL, plotW) => {
          const inset = 22;
          const clamped = Math.max(minP, Math.min(maxP, pVal));
          const t = (Math.log10(clamped) - Math.log10(minP)) / (Math.log10(maxP) - Math.log10(minP));
          return padL + inset + t * (plotW - 2 * inset);
        },
        getRecordsForDecoder: (decId) =>
          codeRecs.filter((r) => r.decoder_id === decId)
      });
    }

    function draw() {
      const rect = canvasWrap.getBoundingClientRect();
      const W = Math.max(320, rect.width || 640);
      const H = Math.max(280, rect.height || 400);
      const dpr = window.devicePixelRatio || 1;
      canvas.width = W * dpr;
      canvas.height = H * dpr;

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      hitTargets = [];
      if (state.mode === "cc_scaling") {
        drawCcScaling(ctx, W, H);
      } else if (state.mode === "bb_psweep") {
        drawBbPsweep(ctx, W, H);
      } else if (state.mode === "tradeoff") {
        drawTradeoff(ctx, W, H);
      } else if (state.mode === "hyperbolic") {
        drawHyperbolic(ctx, W, H);
      } else {
        drawStageBreakdown(ctx, W, H);
      }
    }

    // Mouse interaction: hover tooltip, drag pan, wheel zoom
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;

    function onMouseDown(e) {
      isDragging = true;
      dragStartX = e.clientX - state.panX;
      dragStartY = e.clientY - state.panY;
    }

    function onMouseUp() {
      isDragging = false;
    }

    canvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mouseup", onMouseUp);

    canvas.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      if (isDragging && state.mode === "tradeoff") {
        state.panX = e.clientX - dragStartX;
        state.panY = e.clientY - dragStartY;
        renderToolbar();
        draw();
        return;
      }

      let found = null;
      for (let i = hitTargets.length - 1; i >= 0; i--) {
        const t = hitTargets[i];
        if (t.isRect) {
          if (Math.abs(mx - t.x) <= t.w / 2 + 3 && Math.abs(my - t.y) <= t.h / 2) {
            found = t.record;
            break;
          }
        } else {
          if (Math.hypot(mx - t.x, my - t.y) <= t.r) {
            found = t.record;
            break;
          }
        }
      }

      if (!found) {
        tooltip.style.display = "none";
        return;
      }

      const r = found;
      const pf = (v, nd) => (v === null || v === undefined ? "n/a" : `${v.toFixed(nd)}%`);
      const speedupStr = r.speedup_vs_B
        ? `${r.speedup_vs_B}x vs [B]` + (r.speedup_vs_G ? ` (${r.speedup_vs_G}x vs [G])` : "")
        : "n/a";
      const lerStr = r.ler_upper_bound
        ? `< ${formatSci(r.ler)} (95% upper bound)`
        : `${formatSci(r.ler)} ± ${formatSci(r.ler_1sigma, 1)}`;
      const timeStr = r.time_per_shot ? `${formatTime(r.time_per_shot)} (${formatTime(r.time_per_round)}/rnd)` : "n/a";
      const hasStages = r.stage1_bp_shots_pct !== null && r.stage1_bp_shots_pct !== undefined;
      const stageRows = hasStages
        ? `
          <span class="tviz-tooltip-key">Shots BP / OSD:</span>
          <span class="tviz-tooltip-val">${pf(r.stage1_bp_shots_pct, 1)} / ${pf(r.stage1_osd_shots_pct, 1)}</span>
          <span class="tviz-tooltip-key">CPU BP / OSD:</span>
          <span class="tviz-tooltip-val">${pf(r.stage1_bp_cpu_pct, 1)} / ${pf(r.stage1_osd_cpu_pct, 1)}</span>`
        : "";

      tooltip.innerHTML = `
        <div class="tviz-tooltip-title">${r.title_label} · ${r.decoder_short}</div>
        <div class="tviz-tooltip-grid">
          <span class="tviz-tooltip-key">Physical Noise p:</span>
          <span class="tviz-tooltip-val">${r.p || 0.001} (d=${r.d || "—"}, r=${r.r || "—"})</span>
          <span class="tviz-tooltip-key">Sampled Shots:</span>
          <span class="tviz-tooltip-val">${r.num_shots.toLocaleString()}</span>
          <span class="tviz-tooltip-key">Block LER / Rnd:</span>
          <span class="tviz-tooltip-val">${lerStr}</span>
          <span class="tviz-tooltip-key">Per-Logical LER:</span>
          <span class="tviz-tooltip-val">${r.ler_upper_bound ? "< " : ""}${formatSci(r.ler_per_logical)} (k=${r.k})</span>
          <span class="tviz-tooltip-key">CPU Time / Shot:</span>
          <span class="tviz-tooltip-val">${timeStr}</span>
          ${speedupStr !== "n/a" ? `<span class="tviz-tooltip-key">Speedup:</span><span class="tviz-tooltip-val" style="color:#4ade80;font-weight:700;">${speedupStr}</span>` : ""}${stageRows}
          <span class="tviz-tooltip-key">Source:</span>
          <span class="tviz-tooltip-val">${r.json_source}</span>
        </div>
        <div class="tviz-tooltip-cli">$ ${r.command_line}</div>
      `;
      tooltip.style.display = "block";
      const tx = Math.min(rect.width - 345, Math.max(8, mx + 14));
      const ty = Math.min(rect.height - 195, Math.max(8, my - 14));
      tooltip.style.left = `${tx}px`;
      tooltip.style.top = `${ty}px`;
    });

    canvas.addEventListener("mouseleave", () => {
      tooltip.style.display = "none";
    });

    canvas.addEventListener(
      "wheel",
      (e) => {
        if (state.mode !== "tradeoff") return;
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        state.zoom = Math.max(0.6, Math.min(6.0, state.zoom * factor));
        renderToolbar();
        draw();
      },
      { passive: false }
    );

    const onResize = () => draw();
    window.addEventListener("resize", onResize);

    renderToolbar();
    const initTimer = setTimeout(draw, 10);

    return {
      cleanup: function () {
        clearTimeout(initTimer);
        window.removeEventListener("mouseup", onMouseUp);
        window.removeEventListener("resize", onResize);
      }
    };
  }

  window.TesseractPlotWidget = {
    mount: mountVizPlot
  };
  window.TesseractViz = window.TesseractViz || {};
  window.TesseractViz["viz_plot"] = mountVizPlot;

  if (window.SlideEngine && typeof window.SlideEngine.registerAnimation === "function") {
    window.SlideEngine.registerAnimation("viz_plot", mountVizPlot);
  }
  document.addEventListener("DOMContentLoaded", () => {
    if (window.SlideEngine && typeof window.SlideEngine.registerAnimation === "function") {
      window.SlideEngine.registerAnimation("viz_plot", mountVizPlot);
    }
  });
})();
