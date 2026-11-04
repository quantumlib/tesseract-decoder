/* ==========================================================================
   Animation Module: stim3d_shattering
   Clean, Static 2D Abstract Bipartite Tanner Graph Visualizer:
     - Error nodes = Circles (○), Check nodes = Squares (□)
     - Mode 1 ("monolithic"): (a) 4-Cycle & (b) Biclique (K_{3,4}) in Bipartite Decoding Graphs
     - Mode 2 ("gari"): Oscar's Fig. 2 — (a) 4-Cycle, (b) Biclique -> (c) GARI Transform
       (eliminating biclique 4-cycles by introducing one blue auxiliary error circle i
       and one blue auxiliary check square i)
     - Mode 3 ("shattered"): Multi-Pass Sector Shattering — Pass 0 (Z-Check Graph) ->
       Soft 75%/25% Conditional LLR Prior Update -> Pass 1 (X-Check Graph) +
       Duplicate-Symptom XOR Merging (8.6x graph compression at d=23).
   Zero continuous animation, no 3D spinning circuits, no circuit dropdowns.
   Registered ID: "stim3d_shattering"
   ========================================================================== */

(function () {
  function ensureSharedStyles() {
    if (document.getElementById("tesseract-bipartite-styles")) return;
    const style = document.createElement("style");
    style.id = "tesseract-bipartite-styles";
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
        --qai-orange: #F29900;
        --qai-green: #1E8E3E;
        --qai-red: #D93025;
        --qai-teal: #00796B;
        --qai-white: #FFFFFF;
        --qai-badge-blue: #E8F0FE;
        --qai-badge-gold: #FEF7E0;
        --qai-badge-green: #E6F4EA;
        --qai-badge-red: #FCE8E6;
      }
      .bip-root {
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        gap: 10px;
        font-family: "DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: var(--qai-charcoal);
        box-sizing: border-box;
      }
      .bip-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
        padding: 8px 12px;
        background: var(--qai-cream-card);
        border: 1px solid var(--qai-hairline);
        border-radius: 8px;
      }
      .bip-legend {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--qai-dark-grey);
      }
      .bip-legend-item {
        display: inline-flex;
        align-items: center;
        gap: 5px;
      }
      .bip-tabs {
        display: inline-flex;
        gap: 5px;
        flex-wrap: wrap;
      }
      .bip-tab-btn {
        font-family: "Space Mono", monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 5px 11px;
        border-radius: 6px;
        border: 1px solid #CBD5E1;
        background: #FFFFFF;
        color: #1E293B;
        cursor: pointer;
        transition: background 0.12s ease, color 0.12s ease, border-color 0.12s ease;
      }
      .bip-tab-btn:hover {
        border-color: var(--qai-deep-blue);
        color: var(--qai-deep-blue);
      }
      .bip-tab-btn.active {
        background: var(--qai-deep-blue);
        color: #FFFFFF;
        border-color: var(--qai-deep-blue);
      }
      .bip-canvas-card {
        flex: 1;
        min-height: 290px;
        background: #FFFFFF;
        border: 1px solid var(--qai-hairline);
        border-radius: 10px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
      }
      .bip-svg-wrap {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
      }
      .bip-svg-wrap svg {
        width: 100%;
        height: auto;
        max-height: 340px;
        overflow: visible;
      }
      .bip-caption-box {
        margin-top: 8px;
        padding: 9px 12px;
        background: var(--qai-cream);
        border: 1px solid var(--qai-hairline);
        border-radius: 7px;
        font-size: 0.78rem;
        line-height: 1.42;
        color: var(--qai-dark-grey);
      }
      .bip-metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
      }
      .bip-metric-card {
        background: var(--qai-cream-card);
        border: 1px solid var(--qai-hairline);
        border-radius: 8px;
        padding: 9px 12px;
      }
      .bip-metric-kicker {
        font-family: "Space Mono", monospace;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 3px;
      }
      .bip-metric-val {
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--qai-charcoal);
        margin-bottom: 2px;
      }
      .bip-metric-sub {
        font-size: 0.72rem;
        color: var(--qai-muted);
        line-height: 1.32;
      }
    `;
    document.head.appendChild(style);
  }

  function resolveInitialMode(props) {
    props = props || {};
    const raw = String(props.mode || props.view || props.preset || props.initialMode || "").toLowerCase();
    if (
      raw.includes("shatter") ||
      raw.includes("multipass") ||
      raw.includes("multi") ||
      raw === "3" ||
      props.shattered === true
    ) {
      return "shattered";
    }
    if (raw.includes("gari") || raw === "g" || raw === "2") {
      return "gari";
    }
    if (raw.includes("mono") || raw.includes("cycle") || raw.includes("biclique") || raw === "1") {
      return "monolithic";
    }
    return "gari";
  }

  // Helper to build SVG for (a) 4-cycle at given xOffset, yOffset, scale
  function svgFourCycle(ox, oy, titleText) {
    const cY = oy + 46;
    const sY = oy + 168;
    const cX = [ox + 68, ox + 192];
    const sX = [ox + 68, ox + 192];
    const r = 17;
    const sq = 30;

    const cBot = (x) => ({ x, y: cY + r });
    const sTop = (x) => ({ x, y: sY - sq / 2 });

    let lines = "";
    const dashed = [
      [cBot(cX[0]).x, cBot(cX[0]).y, cX[0] - 46, cY + 46],
      [cBot(cX[0]).x, cBot(cX[0]).y, cX[0] + 18, cY + 52],
      [cBot(cX[1]).x, cBot(cX[1]).y, cX[1] - 18, cY + 52],
      [cBot(cX[1]).x, cBot(cX[1]).y, cX[1] + 46, cY + 46],
      [sTop(sX[0]).x, sTop(sX[0]).y, sX[0] - 50, sY - 46],
      [sTop(sX[0]).x, sTop(sX[0]).y, sX[0] - 20, sY - 50],
      [sTop(sX[0]).x, sTop(sX[0]).y, sX[0] + 18, sY - 50],
      [sTop(sX[1]).x, sTop(sX[1]).y, sX[1] - 18, sY - 50],
      [sTop(sX[1]).x, sTop(sX[1]).y, sX[1] + 20, sY - 50],
      [sTop(sX[1]).x, sTop(sX[1]).y, sX[1] + 50, sY - 46]
    ];
    dashed.forEach(([x1, y1, x2, y2]) => {
      lines += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#475569" stroke-width="1.5" stroke-dasharray="4,4" />`;
    });

    for (let ci = 0; ci < 2; ci++) {
      for (let si = 0; si < 2; si++) {
        const p1 = cBot(cX[ci]);
        const p2 = sTop(sX[si]);
        lines += `<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" stroke="#1E2022" stroke-width="2.4" />`;
      }
    }

    let nodes = "";
    cX.forEach((x) => {
      nodes += `<circle cx="${x}" cy="${cY}" r="${r}" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />`;
    });

    sX.forEach((x) => {
      nodes += `<rect x="${x - sq / 2}" y="${sY - sq / 2}" width="${sq}" height="${sq}" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />`;
    });

    const label = `<text x="${ox + 130}" y="${oy + 216}" text-anchor="middle" font-family="Georgia, 'Times New Roman', serif" font-size="15.5" font-weight="600" fill="#1E2022">${titleText || "(a) 4-cycle"}</text>`;
    return `<g>${lines}${nodes}${label}</g>`;
  }

  // Helper to build SVG for (b) Biclique K_{3,4}
  function svgBiclique(ox, oy, titleText) {
    const cY = oy + 46;
    const sY = oy + 168;
    const cX = [ox + 54, ox + 130, ox + 206];
    const sX = [ox + 32, ox + 97, ox + 163, ox + 228];
    const r = 17;
    const sq = 30;

    let lines = "";
    for (let ci = 0; ci < 3; ci++) {
      for (let si = 0; si < 4; si++) {
        lines += `<line x1="${cX[ci]}" y1="${cY + r}" x2="${sX[si]}" y2="${sY - sq / 2}" stroke="#1E2022" stroke-width="1.8" />`;
      }
    }

    let nodes = "";
    const cLabels = [
      `<text x="${cX[0]}" y="${cY + 5}" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-size="16" fill="#1E2022">i</text>`,
      `<text x="${cX[1] - 2}" y="${cY + 4}" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-size="15" fill="#1E2022">j<tspan dy="4" font-style="normal" font-size="10.5">1</tspan></text>`,
      `<text x="${cX[2] - 2}" y="${cY + 4}" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-size="15" fill="#1E2022">j<tspan dy="4" font-style="normal" font-size="10.5">2</tspan></text>`
    ];
    cX.forEach((x, idx) => {
      nodes += `<circle cx="${x}" cy="${cY}" r="${r}" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />`;
      nodes += cLabels[idx];
    });

    sX.forEach((x) => {
      nodes += `<rect x="${x - sq / 2}" y="${sY - sq / 2}" width="${sq}" height="${sq}" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />`;
    });

    const label = `<text x="${ox + 130}" y="${oy + 216}" text-anchor="middle" font-family="Georgia, 'Times New Roman', serif" font-size="15.5" font-weight="600" fill="#1E2022">${titleText || "(b) Biclique"}</text>`;
    return `<g>${lines}${nodes}${label}</g>`;
  }

  // Helper to build SVG for (c) GARI transform (exact reproduction of Oscar's Fig. 2c)
  function svgGariTransform(ox, oy, titleText) {
    const cY = oy + 46;
    const sY = oy + 168;
    const cX = [ox + 44, ox + 108, ox + 172, ox + 240];
    const sX = [ox + 26, ox + 84, ox + 142, ox + 200, ox + 264];
    const r = 17;
    const sq = 30;

    let lines = "";
    for (let ci = 0; ci < 3; ci++) {
      lines += `<line x1="${cX[ci]}" y1="${cY + r}" x2="${sX[4]}" y2="${sY - sq / 2}" stroke="#1E2022" stroke-width="1.85" />`;
    }
    for (let si = 0; si < 5; si++) {
      lines += `<line x1="${cX[3]}" y1="${cY + r}" x2="${sX[si]}" y2="${sY - sq / 2}" stroke="#1E2022" stroke-width="1.85" />`;
    }

    let nodes = "";
    const cLabels = [
      `<text x="${cX[0]}" y="${cY + 5}" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-size="16" fill="#1E2022">i</text>`,
      `<text x="${cX[1] - 2}" y="${cY + 4}" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-size="15" fill="#1E2022">j<tspan dy="4" font-style="normal" font-size="10.5">1</tspan></text>`,
      `<text x="${cX[2] - 2}" y="${cY + 4}" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-size="15" fill="#1E2022">j<tspan dy="4" font-style="normal" font-size="10.5">2</tspan></text>`
    ];
    for (let ci = 0; ci < 3; ci++) {
      nodes += `<circle cx="${cX[ci]}" cy="${cY}" r="${r}" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />`;
      nodes += cLabels[ci];
    }
    nodes += `<circle cx="${cX[3]}" cy="${cY}" r="${r}" fill="#0018F0" stroke="#1E2022" stroke-width="2" />`;
    nodes += `<text x="${cX[3]}" y="${cY + 5}" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-weight="700" font-size="16" fill="#FFFFFF">i</text>`;

    for (let si = 0; si < 4; si++) {
      nodes += `<rect x="${sX[si] - sq / 2}" y="${sY - sq / 2}" width="${sq}" height="${sq}" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />`;
    }
    nodes += `<rect x="${sX[4] - sq / 2}" y="${sY - sq / 2}" width="${sq}" height="${sq}" fill="#0018F0" stroke="#1E2022" stroke-width="2" />`;
    nodes += `<text x="${sX[4]}" y="${sY + 5}" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-weight="700" font-size="16" fill="#FFFFFF">i</text>`;

    const label = `<text x="${ox + 145}" y="${oy + 216}" text-anchor="middle" font-family="Georgia, 'Times New Roman', serif" font-size="15.5" font-weight="600" fill="#1E2022">${titleText || "(c) GARI transform"}</text>`;
    return `<g>${lines}${nodes}${label}</g>`;
  }

  function renderMonolithicTab(canvasCard, metricsGrid) {
    canvasCard.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; flex-wrap:wrap; gap:6px;">
        <span style="font-family:'Space Mono',monospace; font-size:0.72rem; font-weight:700; color:var(--qai-deep-blue); text-transform:uppercase;">
          Bipartite Decoding Graph: Why 4-Cycles &amp; Bicliques Stall Message Passing
        </span>
        <span style="font-family:'Space Mono',monospace; font-size:0.68rem; background:var(--qai-badge-red); color:var(--qai-red); padding:3px 8px; border-radius:4px; font-weight:700;">
          MP Inference Not Exact on 4-Cycles
        </span>
      </div>
      <div class="bip-svg-wrap">
        <svg viewBox="0 0 700 235" xmlns="http://www.w3.org/2000/svg">
          <rect x="12" y="8" width="310" height="220" rx="8" fill="#FAF8F5" stroke="#E2E8F0" stroke-width="1.2" />
          ${svgFourCycle(37, 6, "(a) 4-cycle")}
          <rect x="352" y="8" width="336" height="220" rx="8" fill="#FAF8F5" stroke="#E2E8F0" stroke-width="1.2" />
          ${svgBiclique(390, 6, "(b) Biclique (K₃,₄)")}
        </svg>
      </div>
      <div class="bip-caption-box">
        <strong>(a) 4-Cycle:</strong> Error nodes (circles) and check nodes (squares) sharing two edges each form a length-4 cycle (bold); extrinsic messages circulate back in 2 steps. &nbsp;
        <strong>(b) Biclique ($K_{3,4}$):</strong> When 3 error nodes ($i, j_1, j_2$, e.g. $Y = XZ$ faults) couple to the same 4 check nodes, they create $\\binom{3}{2}\\binom{4}{2} = 18$ overlapping 4-cycles, trapping belief propagation.
      </div>
    `;

    metricsGrid.innerHTML = `
      <div class="bip-metric-card">
        <div class="bip-metric-kicker" style="color:var(--qai-red);">Problem: Short 4-Cycles</div>
        <div class="bip-metric-val">Correlated Beliefs Loop</div>
        <div class="bip-metric-sub">Messages sent from an error circle return via 2 check squares in 2 iterations, causing BP oscillation.</div>
      </div>
      <div class="bip-metric-card">
        <div class="bip-metric-kicker" style="color:var(--qai-deep-blue);">Approach 1: GARI Transform</div>
        <div class="bip-metric-val">Auxiliary Node &amp; Check</div>
        <div class="bip-metric-sub">Eliminates biclique 4-cycles via a variable change introducing 1 blue error circle &amp; 1 blue check square.</div>
      </div>
      <div class="bip-metric-card">
        <div class="bip-metric-kicker" style="color:var(--qai-green);">Approach 2: Multi-Pass Split</div>
        <div class="bip-metric-val">Pass 0 (Z) &rarr; Pass 1 (X)</div>
        <div class="bip-metric-sub">Splits $Z$ and $X$ check squares into separate bipartite graphs + 75%/25% soft LLR prior update.</div>
      </div>
    `;
  }

  function renderGariTab(canvasCard, metricsGrid) {
    canvasCard.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; flex-wrap:wrap; gap:6px;">
        <span style="font-family:'Space Mono',monospace; font-size:0.72rem; font-weight:700; color:var(--qai-deep-blue); text-transform:uppercase;">
          GARI Bipartite Transformation: Eliminating Biclique 4-Cycles via Auxiliary Error &amp; Check Nodes
        </span>
        <span style="font-family:'Space Mono',monospace; font-size:0.68rem; background:var(--qai-badge-blue); color:var(--qai-gari-blue); padding:3px 8px; border-radius:4px; font-weight:700;">
          Top Row (Circles) = Error Nodes &nbsp;&middot;&nbsp; Bottom Row (Squares) = Check Nodes
        </span>
      </div>
      <div class="bip-svg-wrap">
        <svg viewBox="0 0 880 248" xmlns="http://www.w3.org/2000/svg">
          <text x="14" y="16" font-family="'Space Mono', monospace" font-size="10.5" font-weight="700" fill="#475569">ERROR NODES (CIRCLES) &rarr;</text>
          <text x="14" y="198" font-family="'Space Mono', monospace" font-size="10.5" font-weight="700" fill="#475569">CHECK NODES (SQUARES) &rarr;</text>
          ${svgFourCycle(6, 14, "(a) 4-cycle")}
          ${svgBiclique(282, 14, "(b) Biclique")}
          <g>
            <line x1="536" y1="122" x2="574" y2="122" stroke="#0018F0" stroke-width="2.4" />
            <polygon points="574,122 565,116 565,128" fill="#0018F0" />
            <text x="555" y="109" text-anchor="middle" font-family="'Space Mono', monospace" font-size="10.5" font-weight="700" fill="#0018F0">GARI</text>
          </g>
          ${svgGariTransform(576, 14, "(c) GARI transform")}
        </svg>
      </div>
      <div class="bip-caption-box">
        <strong>(a) 4-Cycle:</strong> Error nodes (circles) and check nodes (squares) sharing two edges form a 4-cycle (bold); other incident edges are dashed. &nbsp;
        <strong>(b) Biclique ($K_{3,4}$):</strong> Error nodes $i, j_1, j_2$ connected to 4 check squares form dense 4-cycles where MP inference is not exact. &nbsp;
        <strong>(c) GARI Transform:</strong> Eliminates all 4-cycles in the biclique by introducing <strong>one new auxiliary error node (blue circle $i$)</strong> and <strong>one new auxiliary check node (blue square $i$)</strong>.
      </div>
    `;

    metricsGrid.innerHTML = `
      <div class="bip-metric-card">
        <div class="bip-metric-kicker" style="color:var(--qai-gari-blue);">Variable Change</div>
        <div class="bip-metric-val">1 Blue Error Circle + 1 Blue Check Square</div>
        <div class="bip-metric-sub">Original error nodes $i, j_1, j_2$ connect only to new blue check square $i$; new blue error circle $i$ connects to blue check $i$ and the 4 original checks.</div>
      </div>
      <div class="bip-metric-card">
        <div class="bip-metric-kicker" style="color:var(--qai-green);">Cycle Elimination</div>
        <div class="bip-metric-val">12 Biclique Edges &rarr; 8 Tree Edges</div>
        <div class="bip-metric-sub">All 4-cycles inside the biclique are eliminated; if no other cycles exist in the bipartite graph, MP inference becomes exact.</div>
      </div>
      <div class="bip-metric-card">
        <div class="bip-metric-kicker" style="color:var(--qai-orange);">Search Speedup</div>
        <div class="bip-metric-val">8&times;&ndash;36&times; Faster A* Search</div>
        <div class="bip-metric-sub">Simplifies local detector branching for beam search, though cross-basis sectors remain coupled in one graph.</div>
      </div>
    `;
  }

  function renderShatteredTab(canvasCard, metricsGrid) {
    canvasCard.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; flex-wrap:wrap; gap:6px;">
        <span style="font-family:'Space Mono',monospace; font-size:0.72rem; font-weight:700; color:var(--qai-deep-blue); text-transform:uppercase;">
          Multi-Pass Sector Shattering: Disjoint Z / X Bipartite Graphs + Soft (75% / 25%) Prior Update
        </span>
        <span style="font-family:'Space Mono',monospace; font-size:0.68rem; background:var(--qai-badge-green); color:var(--qai-green); padding:3px 8px; border-radius:4px; font-weight:700;">
          8.6&times; Fewer Variables (d=23) &nbsp;&middot;&nbsp; Zero Cross-Basis 4-Cycles
        </span>
      </div>
      <div class="bip-svg-wrap">
        <svg viewBox="0 0 880 235" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="8" width="320" height="218" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
          <text x="170" y="28" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11.5" font-weight="700" fill="#B45309">
            PASS 0: Z-CHECK BIPARTITE GRAPH (D₀)
          </text>

          <line x1="65" y1="79" x2="65" y2="155" stroke="#1E2022" stroke-width="2" />
          <line x1="65" y1="79" x2="135" y2="155" stroke="#1E2022" stroke-width="2" />
          <line x1="170" y1="79" x2="135" y2="155" stroke="#1E2022" stroke-width="2" />
          <line x1="170" y1="79" x2="205" y2="155" stroke="#1E2022" stroke-width="2" />
          <line x1="265" y1="79" x2="205" y2="155" stroke="#1E2022" stroke-width="2" />
          <line x1="265" y1="79" x2="275" y2="155" stroke="#1E2022" stroke-width="2" />

          <circle cx="65" cy="62" r="17" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
          <text x="65" y="66" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-size="14" fill="#1E2022">e<tspan dy="3" font-style="normal" font-size="10">Z</tspan></text>

          <circle cx="170" cy="62" r="17" fill="#FEF3C7" stroke="#D97706" stroke-width="2.2" />
          <text x="170" y="66" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-weight="700" font-size="14" fill="#92400E">e<tspan dy="3" font-style="normal" font-size="10">Y</tspan></text>

          <circle cx="265" cy="62" r="17" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
          <text x="265" y="66" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-size="13.5" fill="#1E2022">e<tspan dy="3" font-style="normal" font-size="9.5">mrg</tspan></text>

          <rect x="50" y="155" width="30" height="30" fill="#FFFBEB" stroke="#B45309" stroke-width="2" />
          <text x="65" y="175" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11" font-weight="700" fill="#B45309">Z₁</text>
          <rect x="120" y="155" width="30" height="30" fill="#FFFBEB" stroke="#B45309" stroke-width="2" />
          <text x="135" y="175" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11" font-weight="700" fill="#B45309">Z₂</text>
          <rect x="190" y="155" width="30" height="30" fill="#FFFBEB" stroke="#B45309" stroke-width="2" />
          <text x="205" y="175" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11" font-weight="700" fill="#B45309">Z₃</text>
          <rect x="260" y="155" width="30" height="30" fill="#FFFBEB" stroke="#B45309" stroke-width="2" />
          <text x="275" y="175" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11" font-weight="700" fill="#B45309">Z₄</text>

          <text x="170" y="211" text-anchor="middle" font-family="'DM Sans', sans-serif" font-size="12.5" font-weight="600" fill="#334155">
            Duplicate symptoms XOR-merged (8.6&times; fewer cols)
          </text>

          <g>
            <rect x="346" y="42" width="188" height="126" rx="8" fill="#E8F0FE" stroke="#1967D2" stroke-width="1.6" />
            <text x="440" y="64" text-anchor="middle" font-family="'Space Mono', monospace" font-size="10.5" font-weight="700" fill="#1967D2">
              SOFT PRIOR UPDATE
            </text>
            <text x="440" y="84" text-anchor="middle" font-family="'DM Sans', sans-serif" font-size="12" font-weight="700" fill="#1E2022">
              75% Conditional + 25% Prior
            </text>
            <text x="440" y="104" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-size="12" fill="#1E293B">
              &Lambda;₁ &larr; 0.25 &Lambda;₀ + 0.75 &Lambda;(X | Z&#770;)
            </text>
            <line x1="364" y1="122" x2="512" y2="122" stroke="#1967D2" stroke-width="2.5" />
            <polygon points="516,122 506,116 506,128" fill="#1967D2" />
            <text x="440" y="152" text-anchor="middle" font-family="'DM Sans', sans-serif" font-size="11" font-weight="600" fill="#1E8E3E">
              2.4&times; lower LER vs. monolithic
            </text>
          </g>

          <path d="M 187 56 Q 340 16, 703 56" fill="none" stroke="#1967D2" stroke-width="1.8" stroke-dasharray="5,4" />

          <rect x="550" y="8" width="320" height="218" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
          <text x="710" y="28" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11.5" font-weight="700" fill="#1967D2">
            PASS 1: X-CHECK BIPARTITE GRAPH (D₁)
          </text>

          <line x1="605" y1="79" x2="605" y2="155" stroke="#1E2022" stroke-width="2" />
          <line x1="605" y1="79" x2="675" y2="155" stroke="#1E2022" stroke-width="2" />
          <line x1="710" y1="79" x2="675" y2="155" stroke="#1967D2" stroke-width="2.4" />
          <line x1="710" y1="79" x2="745" y2="155" stroke="#1967D2" stroke-width="2.4" />
          <line x1="805" y1="79" x2="745" y2="155" stroke="#1E2022" stroke-width="2" />
          <line x1="805" y1="79" x2="815" y2="155" stroke="#1E2022" stroke-width="2" />

          <circle cx="605" cy="62" r="17" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
          <text x="605" y="66" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-size="14" fill="#1E2022">e<tspan dy="3" font-style="normal" font-size="10">X</tspan></text>

          <circle cx="710" cy="62" r="17" fill="#0018F0" stroke="#1E2022" stroke-width="2" />
          <text x="710" y="66" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-weight="700" font-size="14" fill="#FFFFFF">e<tspan dy="3" font-style="normal" font-size="10">Y</tspan></text>

          <circle cx="805" cy="62" r="17" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
          <text x="805" y="66" text-anchor="middle" font-family="Georgia, serif" font-style="italic" font-size="13.5" fill="#1E2022">e<tspan dy="3" font-style="normal" font-size="9.5">mrg</tspan></text>

          <rect x="590" y="155" width="30" height="30" fill="#EFF6FF" stroke="#1967D2" stroke-width="2" />
          <text x="605" y="175" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11" font-weight="700" fill="#1967D2">X₁</text>
          <rect x="660" y="155" width="30" height="30" fill="#EFF6FF" stroke="#1967D2" stroke-width="2" />
          <text x="675" y="175" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11" font-weight="700" fill="#1967D2">X₂</text>
          <rect x="730" y="155" width="30" height="30" fill="#EFF6FF" stroke="#1967D2" stroke-width="2" />
          <text x="745" y="175" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11" font-weight="700" fill="#1967D2">X₃</text>
          <rect x="800" y="155" width="30" height="30" fill="#EFF6FF" stroke="#1967D2" stroke-width="2" />
          <text x="815" y="175" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11" font-weight="700" fill="#1967D2">X₄</text>

          <text x="710" y="211" text-anchor="middle" font-family="'DM Sans', sans-serif" font-size="12.5" font-weight="600" fill="#334155">
            Decoded with soft-updated priors on e_Y (blue circle)
          </text>
        </svg>
      </div>
      <div class="bip-caption-box">
        <strong>Multi-Pass Sector Shattering:</strong> Partitioning check squares into <strong>Pass 0 ($Z$-checks)</strong> and <strong>Pass 1 ($X$-checks)</strong> severs all cross-basis $X$–$Z$ 4-cycles with zero fill-in. Parallel error circles with identical single-basis symptoms merge via exact XOR convolution (<strong>8.6&times; fewer variables</strong> at $d=23$), and Pass 0 updates shared $Y$-fault priors in Pass 1 via a <strong>75%/25% soft LLR blend</strong>.
      </div>
    `;

    metricsGrid.innerHTML = `
      <div class="bip-metric-card">
        <div class="bip-metric-kicker" style="color:var(--qai-deep-blue);">Graph Compression (d=23)</div>
        <div class="bip-metric-val">8.6&times; Fewer Error Nodes</div>
        <div class="bip-metric-sub">Merges duplicate single-basis columns from $434{,}705 \\to 50{,}259$ variables per pass (` + "`4.2x`" + ` faster BP).</div>
      </div>
      <div class="bip-metric-card">
        <div class="bip-metric-kicker" style="color:var(--qai-green);">Cross-Basis 4-Cycles</div>
        <div class="bip-metric-val">Severed Structurally</div>
        <div class="bip-metric-sub">Disjoint $Z$ and $X$ bipartite graphs prevent $Y$-fault beliefs from looping between bases.</div>
      </div>
      <div class="bip-metric-card">
        <div class="bip-metric-kicker" style="color:var(--qai-orange);">Soft 75%/25% Reweighting</div>
        <div class="bip-metric-val">2.4&times; Lower LER (Color d=9)</div>
        <div class="bip-metric-sub">Reduces LER/round from $1.70\\times 10^{-4}$ (monolithic) to $7.00\\times 10^{-5}$ while outperforming hard 100% updates.</div>
      </div>
    `;
  }

  function mountStim3dShattering(container, props) {
    ensureSharedStyles();
    const currentMode = resolveInitialMode(props);

    container.innerHTML = `
      <div class="bip-root">
        <div class="bip-canvas-card" id="bipCanvasCard"></div>
        <div class="bip-metrics-grid" id="bipMetricsGrid"></div>
      </div>
    `;

    const canvasCard = container.querySelector("#bipCanvasCard");
    const metricsGrid = container.querySelector("#bipMetricsGrid");

    if (currentMode === "monolithic") {
      renderMonolithicTab(canvasCard, metricsGrid);
    } else if (currentMode === "shattered") {
      renderShatteredTab(canvasCard, metricsGrid);
    } else {
      renderGariTab(canvasCard, metricsGrid);
    }

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

    return {
      cleanup() {}
    };
  }

  function mountGariBipartite(container, props) {
    return mountStim3dShattering(container, Object.assign({}, props || {}, { mode: "gari" }));
  }

  function mountMultipassBipartite(container, props) {
    return mountStim3dShattering(container, Object.assign({}, props || {}, { mode: "shattered" }));
  }

  window.TesseractViz = window.TesseractViz || {};
  window.TesseractViz["stim3d_shattering"] = mountStim3dShattering;
  window.TesseractViz["gari_bipartite"] = mountGariBipartite;
  window.TesseractViz["multipass_bipartite"] = mountMultipassBipartite;

  function registerAll() {
    if (window.SlideEngine && typeof window.SlideEngine.registerAnimation === "function") {
      window.SlideEngine.registerAnimation("stim3d_shattering", mountStim3dShattering);
      window.SlideEngine.registerAnimation("gari_bipartite", mountGariBipartite);
      window.SlideEngine.registerAnimation("multipass_bipartite", mountMultipassBipartite);
    }
  }
  registerAll();
  document.addEventListener("DOMContentLoaded", registerAll);
})();
