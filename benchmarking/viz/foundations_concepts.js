/* ==========================================================================
   Animation Module: foundations_concepts
   Clean, Static 2D Pedagogical Diagrams for Foundational Concepts:
     - "dem_h_concept" (Slide 3): Detector Error Model (DEM), Binary Parity-Check
       Matrix H (H e = s mod 2), and Bipartite Tanner Graph Correspondence
     - "llr_concept" (Slide 4): Log-Likelihood Ratio (LLR) Scale & Why LLR Turns
       Syndrome Decoding into Additive Cost Minimization
     - "bp_concept" (Slide 5): How Min-Sum BP Works (Check MIN Rule + Error SUM
       Rule) & Why It Converges on a Tree vs. Oscillates on a 4-Cycle
     - "osd_concept" (Slide 9): 3-Step Ordered Statistics Decoding (OSD-0 & OSD-w)
       Pipeline (Confidence Ranking -> RREF Pivot Basis -> Exact Syndrome Solve)
   Zero continuous animation, zero toggle buttons, zero checkbox legends.
   Registered IDs: "dem_h_concept", "llr_concept", "bp_concept", "osd_concept", "foundations_concepts"
   ========================================================================== */

(function () {
  function ensureSharedStyles() {
    if (document.getElementById("tesseract-foundations-concepts-styles")) return;
    const style = document.createElement("style");
    style.id = "tesseract-foundations-concepts-styles";
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
      .fc-root {
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        gap: 10px;
        font-family: "DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: var(--qai-charcoal);
        box-sizing: border-box;
      }
      .fc-card {
        flex: 1;
        background: #FFFFFF;
        border: 1px solid var(--qai-hairline);
        border-radius: 10px;
        padding: 12px 15px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
      }
      .fc-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 6px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--qai-hairline);
      }
      .fc-card-title {
        font-family: "Space Mono", monospace;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }
      .fc-badge {
        font-family: "Space Mono", monospace;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
      }
      .fc-svg-wrap {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
      }
      .fc-svg-wrap svg {
        width: 100%;
        height: auto;
        overflow: visible;
      }
      .fc-metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
      }
      .fc-metric-card {
        background: var(--qai-cream-card);
        border: 1px solid var(--qai-hairline);
        border-radius: 8px;
        padding: 9px 12px;
      }
      .fc-metric-kicker {
        font-family: "Space Mono", monospace;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 3px;
      }
      .fc-metric-val {
        font-size: 0.88rem;
        font-weight: 700;
        color: var(--qai-charcoal);
        margin-bottom: 2px;
      }
      .fc-metric-sub {
        font-size: 0.72rem;
        color: var(--qai-muted);
        line-height: 1.34;
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

  /* ==========================================================================
     0. DEM, PARITY MATRIX H & BIPARTITE GRAPH CONCEPT ("dem_h_concept")
     ========================================================================== */
  function renderDemHConcept(container) {
    container.innerHTML = `
      <div class="fc-root">
        <!-- Top Card: 3-Way Equivalence (DEM <-> Matrices H & O <-> Bipartite Graph) -->
        <div class="fc-card">
          <div class="fc-card-header">
            <span class="fc-card-title" style="color:var(--qai-deep-blue);">
              1. Three Equivalent Views: Detector Error Model (DEM) &harr; Matrices H &amp; O &harr; Bipartite Graph
            </span>
            <span class="fc-badge" style="background:var(--qai-badge-blue); color:var(--qai-deep-blue);">
              H = Detector Parity Matrix &nbsp;&middot;&nbsp; O = Logical Observable Matrix
            </span>
          </div>
          <div class="fc-svg-wrap">
            <svg viewBox="0 0 860 206" style="max-height:212px;" xmlns="http://www.w3.org/2000/svg">
              <!-- PANEL 1: Stim Detector Error Model (DEM) -->
              <rect x="6" y="4" width="268" height="198" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="140" y="23" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1967D2">
                A. DETECTOR ERROR MODEL (DEM)
              </text>
              <text x="140" y="39" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#475569">
                Lists fault probability p_v, detectors D_c &amp; logicals L_k
              </text>

              <!-- Fault e1 -->
              <rect x="16" y="48" width="248" height="38" rx="5" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2" />
              <text x="24" y="64" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1E2022">
                e₁: error(0.01) D0
              </text>
              <text x="24" y="79" font-family="'DM Sans',sans-serif" font-size="9.5" fill="#5F6368">
                Flips detector D₀ only (prob p₁ = 1%)
              </text>

              <!-- Fault e2 (Highlighted active example) -->
              <rect x="16" y="94" width="248" height="42" rx="5" fill="#FEF2F2" stroke="#D93025" stroke-width="1.8" />
              <text x="24" y="111" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#D93025">
                e₂: error(0.001) D0 D1 L0
              </text>
              <text x="24" y="128" font-family="'DM Sans',sans-serif" font-size="9.5" font-weight="600" fill="#991B1B">
                Flips D₀ &amp; D₁ + logical L₀ (prob p₂ = 0.1%)
              </text>

              <!-- Fault e3 -->
              <rect x="16" y="144" width="248" height="38" rx="5" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2" />
              <text x="24" y="160" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1E2022">
                e₃: error(0.001) D1 D2
              </text>
              <text x="24" y="175" font-family="'DM Sans',sans-serif" font-size="9.5" fill="#5F6368">
                Flips detectors D₁ &amp; D₂ (prob p₃ = 0.1%)
              </text>

              <!-- Arrow A -> B -->
              <line x1="276" y1="103" x2="292" y2="103" stroke="#1967D2" stroke-width="2.2" />
              <polygon points="295,103 287,98 287,108" fill="#1967D2" />

              <!-- PANEL 2: Binary Parity Matrix H & Logical Matrix O -->
              <rect x="297" y="4" width="266" height="198" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="430" y="23" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1967D2">
                B. PARITY MATRIX H &amp; LOGICAL MATRIX O
              </text>
              <text x="430" y="39" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#475569">
                Columns = errors (e₁,e₂,e₃); H = checks, O = logicals
              </text>

              <!-- Column Headers -->
              <text x="404" y="58" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="13" fill="#1E2022">e₁</text>
              <text x="454" y="58" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="13" fill="#D93025">e₂</text>
              <text x="504" y="58" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="13" fill="#1E2022">e₃</text>

              <!-- Highlight column e2 -->
              <rect x="433" y="64" width="42" height="118" rx="4" fill="#FEE2E2" stroke="#D93025" stroke-width="1.5" />

              <!-- H Matrix Label & Brackets -->
              <text x="322" y="110" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="14" fill="#1967D2">H =</text>
              <path d="M 384,65 L 376,65 L 376,147 L 384,147" fill="none" stroke="#1E2022" stroke-width="2" />
              <path d="M 524,65 L 532,65 L 532,147 L 524,147" fill="none" stroke="#1E2022" stroke-width="2" />

              <!-- Row Labels & Entries -->
              <!-- Row D0 -->
              <text x="366" y="84" text-anchor="end" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#D93025">D₀</text>
              <text x="404" y="84" text-anchor="middle" font-family="'Space Mono',monospace" font-size="13" font-weight="700" fill="#1967D2">1</text>
              <text x="454" y="84" text-anchor="middle" font-family="'Space Mono',monospace" font-size="13" font-weight="700" fill="#D93025">1</text>
              <text x="504" y="84" text-anchor="middle" font-family="'Space Mono',monospace" font-size="13" fill="#94A3B8">0</text>

              <!-- Row D1 -->
              <text x="366" y="112" text-anchor="end" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#D93025">D₁</text>
              <text x="404" y="112" text-anchor="middle" font-family="'Space Mono',monospace" font-size="13" fill="#94A3B8">0</text>
              <text x="454" y="112" text-anchor="middle" font-family="'Space Mono',monospace" font-size="13" font-weight="700" fill="#D93025">1</text>
              <text x="504" y="112" text-anchor="middle" font-family="'Space Mono',monospace" font-size="13" font-weight="700" fill="#1967D2">1</text>

              <!-- Row D2 -->
              <text x="366" y="140" text-anchor="end" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1E2022">D₂</text>
              <text x="404" y="140" text-anchor="middle" font-family="'Space Mono',monospace" font-size="13" fill="#94A3B8">0</text>
              <text x="454" y="140" text-anchor="middle" font-family="'Space Mono',monospace" font-size="13" fill="#94A3B8">0</text>
              <text x="504" y="140" text-anchor="middle" font-family="'Space Mono',monospace" font-size="13" font-weight="700" fill="#1967D2">1</text>

              <!-- Logical Row O -->
              <line x1="312" y1="154" x2="546" y2="154" stroke="#CBD5E1" stroke-width="1.2" stroke-dasharray="4,3" />
              <text x="322" y="175" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="14" fill="#B45309">O =</text>
              <path d="M 384,160 L 376,160 L 376,182 L 384,182" fill="none" stroke="#B45309" stroke-width="1.8" />
              <path d="M 524,160 L 532,160 L 532,182 L 524,182" fill="none" stroke="#B45309" stroke-width="1.8" />
              <text x="366" y="175" text-anchor="end" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#B45309">L₀</text>
              <text x="404" y="175" text-anchor="middle" font-family="'Space Mono',monospace" font-size="12.5" fill="#94A3B8">0</text>
              <text x="454" y="175" text-anchor="middle" font-family="'Space Mono',monospace" font-size="12.5" font-weight="700" fill="#B45309">1</text>
              <text x="504" y="175" text-anchor="middle" font-family="'Space Mono',monospace" font-size="12.5" fill="#94A3B8">0</text>
              <text x="430" y="195" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.3" font-weight="600" fill="#475569">
                H_(c,v)=1 iff e_v flips D_c &nbsp;&middot;&nbsp; O_(k,v)=1 iff e_v flips L_k
              </text>

              <!-- Arrow B -> C -->
              <line x1="565" y1="103" x2="581" y2="103" stroke="#1967D2" stroke-width="2.2" />
              <polygon points="584,103 576,98 576,108" fill="#1967D2" />

              <!-- PANEL 3: Bipartite Tanner Graph -->
              <rect x="586" y="4" width="268" height="198" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="720" y="23" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1967D2">
                C. BIPARTITE TANNER GRAPH
              </text>
              <text x="720" y="39" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#475569">
                Circles = Error Columns, Squares = Check Rows
              </text>

              <!-- Bipartite Edges (1 per non-zero in H) -->
              <!-- e1 -> D0 -->
              <line x1="644" y1="84" x2="644" y2="142" stroke="#1967D2" stroke-width="2.2" />
              <!-- e2 -> D0, D1 (Highlighted active fault e2=1) -->
              <line x1="720" y1="84" x2="644" y2="142" stroke="#D93025" stroke-width="2.8" />
              <line x1="720" y1="84" x2="720" y2="142" stroke="#D93025" stroke-width="2.8" />
              <!-- e3 -> D1, D2 -->
              <line x1="796" y1="84" x2="720" y2="142" stroke="#1967D2" stroke-width="2.2" />
              <line x1="796" y1="84" x2="796" y2="142" stroke="#1967D2" stroke-width="2.2" />

              <!-- Error Circles e1, e2, e3 -->
              <circle cx="644" cy="68" r="16" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
              <text x="644" y="73" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="14" fill="#1E2022">e₁</text>

              <circle cx="720" cy="68" r="16" fill="#FEF2F2" stroke="#D93025" stroke-width="2.4" />
              <text x="720" y="73" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="14" fill="#D93025">e₂=1</text>

              <circle cx="796" cy="68" r="16" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
              <text x="796" y="73" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="14" fill="#1E2022">e₃</text>

              <!-- Check Squares D0, D1, D2 -->
              <rect x="629" y="142" width="30" height="28" fill="#D93025" stroke="#1E2022" stroke-width="1.8" />
              <text x="644" y="160" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#FFFFFF">D₀=1</text>

              <rect x="705" y="142" width="30" height="28" fill="#D93025" stroke="#1E2022" stroke-width="1.8" />
              <text x="720" y="160" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#FFFFFF">D₁=1</text>

              <rect x="781" y="142" width="30" height="28" fill="#FFFFFF" stroke="#1E2022" stroke-width="1.8" />
              <text x="796" y="160" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1E2022">D₂=0</text>

              <text x="720" y="190" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.5" font-weight="600" fill="#991B1B">
                5 ones in H &harr; 5 edges in bipartite graph
              </text>
            </svg>
          </div>
        </div>

        <!-- Bottom Card: How Syndrome Decoding Works (He = s mod 2) -->
        <div class="fc-card" style="flex:0.75;">
          <div class="fc-card-header">
            <span class="fc-card-title" style="color:var(--qai-green);">
              2. The Syndrome Decoding Problem: Given Fired Detectors s, Recover Likely Errors e&#770;
            </span>
            <span class="fc-badge" style="background:var(--qai-badge-green); color:var(--qai-green);">
              Parity Constraint: H e &equiv; s (mod 2) &nbsp;&middot;&nbsp; Logical Flip Prediction: l&#770; = O e&#770; (mod 2)
            </span>
          </div>
          <div class="fc-metrics-grid">
            <div class="fc-metric-card">
              <div class="fc-metric-kicker" style="color:var(--qai-deep-blue);">Forward Syndrome (Nature / Monte Carlo)</div>
              <div class="fc-metric-val">Active Faults Pick Columns of H</div>
              <div class="fc-metric-sub">When nature (or a Monte Carlo simulation shot) triggers a random error vector $e$ weighted by the DEM probabilities ($e = [0,1,0]^T$), the circuit outputs $s = H e \\bmod 2 = [1,1,0]^T$.</div>
            </div>
            <div class="fc-metric-card">
              <div class="fc-metric-kicker" style="color:var(--qai-red);">Inverse Problem (Per-Shot Decoder)</div>
              <div class="fc-metric-val">Only Syndrome s Is Observed</div>
              <div class="fc-metric-sub">For each shot, the decoder sees only the fired check squares $s = [1,1,0]^T$ ($M$ equations) and must infer the most likely binary error vector $\\hat{e} \\in \\mathbb{F}_2^N$ ($N > M$ unknowns).</div>
            </div>
            <div class="fc-metric-card">
              <div class="fc-metric-kicker" style="color:var(--qai-green);">Logical Matrix O (Observable Flip)</div>
              <div class="fc-metric-val">Predict Logical Flip l&#770; = O e&#770; mod 2</div>
              <div class="fc-metric-sub">The binary logical matrix $O$ records which errors flip each logical observable $L_k$; decoding succeeds whenever $O\\hat{e} \\equiv O e \\pmod 2$.</div>
            </div>
          </div>
        </div>
      </div>
    `;
    renderMath(container);
  }

  /* ==========================================================================
     1. LLR CONCEPT ("llr_concept")
     ========================================================================== */
  function renderLlrConcept(container) {
    container.innerHTML = `
      <div class="fc-root">
        <!-- Top Card: The LLR Scale -->
        <div class="fc-card">
          <div class="fc-card-header">
            <span class="fc-card-title" style="color:var(--qai-deep-blue);">
              1. The LLR Scale: Sign = Hard Decision (0 vs 1), Magnitude = Confidence / Flip Cost
            </span>
            <span class="fc-badge" style="background:var(--qai-badge-blue); color:var(--qai-deep-blue);">
              &Lambda;_v^(0) = ln((1 &minus; p_v) / p_v)
            </span>
          </div>
          <div class="fc-svg-wrap">
            <svg viewBox="0 0 860 158" style="max-height:162px;" xmlns="http://www.w3.org/2000/svg">
              <!-- Left Zone: Negative LLR (Error e_v = 1) -->
              <rect x="16" y="10" width="350" height="54" rx="7" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="1.3" />
              <text x="191" y="30" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#D93025">
                NEGATIVE SCORE (L_v &lt; 0 &rArr; e&#770;_v = 1: FLIP TO ERROR)
              </text>
              <text x="191" y="49" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="11" font-weight="600" fill="#991B1B">
                Fired check votes pushed total score below zero
              </text>

              <!-- Center Boundary: L_v = 0 -->
              <rect x="374" y="10" width="112" height="54" rx="7" fill="#FEF7E0" stroke="#F59E0B" stroke-width="1.5" />
              <text x="430" y="29" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#B45309">
                BOUNDARY L_v = 0
              </text>
              <text x="430" y="48" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10.5" font-weight="700" fill="#92400E">
                p = 0.5 (50/50 Tie)
              </text>

              <!-- Right Zone: Positive LLR (No Error e_v = 0) -->
              <rect x="494" y="10" width="350" height="54" rx="7" fill="#E6F4EA" stroke="#86EFAC" stroke-width="1.3" />
              <text x="669" y="30" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#1E8E3E">
                POSITIVE SCORE (L_v &gt; 0 &rArr; e&#770;_v = 0: NO ERROR)
              </text>
              <text x="669" y="49" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="11" font-weight="600" fill="#14532D">
                All physical errors start here (since p_v &ll; 0.5)
              </text>

              <!-- Number Line Axis -->
              <line x1="36" y1="98" x2="824" y2="98" stroke="#1E2022" stroke-width="2.6" />
              <polygon points="28,98 40,92 40,104" fill="#D93025" />
              <polygon points="832,98 820,92 820,104" fill="#1E8E3E" />

              <!-- Tick Marks & Concrete Values -->
              <line x1="86" y1="90" x2="86" y2="106" stroke="#D93025" stroke-width="2.2" />
              <circle cx="86" cy="98" r="5" fill="#D93025" />
              <text x="86" y="122" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#D93025">-6.9</text>
              <text x="86" y="138" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#5F6368">p = 0.999</text>

              <line x1="200" y1="90" x2="200" y2="106" stroke="#D93025" stroke-width="2.2" />
              <circle cx="200" cy="98" r="5" fill="#D93025" />
              <text x="200" y="122" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#D93025">-4.6</text>
              <text x="200" y="138" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#5F6368">p = 0.99</text>

              <line x1="315" y1="90" x2="315" y2="106" stroke="#D93025" stroke-width="2.2" />
              <circle cx="315" cy="98" r="5" fill="#D93025" />
              <text x="315" y="122" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#D93025">-2.2</text>
              <text x="315" y="138" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#5F6368">p = 0.90</text>

              <line x1="430" y1="84" x2="430" y2="112" stroke="#D97706" stroke-width="3" />
              <circle cx="430" cy="98" r="6" fill="#F59E0B" stroke="#1E2022" stroke-width="1.5" />
              <text x="430" y="124" text-anchor="middle" font-family="'Space Mono',monospace" font-size="11" font-weight="700" fill="#B45309">0.0</text>
              <text x="430" y="140" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" font-weight="700" fill="#B45309">p = 0.50</text>

              <line x1="545" y1="90" x2="545" y2="106" stroke="#1967D2" stroke-width="2.2" />
              <circle cx="545" cy="98" r="5" fill="#1967D2" />
              <text x="545" y="122" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#1967D2">+2.2</text>
              <text x="545" y="138" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#5F6368">p = 0.10</text>

              <line x1="660" y1="90" x2="660" y2="106" stroke="#1E8E3E" stroke-width="2.2" />
              <circle cx="660" cy="98" r="5" fill="#1E8E3E" />
              <text x="660" y="122" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#1E8E3E">+4.6</text>
              <text x="660" y="138" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" font-weight="600" fill="#14532D">p = 0.01</text>

              <line x1="774" y1="90" x2="774" y2="106" stroke="#1E8E3E" stroke-width="2.2" />
              <circle cx="774" cy="98" r="5" fill="#1E8E3E" />
              <text x="774" y="122" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#1E8E3E">+6.9</text>
              <text x="774" y="138" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" font-weight="600" fill="#14532D">p = 0.001</text>
            </svg>
          </div>
        </div>

        <!-- Bottom Card: Why LLR Turns Syndrome Decoding into Additive Cost Minimization -->
        <div class="fc-card">
          <div class="fc-card-header">
            <span class="fc-card-title" style="color:var(--qai-green);">
              2. Why LLR Turns Syndrome Decoding into Additive Cost Minimization
            </span>
            <span class="fc-badge" style="background:var(--qai-badge-green); color:var(--qai-green);">
              Multiplying Probabilities &rarr; Adding LLR Flip Costs: min &Sigma; &Lambda;_v e_v s.t. H e &equiv; s (mod 2)
            </span>
          </div>
          <div class="fc-svg-wrap">
            <svg viewBox="0 0 860 190" style="max-height:192px;" xmlns="http://www.w3.org/2000/svg">
              <!-- Left Panel: Bipartite Toy Graph -->
              <rect x="12" y="6" width="405" height="178" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="214" y="25" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#1967D2">
                BIPARTITE TOY GRAPH: FIRED CHECK s₁ = 1, SILENT s₂ = 0
              </text>

              <!-- Edges -->
              <line x1="92" y1="78" x2="145" y2="136" stroke="#1E8E3E" stroke-width="2.8" />
              <line x1="214" y1="78" x2="145" y2="136" stroke="#1E2022" stroke-width="2" />
              <line x1="214" y1="78" x2="295" y2="136" stroke="#1E2022" stroke-width="2" />
              <line x1="336" y1="78" x2="295" y2="136" stroke="#1E2022" stroke-width="2" />

              <!-- Error Node e1 (p=0.01, LLR=+4.6) -->
              <circle cx="92" cy="60" r="18" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="2.5" />
              <text x="92" y="65" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="15" fill="#14532D">e₁</text>
              <text x="92" y="38" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#1E8E3E">Cost +4.6 (p=10⁻²)</text>

              <!-- Error Node e2 (p=0.001, LLR=+6.9) -->
              <circle cx="214" cy="60" r="18" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
              <text x="214" y="65" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="15" fill="#1E2022">e₂</text>
              <text x="214" y="38" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#475569">Cost +6.9 (p=10⁻³)</text>

              <!-- Error Node e3 (p=0.001, LLR=+6.9) -->
              <circle cx="336" cy="60" r="18" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
              <text x="336" y="65" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="15" fill="#1E2022">e₃</text>
              <text x="336" y="38" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#475569">Cost +6.9 (p=10⁻³)</text>

              <!-- Check Square s1 = 1 (Fired, Red) -->
              <rect x="129" y="134" width="32" height="32" fill="#D93025" stroke="#1E2022" stroke-width="2" />
              <text x="145" y="155" text-anchor="middle" font-family="'Space Mono',monospace" font-size="11.5" font-weight="700" fill="#FFFFFF">s₁=1</text>

              <!-- Check Square s2 = 0 (Silent, White) -->
              <rect x="279" y="134" width="32" height="32" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
              <text x="295" y="155" text-anchor="middle" font-family="'Space Mono',monospace" font-size="11.5" font-weight="700" fill="#1E2022">s₂=0</text>

              <text x="214" y="178" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="11" font-weight="600" fill="#334155">
                Parity constraints: e₁ &oplus; e₂ = 1 (s₁ fired) &nbsp;and&nbsp; e₂ &oplus; e₃ = 0 (s₂ silent)
              </text>

              <!-- Right Panel: Comparing Valid Explanations -->
              <rect x="433" y="6" width="415" height="178" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="640" y="25" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#1E2022">
                COMPARING VALID SYNDROME EXPLANATIONS (H e &equiv; s mod 2)
              </text>

              <!-- Candidate A: Winner -->
              <rect x="447" y="36" width="387" height="64" rx="6" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="1.8" />
              <text x="459" y="55" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#14532D">
                CANDIDATE A: Flip e = (1, 0, 0) &mdash; Single Fault on e₁
              </text>
              <text x="459" y="74" font-family="'DM Sans',sans-serif" font-size="11.5" font-weight="600" fill="#1E2022">
                Total LLR Cost: 4.6(1) + 6.9(0) + 6.9(0) = <tspan font-weight="700" fill="#1E8E3E">4.6</tspan>
              </text>
              <text x="459" y="91" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1E8E3E">
                &check; MINIMUM-COST MAP WINNER (Satisfies s₁=1, s₂=0)
              </text>

              <!-- Candidate B: Rejected -->
              <rect x="447" y="108" width="387" height="64" rx="6" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="1.4" />
              <text x="459" y="127" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#991B1B">
                CANDIDATE B: Flip e = (0, 1, 1) &mdash; Two-Fault Chain on (e₂, e₃)
              </text>
              <text x="459" y="146" font-family="'DM Sans',sans-serif" font-size="11.5" font-weight="600" fill="#1E2022">
                Total LLR Cost: 4.6(0) + 6.9(1) + 6.9(1) = <tspan font-weight="700" fill="#D93025">13.8</tspan>
              </text>
              <text x="459" y="163" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#D93025">
                &cross; REJECTED (3.0&times; higher LLR cost; ~10⁴&times; less likely)
              </text>
            </svg>
          </div>
        </div>
      </div>
    `;
    renderMath(container);
  }

  /* ==========================================================================
     2. BP CONCEPT ("bp_concept")
     ========================================================================== */
  function renderBpConcept(container) {
    container.innerHTML = `
      <div class="fc-root">
        <!-- Top Card (Green Header): The Two BP Rules (MIN at Checks, SUM at Errors) on a Tree -->
        <div class="fc-card">
          <div class="fc-card-header">
            <span class="fc-card-title" style="color:var(--qai-green);">
              1. How Min-Sum BP Works on a Tree: Check MIN Rule + Error SUM Rule &rarr; 1-Pass Convergence
            </span>
            <span class="fc-badge" style="background:var(--qai-badge-green); color:var(--qai-green);">
              True Fault e₂ = 1 &nbsp;&middot;&nbsp; Starting Priors (+6, +6, +6) &rarr; Converges to e&#770; = (0, 1, 0) &check;
            </span>
          </div>
          <div class="fc-svg-wrap">
            <svg viewBox="0 0 860 176" style="max-height:180px;" xmlns="http://www.w3.org/2000/svg">
              <!-- Panel A: Check-to-Error MIN Rule -->
              <rect x="8" y="4" width="420" height="168" rx="8" fill="#F0FDF4" stroke="#86EFAC" stroke-width="1.4" />
              <text x="218" y="22" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#14532D">
                STEP A &mdash; CHECK SQUARE &rarr; ERROR CIRCLE ("MIN" WEAKEST-LINK RULE)
              </text>
              <text x="218" y="37" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10.5" font-weight="600" fill="#1E2022">
                Fired check c₁=1 asks: "How hard is it to flip my other neighbor e₁?"
              </text>

              <!-- Tree Edges in Panel A -->
              <line x1="52" y1="126" x2="112" y2="74" stroke="#1E8E3E" stroke-width="2.2" />
              <line x1="164" y1="126" x2="112" y2="74" stroke="#94A3B8" stroke-width="1.6" stroke-dasharray="4,3" />
              <line x1="164" y1="126" x2="218" y2="74" stroke="#D93025" stroke-width="2.8" />
              <line x1="272" y1="126" x2="218" y2="74" stroke="#D93025" stroke-width="2.8" />
              <line x1="272" y1="126" x2="324" y2="74" stroke="#94A3B8" stroke-width="1.6" stroke-dasharray="4,3" />
              <line x1="384" y1="126" x2="324" y2="74" stroke="#1E8E3E" stroke-width="2.2" />

              <!-- Message Callouts on Edges -->
              <text x="64" y="96" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#1E8E3E">+6</text>
              <text x="182" y="102" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#D93025">-6</text>
              <text x="254" y="102" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#D93025">-6</text>
              <text x="372" y="96" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#1E8E3E">+6</text>

              <!-- Error Circles e1, e2, e3 -->
              <circle cx="112" cy="60" r="16" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
              <text x="112" y="64" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="13.5" fill="#1E2022">e₁</text>
              <text x="112" y="40" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9" font-weight="700" fill="#1E8E3E">prior +6</text>

              <circle cx="218" cy="60" r="16" fill="#FEF2F2" stroke="#D93025" stroke-width="2.2" />
              <text x="218" y="64" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="13.5" fill="#D93025">e₂</text>
              <text x="218" y="40" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9" font-weight="700" fill="#D93025">prior +6</text>

              <circle cx="324" cy="60" r="16" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
              <text x="324" y="64" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="13.5" fill="#1E2022">e₃</text>
              <text x="324" y="40" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9" font-weight="700" fill="#1E8E3E">prior +6</text>

              <!-- Check Squares c0=0, c1=1, c2=1, c3=0 -->
              <rect x="38" y="122" width="28" height="24" fill="#FFFFFF" stroke="#1E2022" stroke-width="1.8" />
              <text x="52" y="138" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#1E2022">c₀=0</text>

              <rect x="150" y="122" width="28" height="24" fill="#D93025" stroke="#1E2022" stroke-width="1.8" />
              <text x="164" y="138" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#FFFFFF">c₁=1</text>

              <rect x="258" y="122" width="28" height="24" fill="#D93025" stroke="#1E2022" stroke-width="1.8" />
              <text x="272" y="138" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#FFFFFF">c₂=1</text>

              <rect x="370" y="122" width="28" height="24" fill="#FFFFFF" stroke="#1E2022" stroke-width="1.8" />
              <text x="384" y="138" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#1E2022">c₃=0</text>

              <text x="218" y="162" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10.2" font-weight="600" fill="#14532D">
                Silent c₀,c₃=0 send +6 ("stay 0"); fired c₁,c₂=1 send &minus;6 ("flip to 1!") into e₂
              </text>

              <!-- Panel B: Error Circle SUM Rule -->
              <rect x="436" y="4" width="416" height="168" rx="8" fill="#F0FDF4" stroke="#86EFAC" stroke-width="1.4" />
              <text x="644" y="22" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#14532D">
                STEP B &mdash; ERROR CIRCLE SCORE ("SUM" PRIOR + CHECK VOTES)
              </text>
              <text x="644" y="37" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10.5" font-weight="600" fill="#1E2022">
                Each error circle adds its starting prior (+6) and incoming check hints:
              </text>

              <!-- 3 Variable Summary Boxes -->
              <rect x="448" y="48" width="122" height="90" rx="6" fill="#FFFFFF" stroke="#86EFAC" stroke-width="1.5" />
              <text x="509" y="66" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1E8E3E">ERROR e₁</text>
              <text x="509" y="86" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#1E2022">+6 + 6 &minus; 6 = +6</text>
              <text x="509" y="106" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" font-weight="700" fill="#1E8E3E">Score &gt; 0 &rArr; e&#770;₁ = 0</text>
              <text x="509" y="124" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.5" fill="#5F6368">c₀=0 keeps e₁ quiet</text>

              <rect x="578" y="48" width="132" height="90" rx="6" fill="#FEF2F2" stroke="#D93025" stroke-width="2" />
              <text x="644" y="66" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#D93025">TRUE FAULT e₂</text>
              <text x="644" y="86" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#D93025">+6 &minus; 6 &minus; 6 = &minus;6</text>
              <text x="644" y="106" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" font-weight="700" fill="#D93025">Score &lt; 0 &rArr; e&#770;₂ = 1</text>
              <text x="644" y="124" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.5" font-weight="700" fill="#991B1B">Two &minus;6 votes flip e₂!</text>

              <rect x="718" y="48" width="122" height="90" rx="6" fill="#FFFFFF" stroke="#86EFAC" stroke-width="1.5" />
              <text x="779" y="66" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1E8E3E">ERROR e₃</text>
              <text x="779" y="86" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#1E2022">+6 + 6 &minus; 6 = +6</text>
              <text x="779" y="106" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" font-weight="700" fill="#1E8E3E">Score &gt; 0 &rArr; e&#770;₃ = 0</text>
              <text x="779" y="124" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.5" fill="#5F6368">c₃=0 keeps e₃ quiet</text>

              <text x="644" y="160" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1E8E3E">
                &check; EXACT TREE CONVERGENCE IN 1 PASS: e&#770; = (0, 1, 0) SATISFIES ALL CHECKS!
              </text>
            </svg>
          </div>
        </div>

        <!-- Bottom Card (Red Header): 4-Cycle Oscillation -->
        <div class="fc-card">
          <div class="fc-card-header">
            <span class="fc-card-title" style="color:var(--qai-red);">
              2. Why a Symmetric 4-Cycle Breaks Parallel BP: Both Errors Flip Together &amp; Oscillate
            </span>
            <span class="fc-badge" style="background:var(--qai-badge-red); color:var(--qai-red);">
              Symmetric Tie: Either e=(1,0) or e=(0,1) Fixes c₁=1, c₂=1 &rarr; Flooding Bounces (0,0) &harr; (1,1)
            </span>
          </div>
          <div class="fc-svg-wrap">
            <svg viewBox="0 0 860 172" style="max-height:176px;" xmlns="http://www.w3.org/2000/svg">
              <!-- Panel 1: 4-Cycle Graph -->
              <rect x="8" y="4" width="236" height="164" rx="8" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="1.4" />
              <text x="126" y="22" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.8" font-weight="700" fill="#D93025">
                SYMMETRIC 4-CYCLE GRAPH
              </text>

              <!-- 4-Cycle Edges -->
              <line x1="76" y1="74" x2="76" y2="120" stroke="#D93025" stroke-width="2.4" />
              <line x1="76" y1="74" x2="176" y2="120" stroke="#D93025" stroke-width="2.4" />
              <line x1="176" y1="74" x2="76" y2="120" stroke="#D93025" stroke-width="2.4" />
              <line x1="176" y1="74" x2="176" y2="120" stroke="#D93025" stroke-width="2.4" />

              <circle cx="76" cy="58" r="16" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
              <text x="76" y="62" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="14" fill="#1E2022">e₁</text>
              <text x="76" y="38" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9" font-weight="700" fill="#475569">prior +6</text>

              <circle cx="176" cy="58" r="16" fill="#FFFFFF" stroke="#1E2022" stroke-width="2" />
              <text x="176" y="62" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="14" fill="#1E2022">e₂</text>
              <text x="176" y="38" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9" font-weight="700" fill="#475569">prior +6</text>

              <rect x="61" y="118" width="30" height="24" fill="#D93025" stroke="#1E2022" stroke-width="1.8" />
              <text x="76" y="134" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#FFFFFF">c₁=1</text>

              <rect x="161" y="118" width="30" height="24" fill="#D93025" stroke="#1E2022" stroke-width="1.8" />
              <text x="176" y="134" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#FFFFFF">c₂=1</text>

              <text x="126" y="158" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" font-weight="600" fill="#991B1B">
                e₁ and e₂ share both fired checks
              </text>

              <!-- Panel 2: Pass 1 (Double Flip Overshoot) -->
              <rect x="256" y="4" width="292" height="164" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="402" y="22" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.8" font-weight="700" fill="#D93025">
                PASS 1: BOTH FLIP TO (1, 1) AT ONCE
              </text>
              <rect x="268" y="32" width="268" height="48" rx="5" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="1.2" />
              <text x="402" y="50" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10.5" font-weight="600" fill="#1E2022">
                Both checks c₁=1, c₂=1 send &minus;6 to both errors:
              </text>
              <text x="402" y="68" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#D93025">
                MIN Rule &rArr; Hint = &minus;6 on all 4 edges
              </text>

              <rect x="268" y="88" width="268" height="48" rx="5" fill="#FEF2F2" stroke="#D93025" stroke-width="1.5" />
              <text x="402" y="106" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#991B1B">
                SUM Rule: Score(e₁) = Score(e₂) = +6 &minus; 6 &minus; 6 = &minus;6
              </text>
              <text x="402" y="125" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#D93025">
                &rArr; e&#770;^(1) = (1, 1) &nbsp;[Parity 1 &oplus; 1 = 0 &ne; 1!]
              </text>
              <text x="402" y="156" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" font-weight="700" fill="#991B1B">
                Flipping both cancels out on the checks!
              </text>

              <!-- Panel 3: Pass 2 (Rebound to Zero) -->
              <rect x="560" y="4" width="292" height="164" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="706" y="22" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.8" font-weight="700" fill="#B45309">
                PASS 2: BOTH REBOUND TO (0, 0)
              </text>
              <rect x="572" y="32" width="268" height="48" rx="5" fill="#FEF7E0" stroke="#F59E0B" stroke-width="1.2" />
              <text x="706" y="50" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10.5" font-weight="600" fill="#1E2022">
                Seeing the other error flipped, checks reverse sign:
              </text>
              <text x="706" y="68" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#B45309">
                MIN Rule &rArr; Hint flips to +6 ("flip back to 0!")
              </text>

              <rect x="572" y="88" width="268" height="48" rx="5" fill="#E8F0FE" stroke="#1967D2" stroke-width="1.5" />
              <text x="706" y="106" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1967D2">
                SUM Rule: Score(e₁) = Score(e₂) = +6 + 6 + 6 = +18
              </text>
              <text x="706" y="125" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10.5" font-weight="700" fill="#1967D2">
                &rArr; e&#770;^(2) = (0, 0) &nbsp;[Parity 0 &oplus; 0 = 0 &ne; 1!]
              </text>
              <text x="706" y="156" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#D93025">
                &cross; Bounces (0,0) &harr; (1,1) forever!
              </text>
            </svg>
          </div>
        </div>
      </div>
    `;
    renderMath(container);
  }

  /* ==========================================================================
     3. OSD CONCEPT ("osd_concept")
     ========================================================================== */
  function renderOsdConcept(container) {
    container.innerHTML = `
      <div class="fc-root">
        <!-- Top Card: 3-Step OSD Pipeline -->
        <div class="fc-card">
          <div class="fc-card-header">
            <span class="fc-card-title" style="color:var(--qai-deep-blue);">
              3-Step OSD Pipeline: Turning Unconverged Soft BP Scores into an Exact Valid Correction
            </span>
            <span class="fc-badge" style="background:var(--qai-badge-green); color:var(--qai-green);">
              Guarantees H e &equiv; s (mod 2) via Algebraic Pivot Solve
            </span>
          </div>

          <div class="fc-svg-wrap">
            <svg viewBox="0 0 860 228" style="max-height:234px;" xmlns="http://www.w3.org/2000/svg">
              <!-- STEP 1: Rank by Confidence -->
              <rect x="8" y="6" width="266" height="216" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="141" y="25" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1967D2">
                STEP 1: RANK BY BP CONFIDENCE |L_v|
              </text>
              <text x="141" y="42" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10.5" fill="#334155">
                Sort columns from least confident to most confident
              </text>

              <!-- Sorted Error Circles -->
              <rect x="20" y="54" width="114" height="78" rx="6" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="1.2" />
              <text x="77" y="69" text-anchor="middle" font-family="'Space Mono',monospace" font-size="8.8" font-weight="700" fill="#D93025">
                UNCERTAIN (|L|&approx;0)
              </text>
              <circle cx="50" cy="94" r="14" fill="#FEE2E2" stroke="#D93025" stroke-width="2" />
              <text x="50" y="98" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#991B1B">0.1</text>
              <circle cx="98" cy="94" r="14" fill="#FEE2E2" stroke="#D93025" stroke-width="2" />
              <text x="98" y="98" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#991B1B">0.3</text>
              <text x="77" y="123" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.5" font-weight="600" fill="#991B1B">
                Trapped 4-cycle vars
              </text>

              <rect x="144" y="54" width="118" height="78" rx="6" fill="#E6F4EA" stroke="#86EFAC" stroke-width="1.2" />
              <text x="203" y="69" text-anchor="middle" font-family="'Space Mono',monospace" font-size="8.8" font-weight="700" fill="#1E8E3E">
                RELIABLE (|L|&gg;0)
              </text>
              <circle cx="174" cy="94" r="14" fill="#FFFFFF" stroke="#1E8E3E" stroke-width="2" />
              <text x="174" y="98" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#14532D">14</text>
              <circle cx="226" cy="94" r="14" fill="#FFFFFF" stroke="#1E8E3E" stroke-width="2" />
              <text x="226" y="98" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#14532D">18</text>
              <text x="203" y="123" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.5" font-weight="600" fill="#14532D">
                Confident tree vars
              </text>

              <rect x="20" y="142" width="242" height="68" rx="6" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.2" />
              <text x="141" y="161" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10.5" font-weight="700" fill="#1E2022">
                Move uncertain columns to the left:
              </text>
              <text x="141" y="179" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#475569">
                Variables trapped in 4-cycles have score near 0,
              </text>
              <text x="141" y="195" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" fill="#475569">
                so row reduction picks them as pivots first!
              </text>

              <!-- Arrow 1 -> 2 -->
              <line x1="276" y1="114" x2="292" y2="114" stroke="#1967D2" stroke-width="2.2" />
              <polygon points="295,114 287,109 287,119" fill="#1967D2" />

              <!-- STEP 2: Left-to-Right RREF Pivot Selection -->
              <rect x="297" y="6" width="266" height="216" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="430" y="25" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1967D2">
                STEP 2: SELECT PIVOT BASIS (RREF)
              </text>
              <text x="430" y="42" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="11" fill="#334155">
                H&#771; = [ I_r  |  H&#771;_(B&#772;) ] &nbsp;(rank r)
              </text>

              <!-- Matrix Partition Visual -->
              <rect x="313" y="54" width="106" height="92" rx="5" fill="#FEF2F2" stroke="#D93025" stroke-width="1.8" />
              <line x1="323" y1="64" x2="409" y2="136" stroke="#D93025" stroke-width="2.4" />
              <text x="388" y="80" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#991B1B">I_r</text>
              <text x="342" y="128" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9" font-weight="700" fill="#991B1B">Pivots B</text>

              <rect x="425" y="54" width="122" height="92" rx="5" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="1.8" />
              <text x="486" y="88" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-weight="700" font-size="13" fill="#14532D">H&#771;_(B&#772;)</text>
              <text x="486" y="110" text-anchor="middle" font-family="'Space Mono',monospace" font-size="9" font-weight="700" fill="#1E8E3E">Non-Pivots B&#772;</text>
              <text x="486" y="128" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.5" fill="#14532D">(Reliable bits)</text>

              <rect x="313" y="154" width="234" height="56" rx="6" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.2" />
              <text x="430" y="173" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="10" font-weight="700" fill="#1E2022">
                Gauss&ndash;Jordan elimination over GF(2):
              </text>
              <text x="430" y="191" text-anchor="middle" font-family="'DM Sans',sans-serif" font-size="9.8" fill="#475569">
                Picks r independent uncertain columns as basis B.
              </text>

              <!-- Arrow 2 -> 3 -->
              <line x1="565" y1="114" x2="581" y2="114" stroke="#1967D2" stroke-width="2.2" />
              <polygon points="584,114 576,109 576,119" fill="#1967D2" />

              <!-- STEP 3: Exact Syndrome Solve (OSD-0 & OSD-w) -->
              <rect x="586" y="6" width="266" height="216" rx="8" fill="#FAF8F5" stroke="#CBD5E1" stroke-width="1.4" />
              <text x="719" y="25" text-anchor="middle" font-family="'Space Mono',monospace" font-size="10" font-weight="700" fill="#1E8E3E">
                STEP 3: SOLVE OSD-0 &amp; SEARCH OSD-w
              </text>
              <text x="719" y="42" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="11" fill="#334155">
                e_B = s&#771; &oplus; H&#771;_(B&#772;) e_(B&#772;)
              </text>

              <!-- OSD-0 Box -->
              <rect x="598" y="54" width="242" height="72" rx="6" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="1.6" />
              <text x="610" y="72" font-family="'Space Mono',monospace" font-size="9.8" font-weight="700" fill="#14532D">
                OSD-0 (ZERO-ORDER SOLVE):
              </text>
              <text x="610" y="90" font-family="'DM Sans',sans-serif" font-size="10.2" font-weight="600" fill="#1E2022">
                Fix reliable non-pivots e_(B&#772;) to BP's hard bits
              </text>
              <text x="610" y="108" font-family="'Space Mono',monospace" font-size="9.5" font-weight="700" fill="#1E8E3E">
                &rArr; Uniquely solves pivots e_B so H e &equiv; s!
              </text>

              <!-- OSD-w Box -->
              <rect x="598" y="136" width="242" height="74" rx="6" fill="#E8F0FE" stroke="#1967D2" stroke-width="1.5" />
              <text x="610" y="154" font-family="'Space Mono',monospace" font-size="9.8" font-weight="700" fill="#1967D2">
                HIGHER-ORDER OSD-w (w &in; {1, 2}):
              </text>
              <text x="610" y="172" font-family="'DM Sans',sans-serif" font-size="10" font-weight="600" fill="#1E2022">
                Test small flips among borderline non-pivots,
              </text>
              <text x="610" y="188" font-family="'DM Sans',sans-serif" font-size="10" font-weight="600" fill="#1E2022">
                recompute e_B, and keep the lowest-cost solution!
              </text>
            </svg>
          </div>
        </div>

        <!-- Bottom 3-Metric Strip -->
        <div class="fc-metrics-grid">
          <div class="fc-metric-card">
            <div class="fc-metric-kicker" style="color:var(--qai-red);">Why Soft Scores Help When BP Oscillates</div>
            <div class="fc-metric-val">4-Cycle Variables Have Score Near 0</div>
            <div class="fc-metric-sub">Even when BP fails to converge, oscillating variables hover near $L_v \\approx 0$, so ranking by $|L_v|$ automatically puts them into the pivot basis $B$ to be solved algebraically.</div>
          </div>
          <div class="fc-metric-card">
            <div class="fc-metric-kicker" style="color:var(--qai-green);">OSD-0 Baseline</div>
            <div class="fc-metric-val">Breaks 4-Cycle Ties in One Solve</div>
            <div class="fc-metric-sub">Trusts the reliable non-pivot bits $\\bar{B}$ and solves the uncertain pivot equations $e_B = \\tilde{s} \\oplus \\tilde{H}_{\\bar{B}} e_{\\bar{B}}$, guaranteeing a valid syndrome match $H e \\equiv s \\pmod 2$.</div>
          </div>
          <div class="fc-metric-card">
            <div class="fc-metric-kicker" style="color:var(--qai-deep-blue);">Higher-Order OSD-w</div>
            <div class="fc-metric-val">Tests Borderline Non-Pivot Flips</div>
            <div class="fc-metric-sub">If a true fault landed just outside the pivot basis $B$, flipping 1 or 2 borderline non-pivot bits and updating $e_B$ via fast bit-packed XOR finds the lower-cost codeword.</div>
          </div>
        </div>
      </div>
    `;
    renderMath(container);
  }

  function mountDemHConcept(container) {
    ensureSharedStyles();
    renderDemHConcept(container);
    return { cleanup() {} };
  }

  function mountLlrConcept(container) {
    ensureSharedStyles();
    renderLlrConcept(container);
    return { cleanup() {} };
  }

  function mountBpConcept(container) {
    ensureSharedStyles();
    renderBpConcept(container);
    return { cleanup() {} };
  }

  function mountOsdConcept(container) {
    ensureSharedStyles();
    renderOsdConcept(container);
    return { cleanup() {} };
  }

  function mountFoundationsConcepts(container, props) {
    props = props || {};
    const raw = String(props.preset || props.mode || props.view || props.id || "").toLowerCase();
    if (raw.includes("dem")) {
      return mountDemHConcept(container, props);
    }
    if (raw.includes("bp")) {
      return mountBpConcept(container, props);
    }
    if (raw.includes("osd")) {
      return mountOsdConcept(container, props);
    }
    return mountLlrConcept(container, props);
  }

  window.TesseractViz = window.TesseractViz || {};
  window.TesseractViz["dem_h_concept"] = mountDemHConcept;
  window.TesseractViz["llr_concept"] = mountLlrConcept;
  window.TesseractViz["bp_concept"] = mountBpConcept;
  window.TesseractViz["osd_concept"] = mountOsdConcept;
  window.TesseractViz["foundations_concepts"] = mountFoundationsConcepts;

  function registerAll() {
    if (window.SlideEngine && typeof window.SlideEngine.registerAnimation === "function") {
      window.SlideEngine.registerAnimation("dem_h_concept", mountDemHConcept);
      window.SlideEngine.registerAnimation("llr_concept", mountLlrConcept);
      window.SlideEngine.registerAnimation("bp_concept", mountBpConcept);
      window.SlideEngine.registerAnimation("osd_concept", mountOsdConcept);
      window.SlideEngine.registerAnimation("foundations_concepts", mountFoundationsConcepts);
    }
  }
  registerAll();
  document.addEventListener("DOMContentLoaded", registerAll);
})();
