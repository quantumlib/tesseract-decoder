/* ==========================================================================
   Animation Module: cascade_funnel_explorer
   Clean, Static Architectural Block Diagram for Tesseract-Hybrid:
     - Syndrome s -> Fast Pass: Multi-Pass BP-OSD -> Confidence Gate
     - Branch 1 (Green, High Confidence: ~85%-99% of shots — seeds agree & clear weight gap):
       -> Return BP-OSD Correction (~O(N) Fast Path)
     - Branch 2 (Amber, Ambiguous Shot: ~1%-15% of shots — candidate disagreement or small Δw):
       -> Fallback: Guided Tesseract A* Search -> Return Minimum-Weight Valid Correction
     - Bottom 3-Column Verified LER Reduction Callout vs. Standalone Tesseract-BP-OSD:
       * Color Code (d=9, 11): 2.0x lower LER
       * Mitten [[150, 30, 10]]: 18.0x lower LER
       * Hyperbolic (iter=109): 2.0x lower LER
   Zero moving balls, zero requestAnimationFrame loops, zero circuit dropdowns,
   and zero dense stage-percentage bar charts.
   Registered ID: "cascade_funnel_explorer"
   ========================================================================== */

(function () {
  function ensureSharedStyles() {
    if (document.getElementById("tesseract-hybrid-static-styles")) return;
    const style = document.createElement("style");
    style.id = "tesseract-hybrid-static-styles";
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
        --qai-gold: #FBBC04;
        --qai-orange: #D97706;
        --qai-green: #1E8E3E;
        --qai-red: #D93025;
        --qai-teal: #00796B;
        --qai-white: #FFFFFF;
        --qai-badge-blue: #E8F0FE;
        --qai-badge-gold: #FEF7E0;
        --qai-badge-green: #E6F4EA;
      }
      .hyb-root {
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        gap: 10px;
        font-family: "DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: var(--qai-charcoal);
        box-sizing: border-box;
      }
      .hyb-header {
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
      .hyb-header-title {
        font-family: "Space Mono", monospace;
        font-size: 0.73rem;
        font-weight: 700;
        color: var(--qai-deep-blue);
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }
      .hyb-header-badge {
        font-family: "Space Mono", monospace;
        font-size: 0.69rem;
        font-weight: 700;
        padding: 4px 9px;
        border-radius: 5px;
        background: var(--qai-badge-green);
        color: var(--qai-green);
        border: 1px solid rgba(30, 142, 62, 0.25);
      }
      .hyb-diagram-card {
        flex: 1;
        min-height: 280px;
        background: #FFFFFF;
        border: 1px solid var(--qai-hairline);
        border-radius: 10px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
      }
      .hyb-svg-wrap {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
      }
      .hyb-svg-wrap svg {
        width: 100%;
        height: auto;
        max-height: 320px;
        overflow: visible;
      }
      .hyb-note {
        margin-top: 8px;
        padding: 8px 12px;
        background: var(--qai-cream);
        border: 1px solid var(--qai-hairline);
        border-radius: 7px;
        font-size: 0.77rem;
        line-height: 1.4;
        color: var(--qai-dark-grey);
      }
      .hyb-summary-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
      }
      .hyb-summary-card {
        background: var(--qai-cream-card);
        border: 1px solid var(--qai-hairline);
        border-left: 4px solid var(--qai-deep-blue);
        border-radius: 8px;
        padding: 10px 13px;
      }
      .hyb-summary-card.green {
        border-left-color: var(--qai-green);
      }
      .hyb-summary-card.amber {
        border-left-color: var(--qai-orange);
      }
      .hyb-summary-kicker {
        font-family: "Space Mono", monospace;
        font-size: 0.66rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--qai-muted);
        margin-bottom: 3px;
      }
      .hyb-summary-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--qai-charcoal);
        margin-bottom: 2px;
      }
      .hyb-summary-sub {
        font-size: 0.72rem;
        color: var(--qai-muted);
        line-height: 1.32;
      }
    `;
    document.head.appendChild(style);
  }

  function mountCascadeFunnelExplorer(container) {
    ensureSharedStyles();

    container.innerHTML = `
      <div class="hyb-root">
        <div class="hyb-header">
          <span class="hyb-header-title">Tesseract-Hybrid: Confidence-Gated Architectural Block Diagram</span>
          <span class="hyb-header-badge">~85%&ndash;99% Fast Path &nbsp;&middot;&nbsp; 2&times;&ndash;18&times; Lower LER</span>
        </div>

        <div class="hyb-diagram-card">
          <div class="hyb-svg-wrap">
            <svg viewBox="0 0 880 260" xmlns="http://www.w3.org/2000/svg">
              <rect x="12" y="96" width="112" height="64" rx="8" fill="#F8FAFC" stroke="#334155" stroke-width="2" />
              <text x="68" y="124" text-anchor="middle" font-family="'DM Sans', sans-serif" font-size="14" font-weight="700" fill="#1E2022">Syndrome s</text>
              <text x="68" y="143" text-anchor="middle" font-family="'Space Mono', monospace" font-size="10.5" fill="#5F6368">M detectors</text>

              <line x1="124" y1="128" x2="158" y2="128" stroke="#334155" stroke-width="2.2" />
              <polygon points="162,128 153,123 153,133" fill="#334155" />

              <rect x="164" y="84" width="212" height="88" rx="9" fill="#E8F0FE" stroke="#1967D2" stroke-width="2.2" />
              <text x="270" y="109" text-anchor="middle" font-family="'Space Mono', monospace" font-size="10.5" font-weight="700" fill="#1967D2">FAST PASS (STAGE 1)</text>
              <text x="270" y="131" text-anchor="middle" font-family="'DM Sans', sans-serif" font-size="15" font-weight="700" fill="#1E2022">Multi-Pass BP-OSD</text>
              <text x="270" y="152" text-anchor="middle" font-family="'DM Sans', sans-serif" font-size="11.5" fill="#334155">Shattered BP + Multi-Seed OSD</text>

              <path d="M 376 108 L 418 108 L 418 54 L 542 54" fill="none" stroke="#1E8E3E" stroke-width="2.8" />
              <polygon points="546,54 536,48 536,60" fill="#1E8E3E" />

              <rect x="430" y="12" width="320" height="32" rx="6" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="1.2" />
              <text x="590" y="32" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11" font-weight="700" fill="#1E8E3E">
                High Confidence: ~85%&ndash;99% of shots (seeds agree, large &Delta;w)
              </text>

              <rect x="548" y="24" width="318" height="62" rx="8" fill="#E6F4EA" stroke="#1E8E3E" stroke-width="2.2" />
              <text x="707" y="51" text-anchor="middle" font-family="'DM Sans', sans-serif" font-size="14.5" font-weight="700" fill="#14532D">
                Return BP-OSD Correction
              </text>
              <text x="707" y="71" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11" font-weight="700" fill="#1E8E3E">
                ~O(N) Fast Path &middot; Commit Immediately
              </text>

              <path d="M 376 148 L 418 148 L 418 192 L 456 192" fill="none" stroke="#D97706" stroke-width="2.8" />
              <polygon points="460,192 450,186 450,198" fill="#D97706" />

              <rect x="430" y="124" width="352" height="30" rx="6" fill="#FEF7E0" stroke="#D97706" stroke-width="1.2" />
              <text x="606" y="143" text-anchor="middle" font-family="'Space Mono', monospace" font-size="11" font-weight="700" fill="#B45309">
                Ambiguous Shot: ~1%&ndash;15% of shots (disagreement or small &Delta;w)
              </text>

              <rect x="462" y="162" width="196" height="62" rx="8" fill="#FEF7E0" stroke="#D97706" stroke-width="2.2" />
              <text x="560" y="188" text-anchor="middle" font-family="'DM Sans', sans-serif" font-size="13.5" font-weight="700" fill="#92400E">
                Fallback: Guided
              </text>
              <text x="560" y="207" text-anchor="middle" font-family="'DM Sans', sans-serif" font-size="13.5" font-weight="700" fill="#92400E">
                Tesseract A* Search
              </text>

              <line x1="658" y1="192" x2="684" y2="192" stroke="#D97706" stroke-width="2.4" />
              <polygon points="688,192 679,187 679,197" fill="#D97706" />

              <rect x="690" y="162" width="176" height="62" rx="8" fill="#F8FAFC" stroke="#1967D2" stroke-width="2.2" />
              <text x="778" y="188" text-anchor="middle" font-family="'DM Sans', sans-serif" font-size="13" font-weight="700" fill="#1E2022">
                Return Min-Weight
              </text>
              <text x="778" y="207" text-anchor="middle" font-family="'DM Sans', sans-serif" font-size="13" font-weight="700" fill="#1967D2">
                Valid Correction
              </text>
            </svg>
          </div>

          <div class="hyb-note">
            <strong>How <code>Tesseract-Hybrid</code> Works:</strong> Every shot first runs the fast <strong>Multi-Pass BP-OSD</strong> engine. When the multi-seed OSD candidates agree on the logical observable with a clear soft-weight margin $\\Delta w$ (<strong>~85%&ndash;99% of shots</strong>), the correction is returned immediately in $O(N)$ time. Only ambiguous syndromes (<strong>~1%&ndash;15% of shots</strong>) fall back to <strong>Guided Tesseract $A^*$ Search</strong>, which returns the minimum-weight valid correction.
          </div>
        </div>

        <div class="hyb-summary-grid">
          <div class="hyb-summary-card green">
            <div class="hyb-summary-kicker">Color Code (d = 9, 11)</div>
            <div class="hyb-summary-val">2.0&times; Lower LER</div>
            <div class="hyb-summary-sub">Cuts logical errors from $33 \\to 16$ ($d=9$) and $12 \\to 6$ ($d=11$) vs. standalone <code>Tesseract-BP-OSD</code>.</div>
          </div>
          <div class="hyb-summary-card">
            <div class="hyb-summary-kicker">Mitten [[150, 30, 10]]</div>
            <div class="hyb-summary-val">18.0&times; Lower LER</div>
            <div class="hyb-summary-sub">Reduces logical errors from $18 \\to 1$ across $100{,}000$ shots while triggering fallback on $&lt;0.4\\%$ of shots.</div>
          </div>
          <div class="hyb-summary-card amber">
            <div class="hyb-summary-kicker">Hyperbolic (iter = 109)</div>
            <div class="hyb-summary-val">2.0&times; Lower LER</div>
            <div class="hyb-summary-sub">Halves logical errors ($22 \\to 11$) at $64{,}000$ detectors with only $1.6\\%$ fallback rate.</div>
          </div>
        </div>
      </div>
    `;

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

  window.TesseractViz = window.TesseractViz || {};
  window.TesseractViz["cascade_funnel_explorer"] = mountCascadeFunnelExplorer;

  if (window.SlideEngine && typeof window.SlideEngine.registerAnimation === "function") {
    window.SlideEngine.registerAnimation("cascade_funnel_explorer", mountCascadeFunnelExplorer);
  }
  document.addEventListener("DOMContentLoaded", () => {
    if (window.SlideEngine && typeof window.SlideEngine.registerAnimation === "function") {
      window.SlideEngine.registerAnimation("cascade_funnel_explorer", mountCascadeFunnelExplorer);
    }
  });
})();
