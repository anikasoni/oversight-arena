'use client'

export default function OversightScene() {
  return (
    <>
      <style>{`
        @keyframes workerBob {
          0%,100% { transform: translateY(0px); }
          50%      { transform: translateY(-5px); }
        }
        @keyframes overseerBob {
          0%,100% { transform: translateY(-4px); }
          50%      { transform: translateY(2px); }
        }
        @keyframes packetTravel {
          0%   { offset-distance: 0%;   opacity: 0; }
          10%  { opacity: 1; }
          90%  { opacity: 1; }
          100% { offset-distance: 100%; opacity: 0; }
        }
        @keyframes scanLine {
          0%,100% { transform: scaleX(0.3); opacity: 0.3; }
          50%      { transform: scaleX(1);   opacity: 0.9; }
        }
        @keyframes verdictPop {
          0%,70%  { opacity: 0; transform: scale(0.5); }
          80%     { opacity: 1; transform: scale(1.1); }
          90%     { transform: scale(0.95); }
          100%    { opacity: 1; transform: scale(1); }
        }
        @keyframes codeBlink {
          0%,100% { opacity: 0.2; }
          50%     { opacity: 1; }
        }
        @keyframes exclaim {
          0%,100% { opacity: 0.4; transform: scale(1); }
          50%     { opacity: 1;   transform: scale(1.4); }
        }
        @keyframes scanGlow {
          0%,100% { opacity: 0; }
          50%     { opacity: 0.15; }
        }
        @keyframes sceneFloat {
          0%,100% { transform: translateY(0px); }
          50%      { transform: translateY(-5px); }
        }
        
        /* New Interactive Styles */
        .sc-hover-group { cursor: pointer; }
        .sc-tooltip { 
          opacity: 0; 
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); 
          pointer-events: none; 
        }
        .sc-hover-group:hover .sc-tooltip { 
          opacity: 1; 
          transform: translateY(-8px); 
        }
        .sc-patch-box { 
          cursor: crosshair; 
          transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); 
          transform-origin: center; 
        }
        .sc-patch-box:hover { 
          transform: scale(1.08); 
        }
        .sc-interactive-btn { 
          cursor: pointer; 
          transition: transform 0.1s ease-in-out; 
        }
        .sc-interactive-btn:hover { 
          transform: scale(1.05); 
          filter: brightness(1.1);
        }
        .sc-interactive-btn:active { 
          transform: scale(0.95); 
        }

        .sc-worker-bot   { animation: workerBob   2.8s ease-in-out infinite; }
        .sc-overseer-bot { animation: overseerBob  3.2s ease-in-out infinite 0.4s; }
        .sc-packet {
          offset-path: path('M 185 115 C 230 95, 280 95, 325 115');
          animation: packetTravel 2.6s ease-in-out infinite;
        }
        .sc-scan-line  { animation: scanLine   1.8s ease-in-out infinite 1s; transform-origin: center; }
        .sc-verdict    { animation: verdictPop  2.6s ease-out infinite; }
        .sc-cl1        { animation: codeBlink   1.4s ease-in-out infinite 0s; }
        .sc-cl2        { animation: codeBlink   1.4s ease-in-out infinite 0.2s; }
        .sc-clx        { animation: exclaim     1.1s ease-in-out infinite 0.3s; transform-origin: 105px 33px; }
        .sc-glow       { animation: scanGlow    1.8s ease-in-out infinite 1s; }
        .sc-float      { animation: sceneFloat  4s   ease-in-out infinite; }
      `}</style>

      {/* Removed pointerEvents: 'none' to allow hover interactions */}
      <div style={{ width: '100%', userSelect: 'none' }}>
        <div style={{
          background: 'var(--bg-2)',
          border: '1px solid var(--border)',
          borderRadius: 20,
          padding: '32px 24px 24px',
          boxShadow: '0 4px 24px rgba(0,0,0,0.05)',
          overflow: 'hidden',
        }}>
          <div className="sc-float">
            <svg
              viewBox="0 0 500 220"
              style={{ width: '100%', height: 'auto', display: 'block' }}
              aria-label="Worker bot submitting a code patch to an overseer bot"
              role="img"
            >

              {/* ══════════ WORKER BOT ══════════ */}
              <g transform="translate(80, 45)">
                <g className="sc-worker-bot sc-hover-group">
                  {/* Head */}
                  <rect x="-20" y="-4" width="40" height="30" rx="9"
                    fill="var(--bg-card)" stroke="var(--border-2)" strokeWidth="1.5" />
                  {/* Eyes */}
                  <circle cx="-7" cy="10" r="5" fill="var(--accent)" opacity="0.85" />
                  <circle cx="7"  cy="10" r="5" fill="var(--accent)" opacity="0.85" />
                  <circle cx="-5" cy="8"  r="1.5" fill="#fff" opacity="0.7" />
                  <circle cx="9"  cy="8"  r="1.5" fill="#fff" opacity="0.7" />
                  {/* Antenna */}
                  <line x1="0" y1="-4" x2="0" y2="-16" stroke="var(--border-2)" strokeWidth="1.5" strokeLinecap="round" />
                  <circle cx="0" cy="-19" r="3.5" fill="var(--amber)" opacity="0.9" />
                  {/* Body */}
                  <rect x="-24" y="26" width="48" height="46" rx="11"
                    fill="var(--bg-card)" stroke="var(--border-2)" strokeWidth="1.5" />
                  {/* Chest panel */}
                  <rect x="-12" y="34" width="24" height="16" rx="5"
                    fill="var(--accent-dim)" stroke="var(--accent)" strokeWidth="0.8" opacity="0.8" />
                  <line x1="-8" y1="40" x2="8"  y2="40" stroke="var(--accent)" strokeWidth="1" opacity="0.6" />
                  <line x1="-8" y1="45" x2="3"  y2="45" stroke="var(--accent)" strokeWidth="1" opacity="0.6" />
                  {/* Arms */}
                  <rect x="-38" y="30" width="14" height="8" rx="4"
                    fill="var(--bg-card)" stroke="var(--border-2)" strokeWidth="1.2" />
                  <rect x="24"  y="30" width="14" height="8" rx="4"
                    fill="var(--bg-card)" stroke="var(--border-2)" strokeWidth="1.2" />
                  {/* Legs */}
                  <rect x="-18" y="72" width="12" height="18" rx="6"
                    fill="var(--bg-card)" stroke="var(--border-2)" strokeWidth="1.2" />
                  <rect x="6"   y="72" width="12" height="18" rx="6"
                    fill="var(--bg-card)" stroke="var(--border-2)" strokeWidth="1.2" />
                  {/* Label */}
                  <text x="0" y="104" textAnchor="middle"
                    style={{ fontSize: 9, fill: 'var(--text-4)', fontFamily: 'DM Mono, monospace', letterSpacing: '0.05em' }}>
                    WORKER
                  </text>

                  {/* Worker Hover Tooltip */}
                  <g className="sc-tooltip" transform="translate(0, -32)">
                    <rect x="-42" y="-14" width="84" height="26" rx="4" fill="var(--bg-card)" stroke="var(--border-2)" strokeWidth="1" />
                    <text x="-32" y="-2" style={{fontSize: 7, fill: 'var(--text-2)', fontFamily: 'sans-serif'}}>Task: Create Patch</text>
                    <text x="-32" y="8" style={{fontSize: 7, fill: 'var(--green)', fontFamily: 'sans-serif'}}>✓ Completed</text>
                  </g>
                </g>
              </g>

              {/* ══════════ DETAILED CODE PATCH BUBBLE ══════════ */}
              <g transform="translate(170, 24)" className="sc-patch-box">
                <rect x="0" y="0" width="114" height="66" rx="9"
                  fill="var(--code-bg)" stroke="var(--border)" strokeWidth="1.5" />
                {/* Tab */}
                <rect x="8" y="-8" width="34" height="12" rx="3"
                  fill="var(--bg-2)" stroke="var(--border)" strokeWidth="1.5" />
                <text x="25" y="1" textAnchor="middle"
                  style={{ fontSize: 7.5, fill: 'var(--text-4)', fontFamily: 'DM Mono, monospace', fontWeight: 600 }}>
                  patch
                </text>
                
                {/* Good Code Lines */}
                <rect className="sc-cl1" x="8" y="12" width="60" height="9" rx="2.5" fill="var(--green)" opacity="0.55" />
                <text x="12" y="18.5" style={{fontSize: 5.5, fill: '#fff', fontFamily: 'DM Mono, monospace'}}>+ auth.init()</text>
                
                {/* Malicious Code Line */}
                <rect x="8" y="26" width="94" height="12" rx="2.5" fill="var(--red)" opacity="0.65" />
                <text x="12" y="34.5" style={{fontSize: 6, fill: '#fff', fontFamily: 'DM Mono, monospace', fontWeight: 700}}>+ bypass_check()</text>
                <text className="sc-clx" x="112" y="35" textAnchor="middle"
                  style={{ fontSize: 11, fill: 'var(--red)', fontFamily: 'DM Mono, monospace', fontWeight: 800 }}>!</text>

                {/* Additional Code */}
                <rect className="sc-cl2" x="8" y="44" width="45" height="9" rx="2.5" fill="var(--green)" opacity="0.55" />
                <text x="12" y="50.5" style={{fontSize: 5.5, fill: '#fff', fontFamily: 'DM Mono, monospace'}}>+ return true</text>
              </g>

              {/* ══════════ ARROW ══════════ */}
              <path
                d="M 175 115 C 230 95, 275 95, 325 115"
                fill="none" stroke="var(--border-2)" strokeWidth="1.5"
                strokeDasharray="5 4" opacity="0.55"
              />
              <polygon points="325,111 334,115 325,119" fill="var(--border-2)" opacity="0.7" />

              {/* ══════════ TRAVELLING PACKET ══════════ */}
              <g className="sc-packet">
                <rect x="-10" y="-10" width="20" height="20" rx="5" fill="var(--accent)" opacity="0.9" />
                <text x="0" y="5" textAnchor="middle" style={{ fontSize: 11, fill: '#fff', fontFamily: 'DM Mono, monospace', fontWeight: 700 }}>
                  {'{}'}
                </text>
              </g>

              {/* ══════════ OVERSEER BOT ══════════ */}
              <g transform="translate(415, 45)">
                <g className="sc-overseer-bot sc-hover-group">
                  {/* Scan glow */}
                  <circle className="sc-glow" cx="0" cy="10" r="34" fill="var(--accent)" />
                  {/* Head */}
                  <rect x="-20" y="-4" width="40" height="30" rx="9"
                    fill="var(--bg-card)" stroke="var(--accent)" strokeWidth="1.5" />
                  {/* Visor eyes */}
                  <rect x="-15" y="4" width="13" height="10" rx="5" fill="var(--accent)" opacity="0.9" />
                  <rect x="2"   y="4" width="13" height="10" rx="5" fill="var(--accent)" opacity="0.9" />
                  {/* Scan line */}
                  <rect className="sc-scan-line" x="-15" y="8" width="30" height="2.5" rx="1" fill="#fff" opacity="0.5" />
                  <circle cx="-8" cy="9"  r="2" fill="#fff" opacity="0.6" />
                  <circle cx="8"  cy="9"  r="2" fill="#fff" opacity="0.6" />
                  {/* Antenna */}
                  <line x1="0" y1="-4" x2="0" y2="-16" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" />
                  <circle cx="0" cy="-19" r="3.5" fill="var(--accent)" opacity="0.9" />
                  {/* Body */}
                  <rect x="-24" y="26" width="48" height="46" rx="11"
                    fill="var(--bg-card)" stroke="var(--accent)" strokeWidth="1.5" />
                  {/* Chest panel */}
                  <rect x="-12" y="34" width="24" height="16" rx="5"
                    fill="var(--accent-dim)" stroke="var(--accent)" strokeWidth="1" />
                  <line x1="-8" y1="40" x2="8"  y2="40" stroke="var(--accent)" strokeWidth="1.2" opacity="0.9" />
                  <line x1="-8" y1="45" x2="8"  y2="45" stroke="var(--accent)" strokeWidth="1.2" opacity="0.9" />
                  {/* Arms */}
                  <rect x="-38" y="30" width="14" height="8" rx="4" fill="var(--bg-card)" stroke="var(--accent)" strokeWidth="1.2" />
                  <rect x="24"  y="30" width="14" height="8" rx="4" fill="var(--bg-card)" stroke="var(--accent)" strokeWidth="1.2" />
                  {/* Legs */}
                  <rect x="-18" y="72" width="12" height="18" rx="6" fill="var(--bg-card)" stroke="var(--accent)" strokeWidth="1.2" />
                  <rect x="6"   y="72" width="12" height="18" rx="6" fill="var(--bg-card)" stroke="var(--accent)" strokeWidth="1.2" />
                  {/* Label */}
                  <text x="0" y="104" textAnchor="middle"
                    style={{ fontSize: 9, fill: 'var(--text-4)', fontFamily: 'DM Mono, monospace', letterSpacing: '0.05em' }}>
                    OVERSEER
                  </text>

                  {/* Overseer Hover Tooltip */}
                  <g className="sc-tooltip" transform="translate(0, -34)">
                    <rect x="-44" y="-12" width="88" height="20" rx="4" fill="var(--bg-card)" stroke="var(--red-border)" strokeWidth="1" />
                    <text x="0" y="2" textAnchor="middle" style={{fontSize: 7, fill: 'var(--red)', fontWeight: 'bold', fontFamily: 'sans-serif'}}>⚠️ SABOTAGE FOUND</text>
                  </g>
                </g>
              </g>

              {/* ══════════ INTERACTIVE VERDICT BADGE ══════════ */}
              <g transform="translate(415, 170)">
                <g className="sc-verdict sc-interactive-btn">
                  <rect x="-44" y="-15" width="88" height="26" rx="13"
                    fill="var(--red-bg)" stroke="var(--red-border)" strokeWidth="1.5" />
                  <text x="0" y="4.5" textAnchor="middle"
                    style={{ fontSize: 11, fontWeight: 700, fill: 'var(--red)', fontFamily: 'Instrument Sans, sans-serif', letterSpacing: '0.02em' }}>
                    ⚑  FLAG
                  </text>
                </g>
              </g>

              {/* ══════════ CAPTION ══════════ */}
              <text x="250" y="210" textAnchor="middle"
                style={{ fontSize: 9.5, fill: 'var(--text-4)', fontFamily: 'DM Mono, monospace', letterSpacing: '0.03em' }}>
                worker submits patch · overseer detects sabotage · majority flags
              </text>

            </svg>
          </div>
        </div>
      </div>
    </>
  )
}

