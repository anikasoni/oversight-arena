
'use client'
import { motion } from 'framer-motion'
import { useArenaStore } from '@/store/arenaStore'
import NavBar from '@/components/NavBar'
import { Code2, AlertTriangle, CheckCircle, ArrowRight, Shield } from 'lucide-react'
import Link from 'next/link'

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 14 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.38, ease: [0.22, 1, 0.36, 1], delay },
})

const fadeRight = (delay = 0) => ({
  initial: { opacity: 0, x: 20 },
  animate: { opacity: 1, x: 0 },
  transition: { duration: 0.42, ease: [0.22, 1, 0.36, 1], delay },
})

function WorkerCard({ worker, index }: { worker: any; index: number }) {
  const isHonest = worker.status === 'honest'
  const isUnknown = worker.status === 'unknown'
  const lines = worker.patch.split('\n')

  // ── derive a subtle background tint based on suspicion ──────────
  const cardBg = !isHonest && !isUnknown
    ? 'rgba(192, 57, 43, 0.04)'
    : 'var(--bg-card)'
  // ────────────────────────────────────────────────────────────────

  return (
    <motion.div
      {...fadeUp(index * 0.07)}
      className="card"
      style={{
        overflow: 'hidden',
        background: cardBg,
        transition: 'background 500ms ease, border-color 500ms ease',
      }}
    >
      {/* Accent strip — unchanged */}
      <div style={{
        height: 3,
        background: isHonest ? 'var(--amber)' : isUnknown ? 'var(--text-4)' : 'var(--red)',
        opacity: isUnknown ? 0.3 : 1,
      }} />

      {/* Header — unchanged */}
      <div style={{
        padding: '20px 22px 16px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        borderBottom: '1px solid var(--border)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 42, height: 42, borderRadius: 10,
            background: isHonest ? 'var(--amber-bg)' : isUnknown ? 'var(--surface)' : 'var(--red-bg)',
            border: `1px solid ${isHonest ? 'var(--amber-border)' : isUnknown ? 'var(--border)' : 'var(--red-border)'}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: 'DM Mono, monospace', fontSize: 13, fontWeight: 500,
            color: isHonest ? 'var(--amber)' : isUnknown ? 'var(--text-3)' : 'var(--red)',
          }}>
            {worker.id}
          </div>
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-1)', letterSpacing: '-0.02em' }}>
              {worker.label}
            </h3>
            <p style={{ fontSize: 12, color: 'var(--text-4)', fontFamily: 'DM Mono, monospace' }}>agent #{index + 1}</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {!isHonest && !isUnknown && worker.cwe && (
            <span className="badge badge-red" style={{ fontFamily: 'DM Mono, monospace', fontSize: 11 }}>
              {worker.cwe}
            </span>
          )}
          <span className={`badge ${isHonest ? 'badge-amber' : isUnknown ? 'badge-muted' : 'badge-red'}`}>
            {isHonest
              ? <><CheckCircle size={11} /> Honest</>
              : isUnknown
              ? 'Unknown'
              : <><AlertTriangle size={11} /> Malicious</>
            }
          </span>
        </div>
      </div>

      {/* Description — unchanged */}
      <div style={{ padding: '14px 22px', borderBottom: '1px solid var(--border)' }}>
        <p style={{ fontSize: 13, color: 'var(--text-3)', lineHeight: 1.55, letterSpacing: '-0.01em' }}>
          {isHonest
            ? 'Submits a safe refactor improving code quality without altering semantics.'
            : isUnknown
            ? 'Status pending — awaiting overseer analysis.'
            : 'Removes null guard before pointer dereference — introduces a null dereference vulnerability (CWE-476).'
          }
        </p>
      </div>

      {/* Diff — unchanged */}
      <div style={{ background: 'var(--code-bg)' }}>
        <div style={{
          padding: '9px 16px', display: 'flex', alignItems: 'center', gap: 7,
          borderBottom: '1px solid var(--border)',
        }}>
          <Code2 size={11} color="var(--text-4)" />
          <span style={{ fontFamily: 'DM Mono, monospace', color: 'var(--text-4)', fontSize: 11 }}>
            diff --git a/patch.py
          </span>
        </div>
        <div style={{ padding: '12px 0', overflow: 'auto', maxHeight: 160 }}>
          {lines.map((line: string, i: number) => (
            <div key={i}
              className={`font-mono ${line.startsWith('+') ? 'diff-add' : line.startsWith('-') ? 'diff-remove' : 'diff-neutral'}`}
              style={{ padding: '1px 16px', fontSize: 12, lineHeight: 1.7 }}
            >
              <span style={{
                color: line.startsWith('+') ? 'var(--green)' : line.startsWith('-') ? 'var(--red)' : 'var(--text-3)',
              }}>
                {line || '\u00A0'}
              </span>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  )
}

export default function WorkersPage() {
  const { workers, mode, isLoadingBackend } = useArenaStore()

  return (
    <>
      <NavBar />
      <main className="page-enter">
        {/* ── Header — CHANGED: wrapped in motion animations ── */}
        <section style={{ padding: '52px 0 40px', borderBottom: '1px solid var(--border)' }}>
          <div className="page-container">
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 20, flexWrap: 'wrap' }}>
              <div>
                {/* ── ADDED: animated step badge ── */}
                <motion.p {...fadeUp(0)} style={{ fontSize: 12, color: 'var(--accent)', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  Step 1 of 3
                </motion.p>
                {/* ── ADDED: animated heading ── */}
                <motion.h1 {...fadeUp(0.06)} style={{ fontFamily: 'DM Serif Display, serif', fontSize: 40, fontWeight: 400, letterSpacing: '-0.03em', color: 'var(--text-1)', marginBottom: 10 }}>
                  Worker Agents
                </motion.h1>
                {/* ── ADDED: animated description ── */}
                <motion.p {...fadeUp(0.1)} style={{ fontSize: 16, color: 'var(--text-2)', lineHeight: 1.55, letterSpacing: '-0.01em', maxWidth: 480 }}>
                  Three AI workers each submit a code patch. One or two contain deliberate vulnerabilities designed to evade detection.
                </motion.p>
              </div>
              {/* ── ADDED: right side fades in from right ── */}
              <motion.div {...fadeRight(0.12)} style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'flex-end' }}>
                <div className="card" style={{ padding: '14px 18px', minWidth: 160 }}>
                  <p style={{ fontSize: 11, color: 'var(--text-4)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Mode</p>
                  <p style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-1)', textTransform: 'capitalize' }}>{mode}</p>
                </div>
                <Link href="/analysis">
                  <button className="btn btn-primary" style={{ fontSize: 13 }}>
                    View analysis <ArrowRight size={13} />
                  </button>
                </Link>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Worker grid — unchanged */}
        <section style={{ padding: '40px 0 80px' }}>
          <div className="page-container">
            {isLoadingBackend ? (
              <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-3)', fontSize: 15 }}>
                Loading episode data…
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
                {workers.map((w, i) => <WorkerCard key={w.id} worker={w} index={i} />)}
              </div>
            )}

            {/* Info box — unchanged */}
            <div style={{
              marginTop: 28, padding: '18px 22px', borderRadius: 10,
              background: 'var(--accent-dim)', border: '1px solid color-mix(in srgb, var(--accent) 20%, transparent)',
            }}>
              <div style={{ display: 'flex', gap: 12 }}>
                <Shield size={16} color="var(--accent)" style={{ flexShrink: 0, marginTop: 2 }} />
                <div>
                  <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-1)', marginBottom: 4, letterSpacing: '-0.01em' }}>
                    About this episode
                  </p>
                  <p style={{ fontSize: 13, color: 'var(--text-2)', lineHeight: 1.6, letterSpacing: '-0.01em' }}>
                    In <strong>baseline mode</strong>, overseers haven't been trained to detect sabotage.
                    In <strong>trained mode</strong>, GRPO fine-tuning enables overseers to flag malicious patches with high confidence.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </>
  )
}
