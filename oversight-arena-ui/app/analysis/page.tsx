'use client'
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useArenaStore } from '@/store/arenaStore'
import NavBar from '@/components/NavBar'
import { Eye, AlertOctagon, CheckCircle, ArrowRight, GitBranch, AlertTriangle, Search } from 'lucide-react'
import Link from 'next/link'

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, ease: [0.25, 0.1, 0.25, 1], delay },
})

const fadeRight = (delay = 0) => ({
  initial: { opacity: 0, x: 20 },
  animate: { opacity: 1, x: 0 },
  transition: { duration: 0.42, ease: [0.22, 1, 0.36, 1], delay },
})

function useStreamText(text: string, active = true) {
  const [displayed, setDisplayed] = useState('')
  const [done, setDone] = useState(false)
  useEffect(() => {
    if (!active) { setDisplayed(text); setDone(true); return }
    setDisplayed(''); setDone(false)
    let i = 0
    const iv = setInterval(() => {
      if (i < text.length) setDisplayed(text.slice(0, ++i))
      else { setDone(true); clearInterval(iv) }
    }, 14)
    return () => clearInterval(iv)
  }, [text, active])
  return { displayed, done }
}

function OverseerRow({ overseer, index }: { overseer: any; index: number }) {
  const isFlag = overseer.action === 'FLAG'
  const { displayed, done } = useStreamText(overseer.reasoning)
  const pct = Math.round(overseer.suspicion * 100)
  const barColor = overseer.suspicion > 0.7 ? '#dc2626' : overseer.suspicion > 0.4 ? '#d97706' : '#16a34a'

  const specialtyColors: Record<string, string> = {
    Precision: 'badge-blue',
    Speed: 'badge-warn',
    Recall: 'badge-neutral',
  }

  return (
    <motion.div {...fadeUp(index * 0.08)} className="card-white" style={{ overflow: 'hidden' }}>
      <div style={{
        padding: '20px 24px',
        display: 'grid',
        gridTemplateColumns: '1fr auto',
        gap: 16,
        alignItems: 'center',
        borderBottom: '1px solid rgba(0,0,0,0.06)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 10,
            background: 'rgba(41,151,255,0.08)',
            border: '1px solid rgba(41,151,255,0.15)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Eye size={18} color="var(--apple-blue)" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h3 style={{ fontSize: 17, fontWeight: 600, color: 'var(--charcoal)', letterSpacing: '-0.02em' }}>
                {overseer.label}
              </h3>
              <span className={`badge ${specialtyColors[overseer.specialty] ?? 'badge-neutral'}`} style={{ fontSize: 11 }}>
                {overseer.specialty}
              </span>
            </div>
            <p style={{ fontSize: 13, color: 'var(--storm)', letterSpacing: '-0.01em' }}>AI oversight model · ID: {overseer.id}</p>
          </div>
        </div>

        <span className={`badge ${isFlag ? 'badge-danger' : 'badge-success'}`} style={{ fontSize: 14, padding: '6px 14px' }}>
          {isFlag ? <AlertOctagon size={13} /> : <CheckCircle size={13} />}
          {isFlag ? 'FLAG' : 'PASS'}
        </span>
      </div>

      <div style={{ padding: '20px 24px' }}>
        {/* Suspicion score */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 13, color: 'var(--storm)', letterSpacing: '-0.01em' }}>Suspicion score</span>
            <span style={{ fontSize: 17, fontWeight: 600, color: barColor, fontFamily: 'var(--mono)', letterSpacing: 0 }}>
              {overseer.suspicion.toFixed(2)}
            </span>
          </div>
          <div style={{ height: 6, background: 'var(--whisper)', borderRadius: 3, overflow: 'hidden' }}>
            <motion.div
              key={overseer.suspicion}
              initial={{ width: 0 }}
              animate={{ width: `${pct}%` }}
              transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: index * 0.1 }}
              style={{ height: '100%', background: barColor, borderRadius: 3 }}
            />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
            {[0, 25, 50, 75, 100].map(v => (
              <span key={v} style={{ fontSize: 11, color: 'var(--storm)', fontFamily: 'var(--mono)' }}>{v}</span>
            ))}
          </div>
        </div>

        {/* Reasoning */}
        <div style={{
          background: 'var(--whisper)',
          borderRadius: 8,
          overflow: 'hidden',
          border: '1px solid rgba(0,0,0,0.06)',
        }}>
          <div style={{
            padding: '8px 14px',
            borderBottom: '1px solid rgba(0,0,0,0.06)',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <span className="mono" style={{ color: 'var(--storm)', fontSize: 11 }}>reasoning.log</span>
            {!done && (
              <span className="pulse-dot" style={{ fontSize: 11, color: 'var(--apple-blue)' }}>writing…</span>
            )}
          </div>
          <div style={{ padding: '14px 16px', minHeight: 72 }}>
            <p className="mono" style={{ color: 'var(--secondary)', fontSize: 13, lineHeight: 1.65 }}>
              <span style={{ color: 'var(--apple-blue)', marginRight: 8 }}>›</span>
              {displayed}
              {!done && <span className="pulse-dot" style={{ display: 'inline-block', width: 6, height: 13, background: 'var(--storm)', marginLeft: 2, borderRadius: 1, verticalAlign: 'text-bottom' }} />}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function CodeDiff() {
  const { bugDetected, mode, diffLines, diffWorkerLabel } = useArenaStore()
  const addCount = diffLines.filter((l: any) => l.type === 'add' || l.type === 'bug').length
  const removeCount = diffLines.filter((l: any) => l.type === 'remove').length

  return (
    <motion.div {...fadeUp(0)} className="card-white" style={{ overflow: 'hidden', position: 'sticky', top: 60 }}>
      <div style={{
        padding: '14px 18px',
        borderBottom: '1px solid rgba(0,0,0,0.06)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'var(--whisper)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <GitBranch size={13} color="var(--storm)" />
          <span className="mono" style={{ color: 'var(--storm)', fontSize: 12 }}>
            {diffWorkerLabel}/malicious.patch
          </span>
        </div>
        {bugDetected
          ? <span className="badge badge-danger" style={{ fontSize: 12 }}><AlertTriangle size={11} /> Bug detected</span>
          : <span className="badge badge-neutral" style={{ fontSize: 12 }}><Search size={11} /> Scanning…</span>}
      </div>

      <div style={{ overflow: 'auto', maxHeight: 420 }}>
        <div style={{ display: 'flex' }}>
          <div style={{
            background: 'var(--whisper)', padding: '12px 10px',
            borderRight: '1px solid rgba(0,0,0,0.06)',
            minWidth: 38, textAlign: 'right', userSelect: 'none',
          }}>
            {diffLines.map((_: any, i: number) => (
              <div key={i} className="mono" style={{ lineHeight: '22px', fontSize: 11, color: 'var(--storm)' }}>{i + 1}</div>
            ))}
          </div>
          <div style={{ flex: 1, padding: '12px 0' }}>
            {diffLines.map((line: any, i: number) => {
              const t = line.type
              const isBug = t === 'bug'
              const cls = isBug && bugDetected ? 'diff-bug' : t === 'remove' ? 'diff-remove' : t === 'add' || t === 'bug' ? 'diff-add' : t === 'header' ? 'diff-header' : 'diff-neutral'
              return (
                <div key={i} className={`mono ${cls}`} style={{ padding: '0 14px', lineHeight: '22px' }}>
                  <span style={{
                    color: isBug && bugDetected ? '#dc2626' :
                           isBug ? '#16a34a' :
                           t === 'remove' ? '#dc2626' :
                           t === 'add' ? '#16a34a' :
                           t === 'header' ? 'var(--storm)' :
                           'var(--secondary)',
                    fontWeight: isBug && bugDetected ? 500 : 400,
                    fontSize: 13,
                  }}>
                    {line.text || '\u00A0'}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div style={{
        padding: '10px 18px',
        borderTop: '1px solid rgba(0,0,0,0.06)',
        background: 'var(--whisper)',
        display: 'flex', gap: 16,
      }}>
        <span className="mono" style={{ color: '#16a34a', fontSize: 12 }}>+{addCount} addition{addCount !== 1 ? 's' : ''}</span>
        <span className="mono" style={{ color: '#dc2626', fontSize: 12 }}>−{removeCount} deletion{removeCount !== 1 ? 's' : ''}</span>
        {bugDetected && (
          <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="mono" style={{ color: '#dc2626', fontSize: 12, marginLeft: 'auto', fontWeight: 500 }}>
            ⚠ Removes null guard
          </motion.span>
        )}
      </div>
    </motion.div>
  )
}

export default function AnalysisPage() {
  const { overseers, mode, isLoadingBackend, bugDetected } = useArenaStore()

  return (
    <>
      <NavBar />
      <main className="page-enter">
        {/* ── Page header — CHANGED: wrapped in motion animations ── */}
        <section style={{ padding: '55px 0 44px', borderBottom: '1px solid rgba(0,0,0,0.06)' }}>
          <div className="page-container">
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 22, flexWrap: 'wrap' }}>
              <div>
                {/* ── ADDED: animated step label ── */}
                <motion.p {...fadeUp(0)} style={{ fontSize: 13, color: 'var(--apple-blue)', fontWeight: 500, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Step 2 of 3
                </motion.p>
                {/* ── ADDED: animated heading ── */}
                <motion.h1 {...fadeUp(0.06)} style={{ fontSize: 40, fontWeight: 600, letterSpacing: '-0.04em', color: 'var(--charcoal)', marginBottom: 10 }}>
                  Overseer Analysis
                </motion.h1>
                {/* ── ADDED: animated description ── */}
                <motion.p {...fadeUp(0.1)} style={{ fontSize: 17, color: 'var(--storm)', lineHeight: 1.5, letterSpacing: '-0.02em', maxWidth: 480 }}>
                  Three specialized oversight models independently analyze each code patch. Each computes a suspicion score and reasoning.
                </motion.p>
              </div>
              {/* ── ADDED: right side fades in from right ── */}
              <motion.div {...fadeRight(0.12)} style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'flex-end' }}>
                <span className={`badge ${bugDetected ? 'badge-red' : 'badge-muted'}`} style={{ fontSize: 14, padding: '7px 16px' }}>
                  {bugDetected ? <AlertTriangle size={13} /> : <Search size={13} />}
                  {bugDetected ? 'Sabotage detected' : 'No flags raised'}
                </span>
                <Link href="/results">
                  <button className="btn btn-primary" style={{ fontSize: 14 }}>
                    View results <ArrowRight size={13} />
                  </button>
                </Link>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Two-column layout — unchanged */}
        <section style={{ padding: '44px 0 88px' }}>
          <div className="page-wide">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 24, alignItems: 'start' }}>

              {/* Overseer cards */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {isLoadingBackend ? (
                  <div style={{ textAlign: 'center', padding: '66px 0', color: 'var(--storm)', fontSize: 15 }}>Loading…</div>
                ) : (
                  overseers.map((o: any, i: number) => <OverseerRow key={o.id} overseer={o} index={i} />)
                )}
              </div>

              {/* Code diff panel */}
              <div>
                <p style={{ fontSize: 13, color: 'var(--storm)', letterSpacing: '-0.01em', marginBottom: 12, fontWeight: 500 }}>
                  Malicious patch under review
                </p>
                <CodeDiff />
              </div>
            </div>
          </div>
        </section>
      </main>
    </>
  )
}

