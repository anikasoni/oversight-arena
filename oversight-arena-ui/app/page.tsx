'use client'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { useArenaStore } from '@/store/arenaStore'
import NavBar from '@/components/NavBar'
import OversightScene from '@/components/OversightScene'
import { ArrowRight, Shield, Cpu, TrendingUp, CheckCircle, AlertTriangle, Activity } from 'lucide-react'

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1], delay },
})

const fadeRight = (delay = 0) => ({
  initial: { opacity: 0, x: 32 },
  animate: { opacity: 1, x: 0 },
  transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1], delay },
})

export default function Home() {
  const { mode, bugDetected, f1Score, isLoadingBackend, setMode, isConnected } = useArenaStore()
  const isTrained = mode === 'trained'

  return (
    <>
      <NavBar />
      <main>

        {/* ── HERO ──────────────────────────────────────────────────────── */}
        <section style={{ padding: '80px 0 64px', background: 'var(--bg)', position: 'relative', overflow: 'hidden' }}>
          <div className="grid-bg" style={{
            position: 'absolute', inset: 0, opacity: 0.4,
            maskImage: 'radial-gradient(ellipse 70% 60% at 50% 0%, black, transparent)',
            WebkitMaskImage: 'radial-gradient(ellipse 70% 60% at 50% 0%, black, transparent)',
          }} />

          <div className="page-container" style={{ position: 'relative' }}>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '48px',
              alignItems: 'center',
            }}>

              {/* LEFT: text content */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <motion.div {...fadeUp(0)}>
                  <span className="badge badge-blue" style={{ marginBottom: 24 }}>
                    <Activity size={11} />
                    AI Safety Research
                  </span>
                </motion.div>

                <motion.h1 {...fadeUp(0.06)} style={{
                  fontFamily: 'DM Serif Display, Georgia, serif',
                  fontSize: 'clamp(38px, 5vw, 68px)',
                  fontWeight: 400,
                  color: 'var(--text-1)',
                  letterSpacing: '-0.03em',
                  lineHeight: 1.05,
                  marginBottom: 20,
                  textAlign: 'left',
                }}>
                  Multi-Agent<br />
                  <em style={{ fontStyle: 'italic', color: 'var(--accent)' }}>Scalable Oversight</em>
                </motion.h1>

                <motion.p {...fadeUp(0.1)} style={{
                  fontSize: 17, color: 'var(--text-2)',
                  lineHeight: 1.6, letterSpacing: '-0.01em',
                  maxWidth: 440, marginBottom: 36,
                  textAlign: 'left',
                }}>
                  Three worker agents submit code patches. One or more are malicious.
                  An ensemble of AI overseers must detect sabotage through majority vote.
                </motion.p>

                <motion.div {...fadeUp(0.14)} style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                  <Link href="/workers">
                    <button className="btn btn-primary">
                      View Workers <ArrowRight size={14} />
                    </button>
                  </Link>
                  <Link href="/results">
                    <button className="btn btn-secondary">
                      See Results
                    </button>
                  </Link>
                </motion.div>
              </div>

              {/* RIGHT: animated bot scene */}
              <motion.div {...fadeRight(0.18)} style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <OversightScene />
              </motion.div>

            </div>
          </div>
        </section>

        {/* ── STATUS BANNER ─────────────────────────────────────────────── */}
        <section style={{
          background: isTrained && bugDetected ? 'var(--red-bg)' : isTrained ? 'var(--green-bg)' : 'var(--bg-2)',
          borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)',
        }}>
          <div className="page-container" style={{
            padding: '14px 24px', display: 'flex',
            alignItems: 'center', justifyContent: 'space-between',
            flexWrap: 'wrap', gap: 10,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
              {bugDetected
                ? <AlertTriangle size={15} color="var(--red)" />
                : <CheckCircle size={15} color="var(--green)" />
              }
              <span style={{ fontSize: 14, color: 'var(--text-1)', fontWeight: 500, letterSpacing: '-0.01em' }}>
                {bugDetected
                  ? 'Sabotage detected — malicious worker flagged'
                  : mode === 'trained'
                  ? 'All workers cleared by overseer panel'
                  : 'Baseline mode — overseers not yet trained'
                }
              </span>
              {!isConnected && (
                <span className="badge badge-muted" style={{ fontSize: 11 }}>Demo data</span>
              )}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 13, color: 'var(--text-3)' }}>Mode:</span>
              <span className={`badge ${isTrained ? 'badge-blue' : 'badge-muted'}`}>
                {isTrained ? 'GRPO Trained' : 'Baseline'}
              </span>
              <span style={{ fontSize: 13, color: 'var(--text-2)' }}>
                F1: <strong style={{ fontFamily: 'DM Mono, monospace' }}>{f1Score.toFixed(2)}</strong>
              </span>
            </div>
          </div>
        </section>

        {/* ── STATS ROW ────────────────────────────────────────────────── */}
        <section style={{ padding: '56px 0' }}>
          <div className="page-container">
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: 16, marginBottom: 16,
            }}>
              {[
                { label: 'Worker Agents', value: '3', sub: 'per episode' },
                { label: 'F1 Score', value: f1Score.toFixed(2), sub: isTrained ? '↑ +0.29 vs baseline' : 'Baseline', accent: isTrained },
                { label: 'Reward Weight', value: '0.75', sub: '× F1 score' },
                { label: 'Training', value: isTrained ? 'GRPO' : 'None', sub: isTrained ? 'Qwen2.5 + LoRA' : 'Untrained', accent: isTrained },
              ].map(({ label, value, sub, accent }) => (
                <div key={label} className="card" style={{ padding: '24px 20px', textAlign: 'center' }}>
                  <p style={{ fontSize: 12, color: 'var(--text-3)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 500 }}>{label}</p>
                  <p className="stat-number" style={{ fontSize: 36, color: accent ? 'var(--accent)' : 'var(--text-1)' }}>{value}</p>
                  <p style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 6 }}>{sub}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── PIPELINE STEPS ──────────────────────────────────────────── */}
        <section style={{ padding: '0 0 72px', background: 'var(--bg-2)', borderTop: '1px solid var(--border)' }}>
          <div className="page-container" style={{ paddingTop: 56 }}>
            <motion.div {...fadeUp(0)} style={{ marginBottom: 40 }}>
              <h2 style={{
                fontFamily: 'DM Serif Display, serif',
                fontSize: 34, fontWeight: 400, letterSpacing: '-0.03em',
                color: 'var(--text-1)', marginBottom: 8,
              }}>
                How it works
              </h2>
              <p style={{ fontSize: 15, color: 'var(--text-3)', letterSpacing: '-0.01em' }}>
                Follow the oversight pipeline from submission to detection.
              </p>
            </motion.div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
              {[
                { n: 1, icon: Shield,    title: 'Worker Agents', href: '/workers',  color: 'var(--amber)',  bg: 'var(--amber-bg)',
                  desc: 'Three AI workers submit code patches. At least one contains a deliberate vulnerability.' },
                { n: 2, icon: Cpu,       title: 'Overseer Analysis', href: '/analysis', color: 'var(--accent)', bg: 'var(--accent-dim)',
                  desc: 'Three specialized oversight models independently analyze each patch and compute suspicion scores.' },
                { n: 3, icon: TrendingUp, title: 'Results & Reward', href: '/results',  color: 'var(--green)',  bg: 'var(--green-bg)',
                  desc: 'Majority vote determines the verdict. GRPO training improves detection F1 by +0.29.' },
              ].map(({ n, icon: Icon, title, href, color, bg, desc }, i) => (
                <motion.div key={n} {...fadeUp(i * 0.08)}>
                  <Link href={href} style={{ textDecoration: 'none' }}>
                    <div className="card" style={{
                      padding: '28px 24px', cursor: 'pointer',
                      transition: 'transform 0.18s ease, box-shadow 0.18s ease',
                    }}
                      onMouseEnter={e => {
                        const el = e.currentTarget as HTMLElement
                        el.style.transform = 'translateY(-2px)'
                        el.style.boxShadow = '0 8px 24px rgba(0,0,0,0.08)'
                      }}
                      onMouseLeave={e => {
                        const el = e.currentTarget as HTMLElement
                        el.style.transform = 'translateY(0)'
                        el.style.boxShadow = ''
                      }}
                    >
                      <div style={{
                        width: 44, height: 44, borderRadius: 12,
                        background: bg, display: 'flex', alignItems: 'center', justifyContent: 'center',
                        marginBottom: 16,
                      }}>
                        <Icon size={20} color={color} />
                      </div>
                      <div style={{
                        fontSize: 11, fontWeight: 600, color: 'var(--text-4)',
                        textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6,
                      }}>Step {n}</div>
                      <h3 style={{ fontSize: 17, fontWeight: 600, color: 'var(--text-1)', letterSpacing: '-0.02em', marginBottom: 8 }}>
                        {title}
                      </h3>
                      <p style={{ fontSize: 13, color: 'var(--text-3)', lineHeight: 1.55, letterSpacing: '-0.01em' }}>
                        {desc}
                      </p>
                    </div>
                  </Link>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

      </main>
    </>
  )
}