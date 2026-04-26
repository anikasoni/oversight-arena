
'use client'
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useArenaStore } from '@/store/arenaStore'
import NavBar from '@/components/NavBar'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { AlertOctagon, CheckCircle, TrendingUp, Award, Shield, ArrowLeft } from 'lucide-react'
import Link from 'next/link'

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, ease: [0.25, 0.1, 0.25, 1], delay },
})

function AnimatedNumber({ value, decimals = 2 }: { value: number; decimals?: number }) {
  const [cur, setCur] = useState(0)
  useEffect(() => {
    const start = cur, end = value, dur = 900, t0 = Date.now()
    const tick = () => {
      const p = Math.min((Date.now() - t0) / dur, 1)
      const e = 1 - Math.pow(1 - p, 3)
      setCur(start + (end - start) * e)
      if (p < 1) requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
  }, [value])
  return <>{cur.toFixed(decimals)}</>
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: 8, padding: '8px 14px',
      boxShadow: 'var(--shadow-md)',
    }}>
      <p style={{ fontSize: 12, color: 'var(--text-3)', marginBottom: 2 }}>Step {label}</p>
      <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--accent)' }}>
        Reward: {payload[0].value.toFixed(3)}
      </p>
    </div>
  )
}

function VoteCard({ vote, index }: { vote: any; index: number }) {
  const isFlag = vote.action === 'FLAG'
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), index * 150)
    return () => clearTimeout(t)
  }, [index])

  return (
    <div
      className="card"
      style={{
        padding: '22px',
        textAlign: 'center',
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(10px)',
        transition: 'opacity 0.4s ease, transform 0.4s ease',
      }}
    >
      <p style={{ fontSize: 12, color: 'var(--text-3)', letterSpacing: '-0.01em', marginBottom: 8, fontFamily: 'DM Mono, monospace' }}>
        Overseer {vote.overseerId}
      </p>
      <div style={{
        width: 52, height: 52, borderRadius: 14,
        background: isFlag ? 'var(--red-bg)' : 'var(--green-bg)',
        border: `1px solid ${isFlag ? 'var(--red-border)' : 'var(--green-border)'}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        margin: '0 auto 12px',
      }}>
        {isFlag
          ? <AlertOctagon size={22} color="var(--red)" />
          : <CheckCircle size={22} color="var(--green)" />}
      </div>
      <span className={`badge ${isFlag ? 'badge-red' : 'badge-green'}`} style={{ fontSize: 14, padding: '6px 16px' }}>
        {vote.action}
      </span>
    </div>
  )
}

export default function ResultsPage() {
  const { votes, finalDecision, rewardHistory, f1Score, mode, bugDetected } = useArenaStore()
  const flagCount = votes.filter((v: any) => v.action === 'FLAG').length
  const isMajorityFlag = flagCount >= 2
  const isTrained = mode === 'trained'

  const [bannerReady, setBannerReady] = useState(false)
  useEffect(() => {
    setBannerReady(false)
    const t = setTimeout(() => setBannerReady(true), 850)
    return () => clearTimeout(t)
  }, [finalDecision])

  return (
    <>
      <NavBar />
      <main className="page-enter">
        {/* Page header */}
        <section style={{ padding: '55px 0 44px', borderBottom: '1px solid var(--border)' }}>
          <div className="page-container">
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 22, flexWrap: 'wrap' }}>
              <div>
                <motion.p {...fadeUp(0)} style={{ fontSize: 13, color: 'var(--accent)', fontWeight: 500, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Step 3 of 3
                </motion.p>
                <motion.h1 {...fadeUp(0.06)} style={{ fontSize: 40, fontWeight: 600, letterSpacing: '-0.04em', color: 'var(--text-1)', marginBottom: 10 }}>
                  Results
                </motion.h1>
                <motion.p {...fadeUp(0.1)} style={{ fontSize: 17, color: 'var(--text-2)', lineHeight: 1.5, letterSpacing: '-0.02em', maxWidth: 480 }}>
                  Majority vote determines the final decision. The reward function measures detection quality.
                </motion.p>
              </div>
              {/* FIX: was className="btn-ghost" — needs both btn + btn-ghost */}
              <motion.div {...fadeUp(0.12)}>
                <Link href="/analysis">
                  <button className="btn btn-ghost" style={{ fontSize: 14 }}>
                    <ArrowLeft size={13} /> Back to analysis
                  </button>
                </Link>
              </motion.div>
            </div>
          </div>
        </section>

        <section style={{ padding: '44px 0 88px' }}>
          <div className="page-container">

            {/* VOTE BREAKDOWN */}
            <div style={{ marginBottom: 32 }}>
              <h2 style={{ fontSize: 22, fontWeight: 600, letterSpacing: '-0.03em', color: 'var(--text-1)', marginBottom: 16 }}>
                Vote breakdown
              </h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                {votes.map((v: any, i: number) => <VoteCard key={v.overseerId} vote={v} index={i} />)}
              </div>
            </div>

            {/* FINAL DECISION — FIX: replaced var(--danger-bg/bdr/txt) and var(--success-*) with real variables */}
            <AnimatePresence mode="wait">
              <motion.div
                key={finalDecision + String(bannerReady)}
                initial={{ opacity: 0, scale: 0.92, y: 8 }}
                animate={bannerReady
                  ? { opacity: 1, scale: 1, y: 0 }
                  : { opacity: 0, scale: 0.92, y: 8 }
                }
                exit={{ opacity: 0 }}
                transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
                style={{
                  marginBottom: 32,
                  borderRadius: 18,
                  padding: '44px 40px',
                  background: isMajorityFlag ? 'var(--red-bg)' : 'var(--green-bg)',
                  border: `1px solid ${isMajorityFlag ? 'var(--red-border)' : 'var(--green-border)'}`,
                  textAlign: 'center',
                }}
              >
                <div style={{
                  width: 64, height: 64, borderRadius: 18,
                  background: isMajorityFlag ? 'var(--red)' : 'var(--green)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  margin: '0 auto 20px',
                }}>
                  {isMajorityFlag
                    ? <AlertOctagon size={28} color="#fff" />
                    : <CheckCircle size={28} color="#fff" />}
                </div>
                <p style={{ fontSize: 13, color: 'var(--text-3)', letterSpacing: '0.04em', textTransform: 'uppercase', marginBottom: 10, fontWeight: 500 }}>
                  Final Decision
                </p>
                <h2 style={{
                  fontSize: 28, fontWeight: 600,
                  color: isMajorityFlag ? 'var(--red)' : 'var(--green)',
                  letterSpacing: '-0.03em',
                }}>
                  {finalDecision}
                </h2>
                <p style={{ fontSize: 15, color: 'var(--text-3)', marginTop: 8, letterSpacing: '-0.01em' }}>
                  {flagCount} of {votes.length} overseers voted FLAG
                </p>
              </motion.div>
            </AnimatePresence>

            <div style={{ borderTop: '1px solid var(--border)', margin: '32px 0' }} />

            {/* F1 SCORE + REWARD CHART */}
            <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 24, alignItems: 'start' }}>

              {/* F1 score panel */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <motion.div {...fadeUp(0)} className="card" style={{ padding: '28px 24px', textAlign: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, marginBottom: 12 }}>
                    <Award size={15} color="var(--text-3)" />
                    <span style={{ fontSize: 13, color: 'var(--text-3)', letterSpacing: '-0.01em' }}>F1 Score</span>
                  </div>
                  <AnimatePresence mode="wait">
                    <motion.p
                      key={mode}
                      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                      style={{
                        fontSize: 56, fontWeight: 600, letterSpacing: '-0.04em',
                        color: isTrained ? 'var(--accent)' : 'var(--text-1)',
                        lineHeight: 1,
                      }}
                    >
                      <AnimatedNumber value={f1Score} />
                    </motion.p>
                  </AnimatePresence>
                  <motion.p
                    key={`delta-${mode}`}
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    style={{ fontSize: 14, marginTop: 8, color: isTrained ? 'var(--green)' : 'var(--red)', fontWeight: 500 }}
                  >
                    {isTrained ? '↑ +0.29 vs baseline' : '↓ −0.29 vs trained'}
                  </motion.p>
                </motion.div>

                {/* Comparison bar */}
                <motion.div {...fadeUp(0.05)} className="card" style={{ padding: '20px 22px' }}>
                  <p style={{ fontSize: 13, color: 'var(--text-3)', marginBottom: 14, letterSpacing: '-0.01em', fontWeight: 500 }}>
                    Model comparison
                  </p>
                  {[
                    { label: 'Baseline', value: 0.42, color: 'var(--text-3)' },
                    { label: 'GRPO trained', value: 0.71, color: 'var(--accent)' },
                  ].map(({ label, value, color }) => (
                    <div key={label} style={{ marginBottom: 14 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                        <span style={{ fontSize: 13, color: 'var(--text-1)', letterSpacing: '-0.01em' }}>{label}</span>
                        <span style={{ fontSize: 13, fontWeight: 500, color, fontFamily: 'DM Mono, monospace' }}>{value}</span>
                      </div>
                      <div style={{ height: 4, background: 'var(--bg-2)', borderRadius: 2, overflow: 'hidden' }}>
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${value * 100}%` }}
                          transition={{ duration: 1.0, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
                          style={{ height: '100%', background: color, borderRadius: 2 }}
                        />
                      </div>
                    </div>
                  ))}
                </motion.div>

                {/* Reward formula */}
                <motion.div {...fadeUp(0.1)} className="card" style={{ padding: '18px 20px' }}>
                  <p style={{ fontSize: 12, color: 'var(--text-3)', marginBottom: 6, fontWeight: 500, letterSpacing: '-0.01em' }}>
                    Reward function
                  </p>
                                  <p style={{ fontFamily: 'DM Mono, monospace', fontSize: 12, color: 'var(--text-2)', lineHeight: 1.7 }}>
                                      R = 0.65 · F1<br />
                                      + 0.20 · early<br />
                                      + 0.10 · recall<br />
                                      − 0.20 · FP
                                  </p>
                </motion.div>
              </div>

              {/* Reward chart */}
              <motion.div {...fadeUp(0.08)} className="card" style={{ padding: '24px', overflow: 'hidden' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 22 }}>
                  <TrendingUp size={16} color="var(--text-3)" />
                  <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-1)', letterSpacing: '-0.02em' }}>
                    Reward over training
                  </span>
                  {/* FIX: badge-neutral → badge-muted */}
                  <span className={`badge ${isTrained ? 'badge-blue' : 'badge-muted'}`} style={{ marginLeft: 'auto', fontSize: 12 }}>
                    {isTrained ? 'GRPO' : 'Baseline'}
                  </span>
                </div>
                <div style={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={rewardHistory} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                      <defs>
                        <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={isTrained ? '#1D6AE8' : '#7A7870'} stopOpacity={0.18} />
                          <stop offset="100%" stopColor={isTrained ? '#1D6AE8' : '#7A7870'} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="2 6" stroke="var(--border)" vertical={false} />
                      <XAxis
                        dataKey="step"
                        tick={{ fill: 'var(--text-3)', fontSize: 11, fontFamily: 'DM Mono, monospace' }}
                        tickLine={false} axisLine={false}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tick={{ fill: 'var(--text-3)', fontSize: 11, fontFamily: 'DM Mono, monospace' }}
                        tickLine={false} axisLine={false} tickCount={5}
                      />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine y={0.5} stroke="var(--border-2)" strokeDasharray="4 4" />
                      <Area
                        type="monotone" dataKey="reward"
                        stroke={isTrained ? 'var(--accent)' : 'var(--text-3)'}
                        strokeWidth={2}
                        fill="url(#rewardGrad)"
                        dot={false}
                        animationDuration={800}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <p style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 10, textAlign: 'center', fontFamily: 'DM Mono, monospace' }}>
                  training steps →
                </p>
              </motion.div>
            </div>

          </div>
        </section>
      </main>
    </>
  )
}


