'use client'
import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { OverseerState } from '@/types'
import { AlertOctagon, CheckCircle2, Eye } from 'lucide-react'

interface Props { overseer: OverseerState; index: number; mode: string }

function useStreamText(text: string, speed = 12) {
  const [displayed, setDisplayed] = useState('')
  const [done, setDone] = useState(false)
  useEffect(() => {
    setDisplayed(''); setDone(false)
    let i = 0
    const iv = setInterval(() => {
      if (i < text.length) { setDisplayed(text.slice(0, ++i)) }
      else { setDone(true); clearInterval(iv) }
    }, speed)
    return () => clearInterval(iv)
  }, [text, speed])
  return { displayed, done }
}

export default function OverseerCard({ overseer, index, mode }: Props) {
  const isFlag = overseer.action === 'FLAG'
  const { displayed, done } = useStreamText(overseer.reasoning, 11)
  const pct = Math.round(overseer.suspicion * 100)

  const barColor = overseer.suspicion > 0.7
    ? 'var(--danger)'
    : overseer.suspicion > 0.4
    ? 'var(--warning)'
    : 'var(--success)'

  return (
    <motion.div
      key={`${overseer.id}-${mode}`}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08 + 0.1, duration: 0.3 }}
      className="rounded-md overflow-hidden"
      style={{
        background: 'var(--bg-secondary)',
        border: `1px solid ${isFlag ? 'rgba(248,113,113,0.2)' : 'var(--border-subtle)'}`,
        boxShadow: isFlag ? 'var(--shadow-sm)' : 'none',
      }}
    >
      {isFlag && (
        <div style={{ height: 1, background: 'linear-gradient(90deg, var(--danger), transparent)' }} />
      )}

      <div className="p-3">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2.5">
            <div
              className="flex items-center justify-center w-7 h-7 rounded-md"
              style={{ background: 'var(--accent-dim)', border: '1px solid rgba(143,164,255,0.2)' }}
            >
              <Eye size={13} style={{ color: 'var(--accent)' }} />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
                  {overseer.label}
                </span>
                <span
                  className="px-1.5 py-px rounded"
                  style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', color: 'var(--text-tertiary)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                >
                  {overseer.specialty.slice(0, 3).toUpperCase()}
                </span>
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                {overseer.specialty}
              </div>
            </div>
          </div>

          <motion.div
            key={overseer.action}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: 'spring', stiffness: 400, damping: 20 }}
            className="flex items-center gap-1 px-2 py-0.5 rounded"
            style={{
              background: isFlag ? 'var(--danger-dim)' : 'var(--success-dim)',
              border: `1px solid ${isFlag ? 'rgba(248,113,113,0.25)' : 'rgba(137,209,150,0.25)'}`,
              color: isFlag ? 'var(--danger)' : 'var(--success)',
              fontSize: 11,
              fontWeight: 500,
            }}
          >
            {isFlag ? <AlertOctagon size={10} /> : <CheckCircle2 size={10} />}
            {overseer.action}
          </motion.div>
        </div>

        {/* Suspicion bar */}
        <div className="mb-3">
          <div className="flex justify-between items-center mb-1.5">
            <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>Suspicion score</span>
            <span style={{ fontSize: 11, fontWeight: 500, color: barColor, fontFamily: 'var(--font-mono)' }}>
              {overseer.suspicion.toFixed(2)}
            </span>
          </div>
          <div
            className="h-1 rounded-full overflow-hidden"
            style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-subtle)' }}
          >
            <motion.div
              key={overseer.suspicion}
              initial={{ width: 0 }}
              animate={{ width: `${pct}%` }}
              transition={{ duration: 0.8, delay: index * 0.08 + 0.2, ease: [0.16, 1, 0.3, 1] }}
              className="h-full rounded-full"
              style={{ background: barColor }}
            />
          </div>
          <div className="flex justify-between mt-1">
            {[0, 25, 50, 75, 100].map(v => (
              <span key={v} style={{ fontSize: 10, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>{v}</span>
            ))}
          </div>
        </div>

        {/* Reasoning */}
        <div
          className="rounded overflow-hidden"
          style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)' }}
        >
          <div
            className="flex items-center gap-2 px-2.5 py-1.5"
            style={{ borderBottom: '1px solid var(--border-subtle)' }}
          >
            <span style={{ fontSize: 11, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>reasoning.log</span>
            {!done && (
              <motion.span
                className="ml-auto w-1 h-3 rounded-sm"
                style={{ background: 'var(--accent)' }}
                animate={{ opacity: [1, 0, 1] }}
                transition={{ duration: 0.8, repeat: Infinity }}
              />
            )}
          </div>
          <div className="p-2.5 min-h-[64px]">
            <p style={{ fontSize: 11, fontFamily: 'var(--font-mono)', lineHeight: 1.65, color: 'var(--text-secondary)' }}>
              <span style={{ color: 'var(--accent)', marginRight: 6 }}>›</span>
              {displayed}
              {!done && <span className="inline-block w-1 h-3 ml-0.5 animate-pulse" style={{ background: 'var(--text-secondary)', verticalAlign: 'text-bottom' }} />}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

