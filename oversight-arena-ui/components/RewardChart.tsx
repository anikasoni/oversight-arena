'use client'
import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useArenaStore } from '@/store/arenaStore'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { TrendingUp } from 'lucide-react'

function AnimatedNumber({ value, decimals = 2 }: { value: number; decimals?: number }) {
  const [cur, setCur] = useState(0)
  useEffect(() => {
    const start = cur, end = value, dur = 800, t0 = Date.now()
    const tick = () => {
      const p = Math.min((Date.now() - t0) / dur, 1)
      const e = 1 - Math.pow(1 - p, 3)
      setCur(start + (end - start) * e)
      if (p < 1) requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
  }, [value])
  return <span>{cur.toFixed(decimals)}</span>
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div
      className="px-2.5 py-2 rounded-md"
      style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', boxShadow: 'var(--shadow-md)' }}
    >
      <p style={{ fontSize: 10, color: 'var(--text-dim)', marginBottom: 2, fontFamily: 'var(--font-mono)' }}>Step {label}</p>
      <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--success)', fontFamily: 'var(--font-mono)' }}>
        R: {payload[0].value.toFixed(3)}
      </p>
    </div>
  )
}

export default function RewardChart() {
  const { rewardHistory, f1Score, mode } = useArenaStore()

  return (
    <div
      className="rounded-md overflow-hidden h-full flex flex-col"
      style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)' }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2"
        style={{ borderBottom: '1px solid var(--border-subtle)', background: 'var(--bg-tertiary)' }}
      >
        <div className="flex items-center gap-1.5">
          <TrendingUp size={11} style={{ color: 'var(--text-dim)' }} />
          <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontWeight: 500 }}>Reward</span>
        </div>
        <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>F1 score</span>
      </div>

      <div className="px-3 pt-3 pb-2">
        {/* F1 */}
        <div className="flex items-baseline gap-2 mb-2">
          <AnimatePresence mode="wait">
            <motion.div
              key={mode}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              style={{ fontSize: 28, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.03em', fontFamily: 'var(--font-mono)' }}
            >
              <AnimatedNumber value={f1Score} />
            </motion.div>
          </AnimatePresence>
          <motion.span
            key={`delta-${mode}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="px-1.5 py-0.5 rounded"
            style={{
              fontSize: 11,
              fontWeight: 500,
              color: mode === 'trained' ? 'var(--success)' : 'var(--danger)',
              background: mode === 'trained' ? 'var(--success-dim)' : 'var(--danger-dim)',
              border: `1px solid ${mode === 'trained' ? 'rgba(137,209,150,0.2)' : 'rgba(248,113,113,0.2)'}`,
            }}
          >
            {mode === 'trained' ? '↑ +0.29' : '↓ −0.29'}
          </motion.span>
        </div>

        {/* Comparison bar */}
        <div className="mb-2">
          <div className="flex justify-between mb-1" style={{ fontSize: 10, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
            <span>baseline 0.42</span>
            <span>trained 0.71</span>
          </div>
          <div
            className="h-1 rounded-full overflow-hidden"
            style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)' }}
          >
            <div className="relative h-full">
              <div className="absolute h-full rounded-full" style={{ width: '42%', background: 'var(--danger)', opacity: 0.5 }} />
              {mode === 'trained' && (
                <motion.div
                  className="absolute h-full rounded-full"
                  initial={{ width: '42%' }}
                  animate={{ width: '71%' }}
                  transition={{ duration: 1.0, ease: [0.16, 1, 0.3, 1] }}
                  style={{ background: 'var(--success)' }}
                />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="flex-1 px-1 pb-2 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={rewardHistory} margin={{ top: 4, right: 8, bottom: 0, left: -28 }}>
            <defs>
              <linearGradient id="rg2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#89d196" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#89d196" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 6" stroke="var(--border-subtle)" vertical={false} />
            <XAxis
              dataKey="step"
              tick={{ fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
              tickLine={false}
              axisLine={false}
              tickCount={4}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={0.5} stroke="var(--border)" strokeDasharray="3 4" />
            <Area
              type="monotone"
              dataKey="reward"
              stroke="var(--success)"
              strokeWidth={1.5}
              fill="url(#rg2)"
              dot={false}
              animationDuration={800}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="text-center pb-2" style={{ fontSize: 10, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', letterSpacing: '0.04em' }}>
        training steps →
      </div>
    </div>
  )
}