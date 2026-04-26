'use client'
import { motion, AnimatePresence } from 'framer-motion'
import { useArenaStore } from '@/store/arenaStore'
import { Vote } from '@/types'
import { AlertOctagon, CheckCircle2 } from 'lucide-react'

function VoteBadge({ vote, index }: { vote: Vote; index: number }) {
  const isFlag = vote.action === 'FLAG'
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.1, type: 'spring', stiffness: 350, damping: 24 }}
      className="flex flex-col items-center gap-1.5 px-3 py-2.5 rounded-md flex-1"
      style={{
        background: isFlag ? 'var(--danger-dim)' : 'var(--success-dim)',
        border: `1px solid ${isFlag ? 'rgba(248,113,113,0.22)' : 'rgba(137,209,150,0.22)'}`,
      }}
    >
      <span style={{ fontSize: 10, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
        {vote.overseerId}
      </span>
      <div
        className="flex items-center gap-1"
        style={{ color: isFlag ? 'var(--danger)' : 'var(--success)', fontSize: 11, fontWeight: 600 }}
      >
        {isFlag ? <AlertOctagon size={11} /> : <CheckCircle2 size={11} />}
        {vote.action}
      </div>
    </motion.div>
  )
}

export default function VoteDisplay() {
  const { votes, finalDecision, mode, bugDetected } = useArenaStore()
  const flagCount = votes.filter(v => v.action === 'FLAG').length
  const isMajorityFlag = flagCount >= 2

  return (
    <motion.div
      key={mode}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="rounded-md overflow-hidden"
      style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)' }}
    >
      {/* Header */}
      <div
        className="px-3 py-2 flex items-center justify-between"
        style={{ borderBottom: '1px solid var(--border-subtle)', background: 'var(--bg-tertiary)' }}
      >
        <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontWeight: 500 }}>Majority vote</span>
        <div className="flex items-center gap-3" style={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}>
          <span style={{ color: 'var(--danger)' }}>{flagCount} flag</span>
          <span style={{ color: 'var(--border)' }}>·</span>
          <span style={{ color: 'var(--success)' }}>{votes.length - flagCount} pass</span>
        </div>
      </div>

      <div className="p-3">
        {/* Vote badges */}
        <div className="flex items-stretch gap-2 mb-3">
          {votes.map((v, i) => <VoteBadge key={`${v.overseerId}-${mode}`} vote={v} index={i} />)}
        </div>

        {/* Divider */}
        <div className="flex items-center gap-2 mb-3">
          <div className="flex-1 h-px" style={{ background: 'var(--border-subtle)' }} />
          <span style={{ fontSize: 10, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>consensus</span>
          <div className="flex-1 h-px" style={{ background: 'var(--border-subtle)' }} />
        </div>

        {/* Final decision */}
        <AnimatePresence mode="wait">
          <motion.div
            key={finalDecision}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.25 }}
            className="rounded-md py-3 px-3 text-center"
            style={{
              background: isMajorityFlag ? 'var(--danger-dim)' : 'var(--success-dim)',
              border: `1px solid ${isMajorityFlag ? 'rgba(248,113,113,0.25)' : 'rgba(137,209,150,0.2)'}`,
            }}
          >
            <div style={{ fontSize: 10, color: 'var(--text-dim)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Final decision
            </div>
            <div style={{
              fontSize: 12,
              fontWeight: 600,
              color: isMajorityFlag ? 'var(--danger)' : 'var(--success)',
              letterSpacing: '-0.01em',
            }}>
              {finalDecision}
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

