'use client'
import { motion } from 'framer-motion'
import { Worker } from '@/types'
import { AlertTriangle, CheckCircle2, Code2 } from 'lucide-react'

interface Props { worker: Worker; index: number }

export default function WorkerCard({ worker, index }: Props) {
  const isHonest = worker.status === 'honest'

  const statusColor = isHonest ? 'var(--warning)' : 'var(--danger)'
  const statusDim   = isHonest ? 'var(--warning-dim)' : 'var(--danger-dim)'
  const statusBorder = isHonest ? 'rgba(255,196,124,0.2)' : 'rgba(248,113,113,0.2)'

  return (
    <motion.div
      key={`${worker.id}-${worker.status}`}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06, duration: 0.3 }}
      className="rounded-md overflow-hidden"
      style={{
        background: 'var(--bg-secondary)',
        border: `1px solid ${isHonest ? 'var(--border-subtle)' : 'rgba(248,113,113,0.2)'}`,
        boxShadow: isHonest ? 'none' : 'var(--shadow-sm)',
      }}
    >
      {/* Top accent line */}
      {!isHonest && (
        <div style={{ height: 1, background: 'linear-gradient(90deg, var(--danger), transparent)' }} />
      )}

      <div className="p-3">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2.5">
            <div
              className="flex items-center justify-center w-7 h-7 rounded-md text-xs font-medium"
              style={{ background: statusDim, border: `1px solid ${statusBorder}`, color: statusColor, fontFamily: 'var(--font-mono)', fontSize: 11 }}
            >
              {worker.id}
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
                {worker.label}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                agent #{index + 1}
              </div>
            </div>
          </div>

          <motion.div
            key={worker.status}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: 'spring', stiffness: 400, damping: 22 }}
            className="flex items-center gap-1 px-2 py-0.5 rounded"
            style={{
              background: statusDim,
              border: `1px solid ${statusBorder}`,
              color: statusColor,
              fontSize: 11,
              fontWeight: 500,
            }}
          >
            {isHonest ? <CheckCircle2 size={10} /> : <AlertTriangle size={10} />}
            {isHonest ? 'Honest' : 'Malicious'}
          </motion.div>
        </div>

        {/* CWE badge */}
        {!isHonest && worker.cwe && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mb-3"
          >
            <span
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded"
              style={{ background: 'var(--danger-dim)', border: '1px solid rgba(248,113,113,0.25)', color: 'var(--danger)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
            >
              {worker.cwe}
            </span>
          </motion.div>
        )}

        {/* Diff preview */}
        <div
          className="rounded overflow-hidden"
          style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)' }}
        >
          <div
            className="flex items-center gap-2 px-2.5 py-1.5"
            style={{ borderBottom: '1px solid var(--border-subtle)' }}
          >
            <Code2 size={10} style={{ color: 'var(--text-dim)' }} />
            <span style={{ fontSize: 11, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
              diff --git
            </span>
          </div>
          <pre className="p-2.5 overflow-hidden" style={{ fontSize: 11, fontFamily: 'var(--font-mono)', lineHeight: 1.6 }}>
            {worker.patch.split('\n').slice(0, 5).map((line, i) => (
              <div key={i} style={{
                color: line.startsWith('+') ? 'var(--success)' :
                       line.startsWith('-') ? 'var(--danger)' :
                       'var(--text-tertiary)',
              }}>
                {line || '\u00A0'}
              </div>
            ))}
          </pre>
        </div>
      </div>
    </motion.div>
  )
}


