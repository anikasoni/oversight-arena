'use client'
import { motion } from 'framer-motion'
import { useArenaStore } from '@/store/arenaStore'
import { GitBranch, AlertTriangle, Search } from 'lucide-react'

export default function CodeDiffViewer() {
  const { bugDetected, mode, diffLines, diffWorkerLabel } = useArenaStore()
  const addCount = diffLines.filter(l => l.type === 'add' || l.type === 'bug').length
  const removeCount = diffLines.filter(l => l.type === 'remove').length

  return (
    <motion.div
      key={mode}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="rounded-md overflow-hidden"
      style={{
        background: 'var(--bg-secondary)',
        border: `1px solid ${bugDetected ? 'rgba(248,113,113,0.25)' : 'var(--border-subtle)'}`,
        boxShadow: bugDetected ? 'var(--shadow-sm)' : 'none',
      }}
    >
      {bugDetected && (
        <div style={{ height: 1, background: 'linear-gradient(90deg, var(--danger), transparent)' }} />
      )}

      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2"
        style={{ borderBottom: '1px solid var(--border-subtle)', background: 'var(--bg-tertiary)' }}
      >
        <div className="flex items-center gap-2">
          <GitBranch size={11} style={{ color: 'var(--text-dim)' }} />
          <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
            {diffWorkerLabel}/malicious.patch
          </span>
        </div>

        {bugDetected ? (
          <motion.div
            initial={{ opacity: 0, x: 4 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-1 px-2 py-0.5 rounded"
            style={{ background: 'var(--danger-dim)', border: '1px solid rgba(248,113,113,0.25)', color: 'var(--danger)', fontSize: 11, fontWeight: 500 }}
          >
            <AlertTriangle size={10} />
            Bug detected
          </motion.div>
        ) : (
          <div
            className="flex items-center gap-1 px-2 py-0.5 rounded"
            style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)', color: 'var(--text-dim)', fontSize: 11 }}
          >
            <Search size={10} />
            Scanning…
          </div>
        )}
      </div>

      {/* Diff body */}
      <div className="overflow-x-auto">
        <div className="flex">
          {/* Line numbers */}
          <div
            className="flex-shrink-0 px-3 py-2.5 select-none text-right"
            style={{
              background: 'var(--bg-primary)',
              borderRight: '1px solid var(--border-subtle)',
              minWidth: 40,
            }}
          >
            {diffLines.map((_, i) => (
              <div key={i} style={{ lineHeight: '22px', color: 'var(--text-dim)', fontSize: 10, fontFamily: 'var(--font-mono)' }}>
                {i + 1}
              </div>
            ))}
          </div>

          {/* Diff content */}
          <div className="flex-1 py-2.5 px-3">
            {diffLines.map((line, i) => {
              const isBug = line.type === 'bug'
              const isRemove = line.type === 'remove'
              const isAdd = line.type === 'add'
              const isHeader = line.type === 'header'

              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.018, duration: 0.2 }}
                  className={`relative px-1 ${
                    isBug && bugDetected ? 'diff-bug' :
                    isBug ? 'diff-add' :
                    isRemove ? 'diff-remove' :
                    isAdd ? 'diff-add' : 'diff-neutral'
                  }`}
                  style={{ lineHeight: '22px' }}
                >
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    color: isBug && bugDetected ? 'var(--danger)' :
                           isBug ? 'var(--success)' :
                           isRemove ? 'var(--danger)' :
                           isAdd ? 'var(--success)' :
                           isHeader ? 'var(--text-dim)' :
                           'var(--text-secondary)',
                    fontWeight: isBug && bugDetected ? 500 : 400,
                  }}>
                    {line.text || '\u00A0'}
                  </span>
                </motion.div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div
        className="flex items-center gap-4 px-3 py-2"
        style={{ borderTop: '1px solid var(--border-subtle)', background: 'var(--bg-tertiary)' }}
      >
        <span style={{ fontSize: 11, color: 'var(--success)', fontFamily: 'var(--font-mono)' }}>+{addCount}</span>
        <span style={{ fontSize: 11, color: 'var(--danger)', fontFamily: 'var(--font-mono)' }}>−{removeCount}</span>
        <span style={{ fontSize: 11, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>{diffWorkerLabel}</span>
        {bugDetected && (
          <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="ml-auto"
            style={{ fontSize: 11, color: 'var(--danger)', fontWeight: 500 }}
          >
            Removes null guard
          </motion.span>
        )}
      </div>
    </motion.div>
  )
}


