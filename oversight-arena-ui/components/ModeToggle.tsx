'use client'
import { motion } from 'framer-motion'
import { useArenaStore } from '@/store/arenaStore'
import { Zap, Brain, Wifi, WifiOff } from 'lucide-react'

export default function ModeToggle() {
  const { mode, setMode, isConnected, connectionError, isLoadingBackend } = useArenaStore()

  return (
    <div className="flex items-center gap-2">
      {/* Backend status */}
      <div
        className="hidden md:flex items-center gap-1.5 px-2 py-1 rounded-md"
        style={{
          background: isConnected ? 'var(--success-dim)' : 'var(--danger-dim)',
          border: `1px solid ${isConnected ? 'rgba(137,209,150,0.2)' : 'rgba(248,113,113,0.2)'}`,
        }}
        title={connectionError ?? 'Connected to backend'}
      >
        {isConnected
          ? <Wifi size={10} style={{ color: 'var(--success)' }} />
          : <WifiOff size={10} style={{ color: 'var(--danger)' }} />}
        <span style={{ color: isConnected ? 'var(--success)' : 'var(--danger)', fontSize: 11, fontWeight: 500 }}>
          {isConnected ? 'Backend' : 'Demo'}
        </span>
      </div>

      <span style={{ color: 'var(--text-dim)', fontSize: 11 }}>Mode</span>

      {/* Toggle */}
      <div
        className="relative flex p-0.5 rounded-md gap-0.5"
        style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-subtle)' }}
      >
        <motion.div
          className="absolute inset-0.5 rounded"
          animate={{ x: mode === 'trained' ? '50%' : '0%' }}
          transition={{ type: 'spring', stiffness: 350, damping: 30 }}
          style={{ width: 'calc(50% - 2px)', background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 4 }}
        />
        {[
          { id: 'baseline' as const, icon: Zap, label: 'Baseline' },
          { id: 'trained' as const, icon: Brain, label: 'Trained' },
        ].map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => setMode(id)}
            disabled={isLoadingBackend}
            className="relative z-10 flex items-center gap-1.5 px-3 py-1.5 rounded transition-colors disabled:opacity-40"
            style={{
              color: mode === id ? 'var(--text-primary)' : 'var(--text-secondary)',
              fontSize: 12,
              fontWeight: 500,
              letterSpacing: '-0.01em',
            }}
          >
            <Icon size={12} />
            {label}
          </button>
        ))}
      </div>

      {/* Training status pill */}
      <motion.div
        key={mode}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="hidden md:flex items-center gap-1.5 px-2 py-1 rounded-md"
        style={{
          background: mode === 'trained' ? 'var(--accent-dim)' : 'var(--bg-tertiary)',
          border: `1px solid ${mode === 'trained' ? 'rgba(143,164,255,0.2)' : 'var(--border-subtle)'}`,
        }}
      >
        <span
          className="w-1.5 h-1.5 rounded-full"
          style={{
            background: isLoadingBackend ? 'var(--warning)' : mode === 'trained' ? 'var(--accent)' : 'var(--text-dim)',
          }}
        />
        <span style={{
          color: mode === 'trained' ? 'var(--accent)' : 'var(--text-secondary)',
          fontSize: 11,
          fontWeight: 500,
        }}>
          {isLoadingBackend ? 'Loading…' : mode === 'trained' ? 'GRPO active' : 'No training'}
        </span>
      </motion.div>
    </div>
  )
}
