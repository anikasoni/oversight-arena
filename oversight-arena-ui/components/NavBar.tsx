'use client'
import Link from 'next/link'
import Image from 'next/image'
import { usePathname } from 'next/navigation'
import { useArenaStore } from '@/store/arenaStore'
import { useTheme } from '@/context/ThemeContext'
import { Sun, Moon } from 'lucide-react'

const links = [
  { href: '/',         label: 'Overview' },
  { href: '/workers',  label: 'Workers' },
  { href: '/analysis', label: 'Analysis' },
  { href: '/results',  label: 'Results' },
]

export default function NavBar() {
  const pathname = usePathname()
  const { isConnected, mode, setMode, isLoadingBackend } = useArenaStore()
  const { theme, toggle } = useTheme()

  return (
    <nav className="navbar">
      <div className="page-wide h-full flex items-center justify-between">

        {/* Logo — replaced Shield icon with actual logo image */}
        <Link href="/" style={{ display: 'flex', alignItems: 'center', gap: 9, textDecoration: 'none' }}>
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            overflow: 'hidden', flexShrink: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Image
              src="/logo(1).png"
              alt="Oversight Arena"
              width={28}
              height={28}
              style={{ objectFit: 'contain', width: '100%', height: '100%' }}
              priority
            />
          </div>
          <span style={{
            fontFamily: 'DM Serif Display, Georgia, serif',
            fontSize: 18, color: 'var(--text-1)',
            letterSpacing: '-0.02em', fontWeight: 400,
          }}>
            Oversight Arena
          </span>
        </Link>

        {/* Nav links — unchanged */}
        <div style={{ display: 'flex', alignItems: 'center' }}>
          {links.map(({ href, label }) => {
            const active = pathname === href
            return (
              <Link key={href} href={href} style={{
                padding: '0 14px', height: 52,
                display: 'flex', alignItems: 'center',
                fontSize: 14, fontWeight: active ? 600 : 400,
                color: active ? 'var(--text-1)' : 'var(--text-3)',
                textDecoration: 'none',
                borderBottom: active ? '2px solid var(--text-1)' : '2px solid transparent',
                transition: 'color 0.15s, border-color 0.15s',
                letterSpacing: '-0.01em',
              }}>
                {label}
              </Link>
            )
          })}
        </div>

        {/* Right side — unchanged */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {/* Connection status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="pulse-dot" style={{
              display: 'inline-block', width: 7, height: 7, borderRadius: '50%',
              background: isConnected ? 'var(--green)' : 'var(--text-4)',
            }} />
            <span style={{ fontSize: 12, color: 'var(--text-3)' }}>
              {isConnected ? 'Live' : 'Demo'}
            </span>
          </div>

          {/* Divider */}
          <div style={{ width: 1, height: 18, background: 'var(--border)' }} />

          {/* Mode toggle */}
          <div style={{
            display: 'flex', gap: 2, padding: '3px',
            background: 'var(--surface)', borderRadius: 8,
            border: '1px solid var(--border)',
          }}>
            {(['baseline', 'trained'] as const).map(m => (
              <button key={m} onClick={() => setMode(m)} disabled={isLoadingBackend}
                style={{
                  padding: '4px 12px', borderRadius: 6,
                  border: 'none', cursor: isLoadingBackend ? 'not-allowed' : 'pointer',
                  fontFamily: 'Instrument Sans, sans-serif',
                  fontSize: 12, fontWeight: 500,
                  color: mode === m ? 'var(--text-1)' : 'var(--text-3)',
                  background: mode === m ? 'var(--bg-card)' : 'transparent',
                  boxShadow: mode === m ? 'var(--shadow-sm)' : 'none',
                  transition: 'all 0.15s',
                  opacity: isLoadingBackend ? 0.5 : 1,
                  textTransform: 'capitalize',
                  letterSpacing: '-0.01em',
                }}>
                {m}
              </button>
            ))}
          </div>

          {/* Theme toggle */}
          <button onClick={toggle} style={{
            width: 32, height: 32, borderRadius: 8,
            border: '1px solid var(--border)',
            background: 'var(--surface)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer', color: 'var(--text-2)',
            transition: 'all 0.15s',
          }}>
            {theme === 'light'
              ? <Moon size={14} />
              : <Sun size={14} />
            }
          </button>
        </div>
      </div>
    </nav>
  )
}
