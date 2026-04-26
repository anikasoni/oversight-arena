/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        bg: '#0A0A0A',
        card: '#111111',
        border: '#1F1F1F',
        worker: '#F59E0B',
        overseer: '#3B82F6',
        reward: '#22C55E',
        danger: '#EF4444',
        muted: '#A1A1AA',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'scan': 'scan 2s linear infinite',
      },
      keyframes: {
        glow: {
          from: { boxShadow: '0 0 5px rgba(59,130,246,0.3)' },
          to: { boxShadow: '0 0 20px rgba(59,130,246,0.8)' },
        },
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
      },
      backgroundImage: {
        'grid-pattern': "linear-gradient(rgba(31,31,31,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(31,31,31,0.5) 1px, transparent 1px)",
      },
      backgroundSize: {
        'grid': '40px 40px',
      },
    },
  },
  plugins: [],
}