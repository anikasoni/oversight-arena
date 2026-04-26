/** @type {import('next').NextConfig} */

const backendUrl = process.env.BACKEND_URL ?? 'http://localhost:7860'

const nextConfig = {
  async rewrites() {
    const backendUrl =
      process.env.BACKEND_URL ?? 'https://anikasoni-oversight-arena.hf.space'
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/:path*`,
      },
    ]
  },
  images: {
    remotePatterns: [],
  },
}

module.exports = nextConfig