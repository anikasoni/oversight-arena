'use client'
import { useState } from 'react'

type NodeId = 'workers' | 'patches' | 'overseer' | 'vote' | 'f1' | null

const NODES: { id: NodeId; label: string; sub: string }[] = [
  { id: 'workers',  label: 'Workers',        sub: '3 AI agents'       },
  { id: 'patches',  label: 'Code Patches',   sub: 'malicious + safe'  },
  { id: 'overseer', label: 'Overseer Panel', sub: '3 specialists'     },
  { id: 'vote',     label: 'Majority Vote',  sub: '≥2 FLAG → action'  },
  { id: 'f1',       label: 'F1 Curve',       sub: 'reward signal'     },
]

export default function ArchitectureFlow() {
  const [hovered, setHovered] = useState<NodeId>(null)

  // Each node: cx (center x in SVG viewbox 0→800), cy center y
  const nodePositions = [
    { id: 'workers',  cx: 80,  cy: 60 },
    { id: 'patches',  cx: 240, cy: 60 },
    { id: 'overseer', cx: 400, cy: 60 },
    { id: 'vote',     cx: 560, cy: 60 },
    { id: 'f1',       cx: 720, cy: 60 },
  ]

  // Arrow paths between nodes (simple horizontal lines)
  const arrows = [
    { x1: 140, x2: 178, y: 60 },
    { x1: 300, x2: 338, y: 60 },
    { x1: 460, x2: 498, y: 60 },
    { x1: 620, x2: 658, y: 60 },
  ]

  const isActive = (id: NodeId) => hovered === null || hovered === id
  const BOX_W = 120
  const BOX_H = 52

  return (
    <div
      style={{
        marginTop: 40,
        padding: '32px 28px 24px',
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: 14,
        boxShadow: 'var(--shadow-sm)',
      }}
    >
      <p style={{
        fontSize: 11, fontWeight: 600, textTransform: 'uppercase',
        letterSpacing: '0.08em', color: 'var(--text-4)', marginBottom: 20,
      }}>
        System Architecture
      </p>

      {/* SVG diagram */}
      <svg
        viewBox="0 0 800 120"
        style={{ width: '100%', height: 'auto', display: 'block', overflow: 'visible' }}
        aria-label="Architecture flow diagram"
      >
        <defs>
          <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 z" fill="var(--border-2)" />
          </marker>
          <marker id="arrow-dim" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 z" fill="var(--border)" />
          </marker>
        </defs>

        {/* Connector arrows */}
        {arrows.map(({ x1, x2, y }, i) => {
          const leftId = nodePositions[i].id as NodeId
          const rightId = nodePositions[i + 1].id as NodeId
          const active = hovered === null || hovered === leftId || hovered === rightId
          return (
            <line
              key={i}
              x1={x1} y1={y} x2={x2} y2={y}
              stroke={active ? 'var(--border-2)' : 'var(--border)'}
              strokeWidth={active ? 1.5 : 1}
              markerEnd={active ? 'url(#arrow)' : 'url(#arrow-dim)'}
              style={{ transition: 'stroke 0.3s, stroke-width 0.3s' }}
            />
          )
        })}

        {/* Nodes */}
        {nodePositions.map(({ id, cx, cy }) => {
          const node = NODES.find(n => n.id === id)!
          const active = isActive(id as NodeId)
          const isOverseer = id === 'overseer'

          return (
            <g
              key={id}
              onMouseEnter={() => setHovered(id as NodeId)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: 'default' }}
            >
              {/* Glow ring for overseer on hover */}
              {isOverseer && hovered === 'overseer' && (
                <rect
                  x={cx - BOX_W / 2 - 4}
                  y={cy - BOX_H / 2 - 4}
                  width={BOX_W + 8}
                  height={BOX_H + 8}
                  rx={10}
                  fill="none"
                  stroke="var(--accent)"
                  strokeWidth="1.5"
                  opacity="0.35"
                  style={{ transition: 'opacity 0.3s' }}
                />
              )}

              {/* Card box */}
              <rect
                x={cx - BOX_W / 2}
                y={cy - BOX_H / 2}
                width={BOX_W}
                height={BOX_H}
                rx={8}
                fill={
                  !active
                    ? 'var(--surface)'
                    : isOverseer && hovered === 'overseer'
                    ? 'rgba(29,106,232,0.07)'
                    : 'var(--bg-card)'
                }
                stroke={
                  !active
                    ? 'var(--border)'
                    : isOverseer && hovered === 'overseer'
                    ? 'var(--accent)'
                    : 'var(--border-2)'
                }
                strokeWidth={active ? 1.5 : 1}
                style={{ transition: 'fill 0.35s, stroke 0.35s, opacity 0.35s' }}
                opacity={active ? 1 : 0.4}
              />

              {/* Label */}
              <text
                x={cx}
                y={cy - 7}
                textAnchor="middle"
                style={{
                  fontFamily: 'Instrument Sans, sans-serif',
                  fontSize: 12,
                  fontWeight: 600,
                  fill: active
                    ? isOverseer && hovered === 'overseer'
                      ? 'var(--accent)'
                      : 'var(--text-1)'
                    : 'var(--text-4)',
                  transition: 'fill 0.35s',
                  letterSpacing: '-0.01em',
                }}
              >
                {node.label}
              </text>

              {/* Sub-label */}
              <text
                x={cx}
                y={cy + 11}
                textAnchor="middle"
                style={{
                  fontFamily: 'DM Mono, monospace',
                  fontSize: 9.5,
                  fill: active ? 'var(--text-3)' : 'var(--text-4)',
                  transition: 'fill 0.35s',
                }}
                opacity={active ? 1 : 0.5}
              >
                {node.sub}
              </text>
            </g>
          )
        })}
      </svg>

      {/* Tooltip hint */}
      <p style={{ fontSize: 11, color: 'var(--text-4)', marginTop: 14, textAlign: 'center', letterSpacing: '-0.01em' }}>
        Hover a node to highlight it in context
      </p>
    </div>
  )
}

