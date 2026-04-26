// types/index.ts

export type Mode = 'baseline' | 'trained'

export type WorkerStatus = 'honest' | 'suspicious'
export type OverseerAction = 'FLAG' | 'PASS'

export interface Worker {
  id: string
  label: string
  status: WorkerStatus
  patch: string
  cwe?: string
}

export interface OverseerState {
  id: string
  label: string
  specialty: string
  reasoning: string
  suspicion: number
  action: OverseerAction
}

export interface Vote {
  overseerId: string
  action: OverseerAction
}

export interface RewardPoint {
  step: number
  reward: number
}

export type DiffLineType = 'header' | 'add' | 'remove' | 'bug' | 'neutral'

export interface DiffLine {
  type: DiffLineType
  text: string
}

export interface AppState {
  // UI mode
  mode: Mode
  // Backend connection status
  isConnected: boolean
  connectionError: string | null

  // Current episode data
  workers: Worker[]
  overseers: OverseerState[]
  votes: Vote[]
  finalDecision: string
  rewardHistory: RewardPoint[]
  f1Score: number
  diffLines: DiffLine[]
  diffWorkerLabel: string

  // Episode metadata
  episodeId: string | null
  turn: number
  maliciousWorkers: string[]

  // UI state
  isAnimating: boolean
  bugDetected: boolean
  isLoadingBackend: boolean

  // Actions
  setMode: (mode: Mode) => void
  triggerAnimation: () => void
  refreshFromBackend: () => Promise<void>
}