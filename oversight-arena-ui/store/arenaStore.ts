'use client'

import { create } from 'zustand'
import { AppState, Mode, DiffLine, DiffLineType } from '@/types'
import { apiReset, apiStep, apiGrader, apiHealth, BackendState } from '@/lib/api'
import {
  buildWorkers, buildOverseers, buildFinalDecision,
  buildDiffLines, buildRewardHistory, computeF1,
} from '@/lib/transform'

const VALID_TYPES: DiffLineType[] = ['header', 'add', 'remove', 'bug', 'neutral']

function castDiffLines(raw: { type: string; text: string }[]): DiffLine[] {
  return raw.map((l) => ({
    type: VALID_TYPES.includes(l.type as DiffLineType) ? (l.type as DiffLineType) : 'neutral',
    text: l.text,
  }))
}

const MODE_DIFFICULTY: Record<Mode, number> = { baseline: 0.2, trained: 0.6 }

// Empty BackendState used as a safe fallback when backend returns no state
const EMPTY_STATE: BackendState = {
  episode_id: '',
  step_count: 0,
  max_turns: 10,
  done: false,
  workers: ['W1', 'W2', 'W3'],
  malicious_workers: [],
  flagged_workers: [],
  rejected_workers: [],
  action_history: [],
  cumulative_reward: 0,
  terminal_reward: null,
  difficulty: 0.2,
  suspicion_log: [],
  worker_pattern_ids: [],
  malicious_pattern_ids: [],
  malicious_tier: '',
}

// Loading placeholder workers shown while fetching from backend
const LOADING_WORKERS = ['W1', 'W2', 'W3'].map((id, i) => ({
  id,
  label: `Worker ${i + 1}`,
  status: 'honest' as const,
  patch: '# Loading from backend…',
}))

// Loading placeholder overseers
const LOADING_OVERSEERS = ['A', 'B', 'C'].map((id, i) => ({
  id,
  label: `Overseer ${id}`,
  specialty: ['Precision', 'Speed', 'Recall'][i],
  reasoning: 'Awaiting episode data from backend…',
  suspicion: 0,
  action: 'PASS' as const,
}))

export const useArenaStore = create<AppState>((set, get) => ({
  mode: 'baseline',
  isConnected: false,
  connectionError: null,
  // Start with empty/loading state — no mock data
  workers: LOADING_WORKERS,
  overseers: LOADING_OVERSEERS,
  votes: [
    { overseerId: 'A', action: 'PASS' },
    { overseerId: 'B', action: 'PASS' },
    { overseerId: 'C', action: 'PASS' },
  ],
  finalDecision: 'Connecting to backend…',
  rewardHistory: [{ step: 0, reward: 0 }],
  f1Score: 0,
  diffLines: [{ type: 'neutral', text: 'Loading episode…' }],
  diffWorkerLabel: 'W?',
  episodeId: null,
  turn: 0,
  maliciousWorkers: [],
  isAnimating: false,
  bugDetected: false,
  isLoadingBackend: true,

  setMode: async (mode: Mode) => {
    set({ isAnimating: true, bugDetected: false, isLoadingBackend: true })
    await new Promise((r) => setTimeout(r, 400))
    const { isConnected } = get()

    if (isConnected) {
      try {
        const difficulty = MODE_DIFFICULTY[mode]

        // Reset the episode — backend now returns both observation AND state (FIX 6 in app.py)
        const resetResp = await apiReset(undefined, difficulty)
        const observation = resetResp.observation

        // state is now included in the reset response; fall back to EMPTY_STATE if missing
        const state: BackendState = resetResp.state ?? EMPTY_STATE

        // Build overseers from the live diff + malicious ground truth
        const { overseers, votes } = buildOverseers(observation, state.malicious_workers)
        const flagVotes = votes.filter((v) => v.action === 'FLAG')
        let updatedState: BackendState = state

        if (flagVotes.length >= 2) {
          const targetWorker = state.malicious_workers[0] ?? observation.workers[0]
          const stepResp = await apiStep({
            action: 'flag_worker',
            worker_id: targetWorker,
            reasoning: 'Majority vote from scripted 3-overseer panel.',
            cwe_tag: 'CWE-pattern',
          })
          updatedState = stepResp.state ?? state
        } else {
          const stepResp = await apiStep({ action: 'accept_all' })
          updatedState = stepResp.state ?? state
        }

        const grader = await apiGrader()

        const workers = buildWorkers(observation, updatedState.malicious_workers)
        const finalDec = buildFinalDecision(votes, updatedState.malicious_workers, updatedState.flagged_workers)
        const { lines, workerLabel } = buildDiffLines(observation, updatedState.malicious_workers)
        const f1 = computeF1(grader)
        const rewardHistory = buildRewardHistory(
          updatedState.step_count,
          updatedState.cumulative_reward,
          [{ step: 0, reward: 0 }],
        )

        set({
          mode,
          isConnected: true,
          connectionError: null,
          workers,
          overseers,
          votes,
          finalDecision: finalDec,
          rewardHistory,
          f1Score: f1,
          diffLines: lines,
          diffWorkerLabel: workerLabel,
          episodeId: updatedState.episode_id ?? null,
          turn: updatedState.step_count,
          maliciousWorkers: grader.malicious_workers,
          bugDetected: grader.success,
          isAnimating: false,
          isLoadingBackend: false,
        })
        return
      } catch (err) {
        console.error('[ArenaStore] Backend request failed:', err)
        set({
          isConnected: false,
          connectionError: 'Backend error — could not load episode',
          isAnimating: false,
          isLoadingBackend: false,
          finalDecision: 'Backend unavailable',
          workers: LOADING_WORKERS,
          overseers: LOADING_OVERSEERS,
          rewardHistory: [{ step: 0, reward: 0 }],
          f1Score: 0,
          diffLines: [{ type: 'neutral', text: 'Backend unavailable. Please refresh.' }],
          diffWorkerLabel: '—',
        })
        return
      }
    }

    // Backend not connected — show a clear offline state instead of mock data
    set({
      mode,
      isAnimating: false,
      isLoadingBackend: false,
      connectionError: 'Backend offline — connect to see live episode data',
      finalDecision: 'Backend offline',
      workers: LOADING_WORKERS,
      overseers: LOADING_OVERSEERS,
      votes: [
        { overseerId: 'A', action: 'PASS' },
        { overseerId: 'B', action: 'PASS' },
        { overseerId: 'C', action: 'PASS' },
      ],
      rewardHistory: [{ step: 0, reward: 0 }],
      f1Score: 0,
      diffLines: [{ type: 'neutral', text: 'Connect to backend to view live data.' }],
      diffWorkerLabel: '—',
      bugDetected: false,
    })
  },

  triggerAnimation: () => {
    set({ isAnimating: true })
    setTimeout(() => set({ isAnimating: false }), 1000)
  },

  refreshFromBackend: async () => {
    try {
      const grader = await apiGrader()
      set({
        f1Score: computeF1(grader),
        maliciousWorkers: grader.malicious_workers,
        bugDetected: grader.success,
      })
    } catch { /* silent */ }
  },
}))

if (typeof window !== 'undefined') {
  apiHealth()
    .then(() => {
      useArenaStore.setState({ isConnected: true, connectionError: null })
      // Trigger initial episode load from backend now that we're connected
      useArenaStore.getState().setMode('baseline')
    })
    .catch((err) => {
      console.warn('[ArenaStore] Backend unreachable:', err)
      useArenaStore.setState({
        isConnected: false,
        isLoadingBackend: false,
        connectionError: 'Backend offline — cannot load episode data',
        finalDecision: 'Backend offline',
        workers: LOADING_WORKERS,
        overseers: LOADING_OVERSEERS,
        rewardHistory: [{ step: 0, reward: 0 }],
        f1Score: 0,
        diffLines: [{ type: 'neutral', text: 'Backend offline. Check connection.' }],
        diffWorkerLabel: '—',
      })
    })
}