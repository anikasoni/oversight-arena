/**
 * lib/mockData.ts
 *
 * DEPRECATED — all data now comes exclusively from the backend.
 * This file is kept as an empty stub so that any lingering import
 * doesn't break compilation, but no component should reference it.
 *
 * All mock exports return empty arrays / safe defaults.
 * arenaStore.ts no longer imports from this file.
 */

import { Worker, OverseerState, Vote, RewardPoint } from '@/types'

export const baselineWorkers: Worker[] = []
export const baselineOverseers: OverseerState[] = []
export const baselineVotes: Vote[] = []
export const baselineReward: RewardPoint[] = [{ step: 0, reward: 0 }]

export const trainedWorkers: Worker[] = []
export const trainedOverseers: OverseerState[] = []
export const trainedVotes: Vote[] = []
export const trainedReward: RewardPoint[] = [{ step: 0, reward: 0 }]

export const codeDiffLines: { type: string; text: string }[] = []