import { create } from 'zustand'
import type { ChainInfo, HotspotResidue, RFD3Design, MPNNResult, RF3Result } from '@/types'
import type { RFD3Response, MPNNResponse, RF3Response, RunPipelineResponse } from '@/api'
import { getPipelineJob, runPipeline } from '@/api'

const AUTH_TOKEN_KEY = 'deepbinder_token'
const AUTH_USER_KEY = 'deepbinder_user'
const LEGACY_AUTH_PREFIX = 'ode' + 'sign'
const LEGACY_AUTH_TOKEN_KEY = `${LEGACY_AUTH_PREFIX}_token`
const LEGACY_AUTH_USER_KEY = `${LEGACY_AUTH_PREFIX}_user`

const getStoredItem = (key: string, legacyKey: string) => (
  sessionStorage.getItem(key)
  || localStorage.getItem(key)
  || localStorage.getItem(legacyKey)
)

export interface ResidueRange {
  chain: string;
  startResSeq: number;
  endResSeq: number;
}

interface AuthState {
  token: string | null;
  user: { id: string; username: string; email: string | null; role: string } | null;
  setAuth: (token: string, user: { id: string; username: string; email: string | null; role: string }) => void;
  clearAuth: () => void;
}

interface AppState {
  currentPage: string;
  setCurrentPage: (page: string) => void;

  token: string | null;
  user: { id: string; username: string; email: string | null; role: string } | null;
  setAuth: (token: string, user: { id: string; username: string; email: string | null; role: string }) => void;
  clearAuth: () => void;

  targetFile: File | null;
  setTargetFile: (file: File | null) => void;
  pdbContent: string;
  setPdbContent: (content: string) => void;

  chains: ChainInfo[];
  setChains: (chains: ChainInfo[]) => void;

  selectedRange: ResidueRange | null;
  setSelectedRange: (range: ResidueRange | null) => void;

  selectedHotspots: HotspotResidue[];
  setSelectedHotspots: (hotspots: HotspotResidue[]) => void;
  addHotspot: (hotspot: HotspotResidue) => void;
  removeHotspot: (chain: string, residue: number) => void;
  toggleHotspot: (chain: string, residue: number, score?: number) => void;

  predictedHotspots: HotspotResidue[];
  setPredictedHotspots: (hotspots: HotspotResidue[]) => void;
  removePredictedHotspot: (chain: string, residue: number) => void;
  removePredictedHotspotsInRange: (chain: string, startResSeq: number, endResSeq: number) => void;

  focusedResidue: { chain: string; resSeq: number } | null;
  setFocusedResidue: (residue: { chain: string; resSeq: number } | null) => void;

  hoveredResidue: { chain: string; resSeq: number } | null;
  setHoveredResidue: (residue: { chain: string; resSeq: number } | null) => void;

  hotspotInput: string;
  setHotspotInput: (input: string) => void;

  binderLength: number;
  setBinderLength: (len: number) => void;

  rfd3Config: {
    targetEntityType: string;
    targetStructure: string;
    hotspots: string;
    conditionAtoms: string;
    lengthMin: number;
    lengthMax: number;
    nBatches: number;
    diffusionBatchSize: number;
  };
  setRfd3Config: (config: Partial<AppState['rfd3Config']>) => void;

  currentExperimentId: string | null;
  setCurrentExperimentId: (id: string | null) => void;

  rfd3Results: RFD3Response | null;
  setRfd3Results: (results: RFD3Response | null) => void;

  proteinRfd3Run: {
    status: 'idle' | 'running' | 'completed' | 'failed';
    taskName: string | null;
    startedAt: string | null;
    error: string | null;
  };
  startProteinRfd3Run: (params: {
    pdb_content: string;
    target?: string;
    hotspots?: string[] | Array<{ chain: string; residue: number }>;
    binder_length: number;
    length_min?: number;
    length_max?: number;
    diffusion_batch_size: number;
    n_batches: number;
    task_name: string;
    target_filename?: string;
    chain_type?: string;
  }) => Promise<RunPipelineResponse | null>;

  mpnnResults: MPNNResponse | null;
  setMpnnResults: (results: MPNNResponse | null) => void;

  rf3Results: RF3Response | null;
  setRf3Results: (results: RF3Response | null) => void;

  isRunning: boolean;
  setIsRunning: (running: boolean) => void;

  jobHistory: Array<{id: string; name: string; status: string; time: string}>;
  addJob: (job: {id: string; name: string; status: string; time: string}) => void;

  resetAll: () => void;
}

const getInitialAuth = () => {
  try {
    const token = getStoredItem(AUTH_TOKEN_KEY, LEGACY_AUTH_TOKEN_KEY)
    const user = getStoredItem(AUTH_USER_KEY, LEGACY_AUTH_USER_KEY)

    if (token && !sessionStorage.getItem(AUTH_TOKEN_KEY)) {
      sessionStorage.setItem(AUTH_TOKEN_KEY, token)
    }
    if (user && !sessionStorage.getItem(AUTH_USER_KEY)) {
      sessionStorage.setItem(AUTH_USER_KEY, user)
    }

    return {
      token: token || null,
      user: user ? JSON.parse(user) : null,
    }
  } catch {
    return { token: null, user: null }
  }
}

const initialAuth = getInitialAuth()

const normalizePipelineMpnnResults = (result: MPNNResponse | any | null): MPNNResponse | null => {
  if (!result) return null
  if (Array.isArray(result.sequences)) {
    return {
      ...result,
      num_sequences: result.num_sequences ?? result.sequences.length,
    }
  }
  const sequences = (result.tasks ?? []).flatMap((task: any) => task.result?.sequences ?? [])
  return {
    ...result,
    sequences,
    num_sequences: sequences.length,
    first_sequence_pdb: result.first_sequence_pdb ?? sequences[0]?.pdb_content ?? '',
  }
}

const normalizePipelineRf3Results = (result: RF3Response | any | null): RF3Response | null => {
  if (!result) return null
  if (typeof result.avg_plddt === 'number') return result
  const successfulTasks = (result.tasks ?? []).filter((task: any) => task.success && task.result)
  const best = successfulTasks
    .map((task: any) => task.result)
    .sort((a: any, b: any) => {
      const plddtDiff = (b.avg_plddt ?? Number.NEGATIVE_INFINITY) - (a.avg_plddt ?? Number.NEGATIVE_INFINITY)
      if (plddtDiff !== 0) return plddtDiff
      return (a.rmsd ?? Number.POSITIVE_INFINITY) - (b.rmsd ?? Number.POSITIVE_INFINITY)
    })[0]
  if (!best) return result
  return {
    ...best,
    tasks: result.tasks,
    aggregate_summary: result.summary,
  } as RF3Response
}

const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

const isPipelineTerminal = (status: string) => status !== 'running' && status !== 'queued' && !status.startsWith('running_')

const applyPipelineResult = (
  result: RunPipelineResponse,
  params: { task_name: string },
  set: any,
  get: () => AppState,
) => {
  set({
    rfd3Results: result.rfd3_results ? { ...result.rfd3_results, experiment_id: result.experiment_id || undefined } : get().rfd3Results,
    mpnnResults: result.mpnn_results ? normalizePipelineMpnnResults(result.mpnn_results) : get().mpnnResults,
    rf3Results: result.rf3_results ? normalizePipelineRf3Results(result.rf3_results) : get().rf3Results,
  })
  if (result.experiment_id) {
    set({ currentExperimentId: result.experiment_id })
  }
  if (isPipelineTerminal(result.status) && result.experiment_id) {
    get().addJob({
      id: `rfd3_${result.experiment_id}`,
      name: params.task_name,
      status: result.status,
      time: new Date().toLocaleString('zh-CN'),
    })
  }
}

const waitForPipelineVisualResult = async (jobId: string): Promise<RunPipelineResponse> => {
  while (true) {
    const result = await getPipelineJob(jobId)
    if (result.rfd3_results || isPipelineTerminal(result.status)) {
      return result
    }
    await sleep(3000)
  }
}

const continuePipelinePolling = async (
  jobId: string,
  params: { task_name: string },
  set: any,
  get: () => AppState,
) => {
  while (true) {
    const result = await getPipelineJob(jobId)
    applyPipelineResult(result, params, set, get)
    if (result.status === 'failed') {
      throw new Error(result.error || `完整流水线运行失败: ${result.failed_step || 'unknown'}`)
    }
    if (isPipelineTerminal(result.status)) {
      set((state) => ({
        proteinRfd3Run: {
          ...state.proteinRfd3Run,
          status: 'completed',
          error: null,
        },
      }))
      return
    }
    await sleep(3000)
  }
}

const initialState = {
  currentPage: '新建设计',
  ...initialAuth,
  targetFile: null,
  pdbContent: '',
  chains: [],
  selectedRange: null as ResidueRange | null,
  selectedHotspots: [] as HotspotResidue[],
  predictedHotspots: [] as HotspotResidue[],
  focusedResidue: null as { chain: string; resSeq: number } | null,
  hoveredResidue: null as { chain: string; resSeq: number } | null,
  hotspotInput: '',
  binderLength: 80,
  rfd3Config: {
    targetEntityType: '蛋白',
    targetStructure: '',
    hotspots: '',
    conditionAtoms: '',
    lengthMin: 40,
    lengthMax: 120,
    nBatches: 2,
    diffusionBatchSize: 2,
  },
  currentExperimentId: null as string | null,
  rfd3Results: null as RFD3Response | null,
  proteinRfd3Run: {
    status: 'idle' as const,
    taskName: null as string | null,
    startedAt: null as string | null,
    error: null as string | null,
  },
  mpnnResults: null as MPNNResponse | null,
  rf3Results: null as RF3Response | null,
  isRunning: false,
  jobHistory: [] as Array<{id: string; name: string; status: string; time: string}>,
}

export const useAppStore = create<AppState>((set, get) => ({
  ...initialState,

  setCurrentPage: (page) => set({ currentPage: page }),

  setAuth: (token, user) => {
    sessionStorage.setItem(AUTH_TOKEN_KEY, token)
    sessionStorage.setItem(AUTH_USER_KEY, JSON.stringify(user))
    localStorage.removeItem(AUTH_TOKEN_KEY)
    localStorage.removeItem(AUTH_USER_KEY)
    localStorage.removeItem(LEGACY_AUTH_TOKEN_KEY)
    localStorage.removeItem(LEGACY_AUTH_USER_KEY)
    set({ token, user })
  },
  clearAuth: () => {
    sessionStorage.removeItem(AUTH_TOKEN_KEY)
    sessionStorage.removeItem(AUTH_USER_KEY)
    localStorage.removeItem(AUTH_TOKEN_KEY)
    localStorage.removeItem(AUTH_USER_KEY)
    localStorage.removeItem(LEGACY_AUTH_TOKEN_KEY)
    localStorage.removeItem(LEGACY_AUTH_USER_KEY)
    set({ token: null, user: null })
  },

  setTargetFile: (file) => set({ targetFile: file }),
  setPdbContent: (content) => set({ pdbContent: content }),
  setChains: (chains) => set({ chains }),

  setSelectedRange: (range) => set({ selectedRange: range }),

  setSelectedHotspots: (hotspots) => {
    const input = hotspots.map(h => `${h.chain}/${h.residue}`).join(', ')
    set({ selectedHotspots: hotspots, hotspotInput: input })
  },

  addHotspot: (hotspot) => {
    const current = get().selectedHotspots
    if (!current.find(h => h.chain === hotspot.chain && h.residue === hotspot.residue)) {
      const updated = [...current, hotspot]
      const input = updated.map(h => `${h.chain}/${h.residue}`).join(', ')
      set({ selectedHotspots: updated, hotspotInput: input })
    }
  },

  removeHotspot: (chain, residue) => {
    const updated = get().selectedHotspots.filter(h => !(h.chain === chain && h.residue === residue))
    const input = updated.map(h => `${h.chain}/${h.residue}`).join(', ')
    set({ selectedHotspots: updated, hotspotInput: input })
  },

  toggleHotspot: (chain, residue, score) => {
    const current = get().selectedHotspots
    const exists = current.find(h => h.chain === chain && h.residue === residue)
    let updated: HotspotResidue[]
    if (exists) {
      updated = current.filter(h => !(h.chain === chain && h.residue === residue))
    } else {
      updated = [...current, { chain, residue, score: score ?? 0 }]
    }
    const input = updated.map(h => `${h.chain}/${h.residue}`).join(', ')
    set({ selectedHotspots: updated, hotspotInput: input })
  },

  setPredictedHotspots: (hotspots) => set({ predictedHotspots: hotspots }),

  removePredictedHotspot: (chain, residue) => {
    const predictedHotspots = get().predictedHotspots.filter(h => !(h.chain === chain && h.residue === residue))
    set({ predictedHotspots })
  },
  removePredictedHotspotsInRange: (chain, startResSeq, endResSeq) => {
    const min = Math.min(startResSeq, endResSeq)
    const max = Math.max(startResSeq, endResSeq)
    const predictedHotspots = get().predictedHotspots.filter(
      h => h.chain !== chain || h.residue < min || h.residue > max,
    )
    set({ predictedHotspots })
  },

  setFocusedResidue: (residue) => set({ focusedResidue: residue }),
  setHoveredResidue: (residue) => set({ hoveredResidue: residue }),

  setHotspotInput: (input) => set({ hotspotInput: input }),
  setBinderLength: (len) => set({ binderLength: len }),
  setRfd3Config: (config) => set((state) => ({ rfd3Config: { ...state.rfd3Config, ...config } })),
  setCurrentExperimentId: (id) => set({ currentExperimentId: id }),
  setRfd3Results: (results) => set({ rfd3Results: results }),
  startProteinRfd3Run: async (params) => {
    if (get().proteinRfd3Run.status === 'running') {
      return null
    }
    set({
      proteinRfd3Run: {
        status: 'running',
        taskName: params.task_name,
        startedAt: new Date().toISOString(),
        error: null,
      },
      rfd3Results: null,
      mpnnResults: null,
      rf3Results: null,
    })
    try {
      const hotspotPayload = (params.hotspots || []).map((item) => {
        if (typeof item !== 'string') return item
        const match = item.match(/^([A-Za-z0-9]+)(-?\d+)$/)
        return {
          chain: match?.[1] || item.replace(/[0-9-]/g, ''),
          residue: Number(match?.[2] || 0),
        }
      })
      const result: RunPipelineResponse = await runPipeline({
        pdb_content: params.pdb_content,
        target: params.target,
        hotspots: hotspotPayload,
        binder_length: params.binder_length,
        length_min: params.length_min,
        length_max: params.length_max,
        diffusion_batch_size: params.diffusion_batch_size,
        n_batches: params.n_batches,
        task_name: params.task_name,
        target_filename: params.target_filename,
        chain_type: params.chain_type,
        async_mode: true,
      })
      const visualResult = result.rfd3_results || isPipelineTerminal(result.status)
        ? result
        : await waitForPipelineVisualResult(result.job_id)
      if (visualResult.status === 'failed') {
        throw new Error(visualResult.error || `完整流水线运行失败: ${visualResult.failed_step || 'unknown'}`)
      }
      applyPipelineResult(visualResult, params, set, get)
      if (!isPipelineTerminal(visualResult.status)) {
        continuePipelinePolling(result.job_id, params, set, get).catch((err) => {
          const message = err instanceof Error ? err.message : String(err)
          set((state) => ({
            proteinRfd3Run: {
              ...state.proteinRfd3Run,
              status: 'failed',
              error: message,
            },
          }))
        })
        return visualResult
      }
      const finalResult = visualResult
      if (finalResult.status === 'failed') {
        throw new Error(finalResult.error || `完整流水线运行失败: ${finalResult.failed_step || 'unknown'}`)
      }
      set((state) => ({
        proteinRfd3Run: {
          ...state.proteinRfd3Run,
          status: 'completed',
          error: null,
        },
      }))
      return finalResult
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      set((state) => ({
        proteinRfd3Run: {
          ...state.proteinRfd3Run,
          status: 'failed',
          error: message,
        },
      }))
      throw err
    }
  },
  setMpnnResults: (results) => set({ mpnnResults: results }),
  setRf3Results: (results) => set({ rf3Results: results }),
  setIsRunning: (running) => set({ isRunning: running }),

  addJob: (job) => set((state) => ({ jobHistory: [job, ...state.jobHistory].slice(0, 20) })),

  resetAll: () => set({ ...initialState, ...getInitialAuth() }),
}))
