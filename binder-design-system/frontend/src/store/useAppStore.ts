import { create } from 'zustand'
import type { ChainInfo, HotspotResidue, RFD3Design, MPNNResult, RF3Result } from '@/types'
import type { RFD3Response, MPNNResponse, RF3Response } from '@/api'

export interface ResidueRange {
  chain: string;
  startResSeq: number;
  endResSeq: number;
}

interface AppState {
  currentPage: string;
  setCurrentPage: (page: string) => void;

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

  rfd3Results: RFD3Response | null;
  setRfd3Results: (results: RFD3Response | null) => void;

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

const initialState = {
  currentPage: '新建设计',
  targetFile: null,
  pdbContent: '',
  chains: [],
  selectedRange: null as ResidueRange | null,
  selectedHotspots: [] as HotspotResidue[],
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
  rfd3Results: null as RFD3Response | null,
  mpnnResults: null as MPNNResponse | null,
  rf3Results: null as RF3Response | null,
  isRunning: false,
  jobHistory: [] as Array<{id: string; name: string; status: string; time: string}>,
}

export const useAppStore = create<AppState>((set, get) => ({
  ...initialState,

  setCurrentPage: (page) => set({ currentPage: page }),
  setTargetFile: (file) => set({ targetFile: file }),
  setPdbContent: (content) => set({ pdbContent: content }),
  setChains: (chains) => set({ chains }),

  setSelectedRange: (range) => {
    if (!range) {
      set({ selectedRange: null, selectedHotspots: [], hotspotInput: '' })
    } else {
      const hotspots = get().selectedHotspots.filter(h => h.chain !== range.chain)
      set({ selectedRange: range, selectedHotspots: hotspots, hotspotInput: hotspots.map(h => `${h.chain}/${h.residue}`).join(', ') })
    }
  },

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

  setFocusedResidue: (residue) => set({ focusedResidue: residue }),
  setHoveredResidue: (residue) => set({ hoveredResidue: residue }),

  setHotspotInput: (input) => set({ hotspotInput: input }),
  setBinderLength: (len) => set({ binderLength: len }),
  setRfd3Config: (config) => set((state) => ({ rfd3Config: { ...state.rfd3Config, ...config } })),
  setRfd3Results: (results) => set({ rfd3Results: results }),
  setMpnnResults: (results) => set({ mpnnResults: results }),
  setRf3Results: (results) => set({ rf3Results: results }),
  setIsRunning: (running) => set({ isRunning: running }),

  addJob: (job) => set((state) => ({ jobHistory: [job, ...state.jobHistory].slice(0, 20) })),

  resetAll: () => set(initialState),
}))
