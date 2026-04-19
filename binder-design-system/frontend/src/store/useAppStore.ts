import { create } from 'zustand'
import type { ChainInfo, HotspotResidue, RFD3Design, MPNNResult, RF3Result } from '@/types'

interface AppState {
  currentPage: string;
  setCurrentPage: (page: string) => void;

  targetFile: File | null;
  setTargetFile: (file: File | null) => void;
  pdbContent: string;
  setPdbContent: (content: string) => void;

  chains: ChainInfo[];
  setChains: (chains: ChainInfo[]) => void;

  selectedHotspots: HotspotResidue[];
  setSelectedHotspots: (hotspots: HotspotResidue[]) => void;
  addHotspot: (hotspot: HotspotResidue) => void;
  removeHotspot: (chain: string, residue: number) => void;
  toggleHotspot: (chain: string, residue: number, score?: number) => void;

  hotspotInput: string;
  setHotspotInput: (input: string) => void;

  binderLength: number;
  setBinderLength: (len: number) => void;

  rfd3Results: RFD3Design[] | null;
  setRfd3Results: (results: RFD3Design[] | null) => void;

  mpnnResults: MPNNResult[] | null;
  setMpnnResults: (results: MPNNResult[] | null) => void;

  rf3Results: RF3Result[] | null;
  setRf3Results: (results: RF3Result[] | null) => void;

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
  selectedHotspots: [] as HotspotResidue[],
  hotspotInput: '',
  binderLength: 80,
  rfd3Results: null,
  mpnnResults: null,
  rf3Results: null,
  isRunning: false,
  jobHistory: [] as Array<{id: string; name: string; status: string; time: string}>,
}

export const useAppStore = create<AppState>((set, get) => ({
  ...initialState,

  setCurrentPage: (page) => set({ currentPage: page }),
  setTargetFile: (file) => set({ targetFile: file }),
  setPdbContent: (content) => set({ pdbContent: content }),
  setChains: (chains) => set({ chains }),

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

  setHotspotInput: (input) => set({ hotspotInput: input }),
  setBinderLength: (len) => set({ binderLength: len }),
  setRfd3Results: (results) => set({ rfd3Results: results }),
  setMpnnResults: (results) => set({ mpnnResults: results }),
  setRf3Results: (results) => set({ rf3Results: results }),
  setIsRunning: (running) => set({ isRunning: running }),

  addJob: (job) => set((state) => ({ jobHistory: [job, ...state.jobHistory].slice(0, 20) })),

  resetAll: () => set(initialState),
}))
