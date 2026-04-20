import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 300000,
})

export interface UploadResponse {
  chains: Array<{ chain_id: string; sequence: string; length: number; resSeqs?: number[] }>
  pdb_content: string
}

export interface PredictHotspotResponse {
  hotspots: Array<{ chain: string; residue: number; residue_name: string; score: number; label: string }>
  num_hotspots: number
  method: string
  model_loaded: boolean
  total_residues: number
}

export interface RunPipelineResponse {
  job_id: string
  status: string
  rfd3_results?: {
    success: boolean
    designs: Array<{ index: number; rank: number; plddt: number; pdb_path: string }>
    mock?: boolean
  }
  mpnn_results?: Array<{ design_idx: number; sequence: string; score: number; seq_idx: number }>
  rf3_results?: Array<{
    design_idx: number; sequence: string; rmsd: number;
    avg_plddt: number; passed: boolean; plddt?: number[];
    pae?: number[][]; per_res_rmsd?: number[]; mock?: boolean
  }>
}

export const uploadPdb = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  const res = await api.post('/upload', formData)
  return res.data
}

export const predictHotspot = async (pdbContent: string): Promise<PredictHotspotResponse> => {
  const res = await api.post('/predict-hotspot', { pdb_content: pdbContent })
  return res.data
}

export const runPipeline = async (params: {
  pdb_content: string
  hotspots: Array<{ chain: string; residue: number }>
  binder_length: number
}): Promise<RunPipelineResponse> => {
  const res = await api.post('/run-pipeline', params)
  return res.data
}

export default api
