import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 600000,
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

export interface RFD3Design {
  index: number
  batch: number
  design_in_batch: number
  name: string
  pdb_path: string
  pdb_content: string
  plddt: number
  mock?: boolean
}

export interface RFD3Response {
  success: boolean
  designs: RFD3Design[]
  batches: Array<{ batch_idx: number; num_structures: number; designs: RFD3Design[] }>
  num_batches: number
  num_designs: number
  first_backbone_pdb: string
  output_dir: string
  mock?: boolean
}

export interface MPNNSequence {
  index: number
  name: string
  sequence: string
  pdb_path: string
  pdb_content: string
  score: number
  mock?: boolean
}

export interface MPNNResponse {
  success: boolean
  sequences: MPNNSequence[]
  num_sequences: number
  first_sequence_pdb: string
  output_dir: string
  mock?: boolean
}

export interface RunPipelineResponse {
  job_id: string
  status: string
  rfd3_results?: RFD3Response
  mpnn_results?: MPNNResponse
  rf3_results?: any
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

export const runRFD3 = async (params: {
  pdb_content: string
  target?: string
  hotspots?: string[]
  binder_length?: number
  length_min?: number
  length_max?: number
  diffusion_batch_size?: number
  n_batches?: number
}): Promise<RFD3Response> => {
  const res = await api.post('/run-rfd3', params)
  return res.data
}

export interface RF3Summary {
  chain_ptm: number[]
  overall_plddt: number
  overall_pde: number
  overall_pae: number
  ptm: number
  iptm: number
  has_clash: boolean
  ranking_score: number
}

export interface RF3Response {
  success: boolean
  predicted_pdb: string
  predicted_pdb_path: string
  num_models: number
  summary: RF3Summary
  pae: number[][] | null
  plddt: number[]
  avg_plddt: number
  rmsd: number
  rmsd_interpretation: string
  per_res_rmsd: number[]
  passed: boolean
  output_dir: string
  mock?: boolean
}

export const runRF3 = async (params: {
  mpnn_pdb_content: string
  rfd3_pdb_content?: string
  example_id?: string
}): Promise<RF3Response> => {
  const res = await api.post('/run-rf3', params)
  return res.data
}

export const runMPNN = async (params: {
  backbone_pdb_content?: string
  backbone_pdb_path?: string
  batch_size?: number
  fixed_chains?: string[]
  model_type?: string
}): Promise<MPNNResponse> => {
  const res = await api.post('/run-mpnn', params)
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
