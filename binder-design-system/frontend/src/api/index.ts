import axios from 'axios'

const AUTH_TOKEN_KEY = 'deepbinder_token'
const AUTH_USER_KEY = 'deepbinder_user'
const LEGACY_AUTH_PREFIX = 'ode' + 'sign'
const LEGACY_AUTH_TOKEN_KEY = `${LEGACY_AUTH_PREFIX}_token`
const LEGACY_AUTH_USER_KEY = `${LEGACY_AUTH_PREFIX}_user`

const api = axios.create({
  baseURL: '/api',
  timeout: 1800000,
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem(AUTH_TOKEN_KEY) || localStorage.getItem(LEGACY_AUTH_TOKEN_KEY)
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem(AUTH_TOKEN_KEY)
      localStorage.removeItem(AUTH_USER_KEY)
      localStorage.removeItem(LEGACY_AUTH_TOKEN_KEY)
      localStorage.removeItem(LEGACY_AUTH_USER_KEY)
    }
    return Promise.reject(err)
  }
)

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
  experiment_id: string
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
  experiment_id?: string
}): Promise<RFD3Response> => {
  const res = await api.post('/run-rfd3', params)
  return res.data
}

export const runDeNovoRFD3 = async (params: {
  length: number
  diffusion_batch_size?: number
  n_batches?: number
  experiment_id?: string
}): Promise<RFD3Response> => {
  const res = await api.post('/run-rfd3', {
    binder_length: params.length,
    diffusion_batch_size: params.diffusion_batch_size,
    n_batches: params.n_batches,
    experiment_id: params.experiment_id,
  })
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
  experiment_id?: string
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
  experiment_id?: string
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

export interface AuthResponse {
  access_token: string
  token_type: string
  user: { id: string; username: string; email: string | null; role: string }
}

export const authLogin = async (username: string, password: string): Promise<AuthResponse> => {
  const formData = new URLSearchParams()
  formData.append('username', username)
  formData.append('password', password)
  const res = await api.post('/auth/login', formData)
  return res.data
}

export const authRegister = async (username: string, password: string, email?: string) => {
  const res = await api.post('/auth/register', { username, password, email })
  return res.data
}

export const authMe = async () => {
  const res = await api.get('/auth/me')
  return res.data
}

export interface ExperimentItem {
  id: string
  name: string
  status: string
  created_at: string | null
  updated_at: string | null
  target: string | null
  hotspots: any[] | null
  duration_seconds: number | null
  gpu_info: string | null
  user_id: string | null
  num_designs: number
}

export interface ExperimentDetail extends ExperimentItem {
  input_pdb: string | null
  rfd3_config: any
  mpnn_config: any
  rf3_config: any
  rfd3_results: any
  mpnn_results: any
  rf3_results: any
  designs: Array<{
    id: string
    design_name: string | null
    sequence: string | null
    pdb_content: string | null
    plddt: number | null
    rmsd: number | null
    ranking_score: number | null
    passed_validation: boolean
  }>
}

export const getExperiments = async (params?: { page?: number; page_size?: number; status?: string; keyword?: string }) => {
  const res = await api.get('/experiments', { params })
  return res.data
}

export const getExperiment = async (id: string): Promise<ExperimentDetail> => {
  const res = await api.get(`/experiments/${id}`)
  return res.data
}

export const createExperiment = async (data: { name: string; input_pdb?: string; target?: string; hotspots?: any[]; rfd3_config?: any; mpnn_config?: any; rf3_config?: any }) => {
  const res = await api.post('/experiments', data)
  return res.data
}

export const updateExperiment = async (id: string, data: any) => {
  const res = await api.put(`/experiments/${id}`, data)
  return res.data
}

export const deleteExperiment = async (id: string) => {
  const res = await api.delete(`/experiments/${id}`)
  return res.data
}

export const exportExperiment = async (id: string) => {
  const res = await api.get(`/experiments/${id}/export`)
  return res.data
}

export const exportExperimentCsv = async (id: string) => {
  const res = await api.get(`/experiments/${id}/export/csv`, { responseType: 'blob' })
  return res.data as Blob
}

export const compareExperiments = async (ids: string[]) => {
  const res = await api.post('/experiments/compare', ids)
  return res.data
}

export const getMonitorStatus = async () => {
  const res = await api.get('/monitor/status')
  return res.data
}

export const getMonitorTasks = async () => {
  const res = await api.get('/monitor/tasks')
  return res.data
}

export default api
