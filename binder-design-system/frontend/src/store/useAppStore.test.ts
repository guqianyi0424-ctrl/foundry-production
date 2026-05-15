import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAppStore } from '@/store/useAppStore'
import * as api from '@/api'

describe('auth storage', () => {
  beforeEach(() => {
    useAppStore.getState().resetAll()
    localStorage.clear()
    sessionStorage.clear()
  })

  it('stores the active login in session storage so another tab can use a different user', () => {
    useAppStore.getState().setAuth('admin-token', {
      id: 'u-admin',
      username: 'admin',
      email: 'admin@test.local',
      role: 'admin',
    })

    expect(sessionStorage.getItem('deepbinder_token')).toBe('admin-token')
    expect(localStorage.getItem('deepbinder_token')).toBeNull()
  })
})

describe('pipeline run state', () => {
  beforeEach(() => {
    useAppStore.getState().resetAll()
    vi.restoreAllMocks()
  })

  it('tracks the backend pipeline stage while polling', async () => {
    vi.spyOn(api, 'runPipeline').mockResolvedValue({
      job_id: 'job-stage',
      experiment_id: null,
      status: 'running',
      stage: 'queued',
      rfd3_results: undefined,
      mpnn_results: undefined,
      rf3_results: undefined,
    })
    vi.spyOn(api, 'getPipelineJob')
      .mockResolvedValueOnce({
        job_id: 'job-stage',
        experiment_id: 'exp-stage',
        status: 'running_mpnn',
        stage: 'mpnn',
        rfd3_results: {
          success: true,
          experiment_id: 'exp-stage',
          designs: [],
          batches: [],
          num_batches: 1,
          num_designs: 0,
          first_backbone_pdb: 'ATOM',
          output_dir: '',
        },
        mpnn_results: undefined,
        rf3_results: undefined,
      })
      .mockResolvedValueOnce({
        job_id: 'job-stage',
        experiment_id: 'exp-stage',
        status: 'completed',
        stage: 'completed',
        rfd3_results: undefined,
        mpnn_results: undefined,
        rf3_results: undefined,
      })

    const result = await useAppStore.getState().startProteinRfd3Run({
      pdb_content: 'ATOM',
      hotspots: [{ chain: 'A', residue: 10 }],
      binder_length: 80,
      diffusion_batch_size: 1,
      n_batches: 1,
      task_name: 'Stage test',
    })

    expect(result?.stage).toBe('mpnn')
    expect(useAppStore.getState().proteinRfd3Run.jobId).toBe('job-stage')
    expect(useAppStore.getState().proteinRfd3Run.stage).toBe('completed')
  })
})
