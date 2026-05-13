import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { useAppStore } from '@/store/useAppStore'

vi.mock('@/components/MolstarViewer', () => ({
  MolstarViewer: () => <div data-testid="molstar-viewer">Molstar viewer mock</div>,
}))

vi.mock('@/components/SilentStructureViewer', () => ({
  SilentStructureViewer: ({
    pdbContent,
    secondPdbContent,
    emptyText = '暂无结构数据',
  }: {
    pdbContent?: string
    secondPdbContent?: string
    emptyText?: string
  }) => (
    <div data-testid="silent-structure-viewer">
      {pdbContent ? `PDB:${pdbContent}${secondPdbContent ? `|SECOND:${secondPdbContent}` : ''}` : emptyText}
    </div>
  ),
}))

const mockAuthLogin = vi.fn()
const mockAuthRegister = vi.fn()
const mockRunRFD3 = vi.fn()
const mockRunDeNovoRFD3 = vi.fn()
const mockRunMPNN = vi.fn()
const mockRunRF3 = vi.fn()
const mockRunPipeline = vi.fn()
const mockGetExperiments = vi.fn()
const mockGetExperiment = vi.fn()
const mockGetUsers = vi.fn()

vi.mock('@/api', async () => {
  const actual = await vi.importActual<typeof import('@/api')>('@/api')
  return {
    ...actual,
    authLogin: (...args: unknown[]) => mockAuthLogin(...args),
    authRegister: (...args: unknown[]) => mockAuthRegister(...args),
    runRFD3: (...args: unknown[]) => mockRunRFD3(...args),
    runDeNovoRFD3: (...args: unknown[]) => mockRunDeNovoRFD3(...args),
    runMPNN: (...args: unknown[]) => mockRunMPNN(...args),
    runRF3: (...args: unknown[]) => mockRunRF3(...args),
    runPipeline: (...args: unknown[]) => mockRunPipeline(...args),
    getExperiments: (...args: unknown[]) => mockGetExperiments(...args),
    getExperiment: (...args: unknown[]) => mockGetExperiment(...args),
    getUsers: (...args: unknown[]) => mockGetUsers(...args),
  }
})

const researcher = {
  id: 'u-researcher',
  username: 'researcher',
  email: 'researcher@test.local',
  role: 'researcher',
}

const admin = {
  id: 'u-admin',
  username: 'admin',
  email: 'admin@test.local',
  role: 'admin',
}

beforeEach(() => {
  useAppStore.getState().resetAll()
  mockAuthLogin.mockReset()
  mockAuthRegister.mockReset()
  mockRunRFD3.mockReset()
  mockRunDeNovoRFD3.mockReset()
  mockRunMPNN.mockReset()
  mockRunRF3.mockReset()
  mockRunPipeline.mockReset()
  mockGetExperiments.mockResolvedValue({ total: 0, items: [] })
  mockGetExperiment.mockReset()
  mockGetUsers.mockResolvedValue({ users: [] })
})

describe('system acceptance page flows', () => {
  it('shows login modal for anonymous users while keeping the new design page available behind it', () => {
    render(<App />)

    expect(screen.getByRole('heading', { name: '登录 DeepBinder' })).toBeInTheDocument()
    expect(screen.getByText('目标结构')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'de novo protein' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'protein to protein' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /De Novo Design/ })).not.toBeInTheDocument()
  })

  it('logs in a researcher and hides administrator-only system monitor', async () => {
    mockAuthLogin.mockResolvedValue({
      access_token: 'researcher-token',
      token_type: 'bearer',
      user: researcher,
    })

    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByLabelText(/用户名/), 'researcher')
    await user.type(screen.getByLabelText(/密码/), 'secret123')
    await user.click(screen.getByRole('button', { name: '登 录' }))

    await waitFor(() => {
      expect(screen.queryByRole('heading', { name: '登录 DeepBinder' })).not.toBeInTheDocument()
    })
    expect(screen.getByText('研究员')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /新建设计/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'de novo protein' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /系统监控/ })).not.toBeInTheDocument()
  })

  it('renders backend auth error responses as readable messages', async () => {
    mockAuthLogin.mockRejectedValue({
      response: {
        data: {
          code: 'validation_error',
          message: '请求参数校验失败',
          details: [
            { loc: ['body', 'username'], msg: '用户名不能为空' },
            { loc: ['body', 'password'], msg: '密码至少需要 8 位' },
          ],
        },
      },
    })

    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByLabelText(/用户名/), 'a')
    await user.type(screen.getByLabelText(/密码/), 'secret123')
    await user.click(screen.getByRole('button', { name: '登 录' }))

    expect(await screen.findByText(/请求参数校验失败/)).toBeInTheDocument()
    expect(screen.getByText(/用户名不能为空/)).toBeInTheDocument()
    expect(screen.getByText(/密码至少需要 8 位/)).toBeInTheDocument()
    expect(screen.queryByText(/\{"code"/)).not.toBeInTheDocument()
  })

  it('shows system monitor navigation for administrators', () => {
    useAppStore.getState().setAuth('admin-token', admin)

    render(<App />)

    expect(screen.getByText('管理员')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /系统监控/ })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /作业中心/ })).not.toBeInTheDocument()
    expect(screen.queryByText('DeepBinder v2.0')).not.toBeInTheDocument()
  })

  it('lets administrators open user management and view a user scoped experiment list', async () => {
    useAppStore.getState().setAuth('admin-token', admin)
    mockGetUsers.mockResolvedValue({
      users: [
        admin,
        {
          id: 'u-owner',
          username: 'owner',
          email: 'owner@test.local',
          role: 'researcher',
          created_at: '2026-05-10T12:00:00',
          last_login: '2026-05-11T12:00:00',
        },
      ],
    })
    mockGetExperiments.mockResolvedValue({
      total: 1,
      items: [
        {
          id: 'exp-owner',
          name: 'Owner experiment',
          status: 'completed',
          created_at: '2026-05-10T12:00:00',
          updated_at: '2026-05-10T12:10:00',
          target: null,
          hotspots: null,
          duration_seconds: 12.5,
          gpu_info: null,
          user_id: 'u-owner',
          user: {
            id: 'u-owner',
            username: 'owner',
            email: 'owner@test.local',
            role: 'researcher',
          },
          num_designs: 1,
        },
      ],
    })

    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /用户管理/ }))
    expect(await screen.findByText('用户列表')).toBeInTheDocument()
    expect(screen.getByText('owner')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /查看 owner 实验/ }))

    expect(await screen.findByText('Owner experiment')).toBeInTheDocument()
    expect(mockGetExperiments).toHaveBeenLastCalledWith({
      page: 1,
      page_size: 20,
      keyword: undefined,
      status: undefined,
      user_id: 'u-owner',
    })
    expect(screen.getByText('所属用户')).toBeInTheDocument()
    expect(screen.getByText('owner')).toBeInTheDocument()
  })

  it('removes protein-to-protein submit task entry points', () => {
    useAppStore.getState().setAuth('researcher-token', researcher)

    render(<App />)

    expect(screen.queryByRole('button', { name: /提交任务/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /作业中心/ })).not.toBeInTheDocument()
    expect(screen.queryByText('DeepBinder v2.0')).not.toBeInTheDocument()
  })

  it('runs the De Novo design task and persists MPNN/RF3 experiment results', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    mockRunDeNovoRFD3.mockResolvedValue({
      success: true,
      experiment_id: 'exp-denovo-preview',
      designs: [
        {
          index: 0,
          batch: 0,
          design_in_batch: 0,
          name: 'denovo_0',
          pdb_path: '',
          pdb_content: 'RFD3_PDB',
          plddt: 88.4,
        },
        {
          index: 1,
          batch: 0,
          design_in_batch: 1,
          name: 'denovo_1',
          pdb_path: '',
          pdb_content: 'RFD3_PDB_1',
          plddt: 87.2,
        },
        {
          index: 2,
          batch: 0,
          design_in_batch: 2,
          name: 'denovo_2',
          pdb_path: '',
          pdb_content: 'RFD3_PDB_2',
          plddt: 86.1,
        },
      ],
      batches: [
        {
          batch_idx: 0,
          num_structures: 3,
          designs: [
            {
              index: 0,
              batch: 0,
              design_in_batch: 0,
              name: 'denovo_0',
              pdb_path: '',
              pdb_content: 'RFD3_PDB',
              plddt: 88.4,
            },
            {
              index: 1,
              batch: 0,
              design_in_batch: 1,
              name: 'denovo_1',
              pdb_path: '',
              pdb_content: 'RFD3_PDB_1',
              plddt: 87.2,
            },
            {
              index: 2,
              batch: 0,
              design_in_batch: 2,
              name: 'denovo_2',
              pdb_path: '',
              pdb_content: 'RFD3_PDB_2',
              plddt: 86.1,
            },
          ],
        },
      ],
      num_batches: 1,
      num_designs: 3,
      first_backbone_pdb: 'RFD3_PDB',
      output_dir: '',
    })
    mockRunMPNN.mockResolvedValue({
      success: true,
      sequences: [
        {
          index: 0,
          name: 'seq_0',
          sequence: 'ACDEFGHIK',
          pdb_path: '',
          pdb_content: 'MPNN_PDB',
          score: -0.7,
        },
      ],
      num_sequences: 1,
      first_sequence_pdb: 'MPNN_PDB',
      output_dir: '',
    })
    mockRunRF3.mockResolvedValue({
      success: true,
      predicted_pdb: 'RF3_PDB',
      predicted_pdb_path: '',
      num_models: 1,
      summary: {
        chain_ptm: [0.8],
        overall_plddt: 91.2,
        overall_pde: 0.2,
        overall_pae: 1.3,
        ptm: 0.81,
        iptm: 0.78,
        has_clash: false,
        ranking_score: 0.85,
      },
      pae: null,
      plddt: [91.2],
      avg_plddt: 91.2,
      rmsd: 1.35,
      rmsd_interpretation: 'Excellent',
      per_res_rmsd: [1.0],
      passed: true,
      output_dir: '',
    })

    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: 'de novo protein' }))
    expect(screen.queryByRole('button', { name: /运行 RFD3/ })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /新建任务/ }))

    await screen.findByText('denovo_0')
    expect(screen.getByText('denovo_1')).toBeInTheDocument()
    expect(screen.queryByText('denovo_2')).not.toBeInTheDocument()
    expect(screen.getByText(/当前仅展示 2 个预览 design/)).toBeInTheDocument()
    expect(mockRunDeNovoRFD3).toHaveBeenCalledWith({
      length: 80,
      diffusion_batch_size: 2,
      n_batches: 1,
    })

    await user.click(screen.getByRole('button', { name: /送入 MPNN/ }))
    await screen.findByText('Sequence 1')
    expect(mockRunMPNN).toHaveBeenCalledWith({
      backbone_pdb_content: 'RFD3_PDB',
      batch_size: 10,
      experiment_id: 'exp-denovo-preview',
    })
    expect(screen.getByText(/ProteinMPNN 生成的是氨基酸序列/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /送入 RF3/ }))
    await screen.findByText('验证通过')
    expect(mockRunRF3).toHaveBeenCalledWith({
      mpnn_pdb_content: 'MPNN_PDB',
      rfd3_pdb_content: 'RFD3_PDB',
      example_id: 'denovo_design',
      experiment_id: 'exp-denovo-preview',
    })
    expect(screen.getByText('1.35 Å')).toBeInTheDocument()
  })

  it('switches between de novo and protein-to-protein design modes', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)

    const user = userEvent.setup()
    render(<App />)

    expect(screen.getByText('目标结构')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'de novo protein' }))
    expect(screen.getByRole('heading', { name: 'De Novo Protein Design' })).toBeInTheDocument()
    expect(screen.queryByText('目标结构')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'protein to protein' }))
    expect(screen.getByText('目标结构')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'De Novo Protein Design' })).not.toBeInTheDocument()
  })

  it('runs the complete protein-to-protein pipeline from the primary generate action', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    useAppStore.getState().setPdbContent('TARGET_PDB')
    useAppStore.getState().setRfd3Config({
      targetStructure: 'A/1-100',
      hotspots: 'A/42',
    })
    mockRunPipeline.mockResolvedValue({
      job_id: 'job-pipeline',
      experiment_id: 'exp-pipeline',
      status: 'completed',
      rfd3_results: {
        success: true,
        designs: [
          { index: 0, batch: 0, design_in_batch: 0, name: 'rfd3_0', pdb_path: '', pdb_content: 'RFD3_0', plddt: 80 },
          { index: 1, batch: 0, design_in_batch: 1, name: 'rfd3_1', pdb_path: '', pdb_content: 'RFD3_1', plddt: 81 },
          { index: 2, batch: 0, design_in_batch: 2, name: 'rfd3_2', pdb_path: '', pdb_content: 'RFD3_2', plddt: 82 },
        ],
        batches: [
          {
            batch_idx: 0,
            num_structures: 3,
            designs: [
              { index: 0, batch: 0, design_in_batch: 0, name: 'rfd3_0', pdb_path: '', pdb_content: 'RFD3_0', plddt: 80 },
              { index: 1, batch: 0, design_in_batch: 1, name: 'rfd3_1', pdb_path: '', pdb_content: 'RFD3_1', plddt: 81 },
              { index: 2, batch: 0, design_in_batch: 2, name: 'rfd3_2', pdb_path: '', pdb_content: 'RFD3_2', plddt: 82 },
            ],
          },
        ],
        num_batches: 1,
        num_designs: 3,
        first_backbone_pdb: 'RFD3_0',
        output_dir: '',
      },
      mpnn_results: {
        success: true,
        sequences: [
          { index: 0, name: 'seq_0', sequence: 'ACDEFGHIK', pdb_path: '', pdb_content: 'MPNN_PDB', score: -0.3 },
        ],
        num_sequences: 1,
        first_sequence_pdb: 'MPNN_PDB',
        output_dir: '',
      },
      rf3_results: {
        success: true,
        predicted_pdb: 'RF3_PDB',
        predicted_pdb_path: '',
        num_models: 1,
        summary: {
          chain_ptm: [0.8],
          overall_plddt: 91.2,
          overall_pde: 0.2,
          overall_pae: 1.3,
          ptm: 0.81,
          iptm: 0.78,
          has_clash: false,
          ranking_score: 0.85,
        },
        pae: null,
        plddt: [91.2],
        avg_plddt: 91.2,
        rmsd: 1.35,
        rmsd_interpretation: 'Excellent',
        per_res_rmsd: [1.0],
        passed: true,
        output_dir: '',
      },
    })

    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: '新建任务' }))
    await user.click(screen.getByRole('button', { name: '开始生成' }))
    await user.click(await screen.findByRole('button', { name: /① RFD3 骨架生成/ }))
    await screen.findByText('rfd3_0')
    expect(mockRunPipeline).toHaveBeenCalledWith(expect.objectContaining({
      pdb_content: 'TARGET_PDB',
      target: 'A/1-100',
      hotspots: [{ chain: 'A', residue: 42 }],
      binder_length: 80,
      length_min: 40,
      length_max: 120,
      diffusion_batch_size: 2,
      n_batches: 2,
      task_name: expect.stringMatching(/^Pipeline_/),
      chain_type: 'proteinChain',
    }))
    expect(mockRunRFD3).not.toHaveBeenCalled()
    expect(mockRunMPNN).not.toHaveBeenCalled()
    expect(mockRunRF3).not.toHaveBeenCalled()
    expect(screen.queryByText('rfd3_2')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /② MPNN 序列设计/ }))
    expect(await screen.findByText('Sequence 1')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /③ RF3 结构验证/ }))
    expect(await screen.findByText('验证通过 ✅')).toBeInTheDocument()
  })

  it('shows only two RFD3 preview designs after backend pipeline completion', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    useAppStore.getState().setPdbContent('TARGET_PDB')
    useAppStore.getState().setRfd3Config({
      targetStructure: 'A/1-100',
      hotspots: 'A/42',
    })
    mockRunPipeline.mockResolvedValue({
      job_id: 'job-preview',
      experiment_id: 'exp-preview',
      status: 'completed',
      rfd3_results: {
        success: true,
        designs: [
          { index: 0, batch: 0, design_in_batch: 0, name: 'rfd3_0', pdb_path: '', pdb_content: 'RFD3_0', plddt: 80 },
          { index: 1, batch: 0, design_in_batch: 1, name: 'rfd3_1', pdb_path: '', pdb_content: 'RFD3_1', plddt: 81 },
          { index: 2, batch: 0, design_in_batch: 2, name: 'rfd3_2', pdb_path: '', pdb_content: 'RFD3_2', plddt: 82 },
        ],
        batches: [
          {
            batch_idx: 0,
            num_structures: 3,
            designs: [
              { index: 0, batch: 0, design_in_batch: 0, name: 'rfd3_0', pdb_path: '', pdb_content: 'RFD3_0', plddt: 80 },
              { index: 1, batch: 0, design_in_batch: 1, name: 'rfd3_1', pdb_path: '', pdb_content: 'RFD3_1', plddt: 81 },
              { index: 2, batch: 0, design_in_batch: 2, name: 'rfd3_2', pdb_path: '', pdb_content: 'RFD3_2', plddt: 82 },
            ],
          },
        ],
        num_batches: 1,
        num_designs: 3,
        first_backbone_pdb: 'RFD3_0',
        output_dir: '',
      },
      mpnn_results: {
        success: true,
        sequences: [
          { index: 0, name: 'seq_0', sequence: 'ACDEFGHIK', pdb_path: '', pdb_content: 'MPNN_PDB', score: -0.3 },
        ],
        num_sequences: 1,
        first_sequence_pdb: 'MPNN_PDB',
        output_dir: '',
      },
      rf3_results: {
        success: true,
        predicted_pdb: 'RF3_PDB',
        predicted_pdb_path: '',
        num_models: 1,
        summary: {
          chain_ptm: [0.8],
          overall_plddt: 91.2,
          overall_pde: 0.2,
          overall_pae: 1.3,
          ptm: 0.81,
          iptm: 0.78,
          has_clash: false,
          ranking_score: 0.85,
        },
        pae: null,
        plddt: [91.2],
        avg_plddt: 91.2,
        rmsd: 1.35,
        rmsd_interpretation: 'Excellent',
        per_res_rmsd: [1.0],
        passed: true,
        output_dir: '',
      },
    })

    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: '新建任务' }))
    await user.click(screen.getByRole('button', { name: '开始生成' }))
    await user.click(await screen.findByRole('button', { name: /① RFD3 骨架生成/ }))

    expect(await screen.findByText('rfd3_0')).toBeInTheDocument()
    expect(screen.getByText('rfd3_1')).toBeInTheDocument()
    expect(screen.queryByText('rfd3_2')).not.toBeInTheDocument()
    expect(screen.getByText(/当前仅展示 2 个预览 design/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /② MPNN 序列设计/ }))
    expect(await screen.findByText('Sequence 1')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /③ RF3 结构验证/ }))
    await screen.findByText('验证通过 ✅')
    expect(mockRunRFD3).not.toHaveBeenCalled()
    expect(mockRunMPNN).not.toHaveBeenCalled()
    expect(mockRunRF3).not.toHaveBeenCalled()
  })

  it('keeps manual protein-to-protein MPNN and RF3 clicks as preview-only process displays', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    useAppStore.getState().setPdbContent('TARGET_PDB')
    useAppStore.getState().setRfd3Config({
      targetStructure: 'A/1-100',
      hotspots: 'A/42',
    })
    mockRunPipeline.mockResolvedValue({
      job_id: 'job-preview-only',
      experiment_id: 'exp-preview-only',
      status: 'completed',
      rfd3_results: {
        success: true,
        designs: [
          { index: 0, batch: 0, design_in_batch: 0, name: 'rfd3_0', pdb_path: '', pdb_content: 'RFD3_0', plddt: 80 },
        ],
        batches: [
          {
            batch_idx: 0,
            num_structures: 1,
            designs: [
              { index: 0, batch: 0, design_in_batch: 0, name: 'rfd3_0', pdb_path: '', pdb_content: 'RFD3_0', plddt: 80 },
            ],
          },
        ],
        num_batches: 1,
        num_designs: 1,
        first_backbone_pdb: 'RFD3_0',
        output_dir: '',
      },
    })
    mockRunMPNN.mockResolvedValue({
      success: true,
      sequences: [
        { index: 0, name: 'seq_preview', sequence: 'ACDEFGHIK', pdb_path: '', pdb_content: 'MPNN_PREVIEW', score: -0.3 },
      ],
      num_sequences: 1,
      first_sequence_pdb: 'MPNN_PREVIEW',
      output_dir: '',
    })
    mockRunRF3.mockResolvedValue({
      success: true,
      predicted_pdb: 'RF3_PREVIEW',
      predicted_pdb_path: '',
      num_models: 1,
      summary: {
        chain_ptm: [0.8],
        overall_plddt: 91.2,
        overall_pde: 0.2,
        overall_pae: 1.3,
        ptm: 0.81,
        iptm: 0.78,
        has_clash: false,
        ranking_score: 0.85,
      },
      pae: null,
      plddt: [91.2],
      avg_plddt: 91.2,
      rmsd: 1.35,
      rmsd_interpretation: 'Excellent',
      per_res_rmsd: [1.0],
      passed: true,
      output_dir: '',
    })

    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: '新建任务' }))
    await user.click(screen.getByRole('button', { name: '开始生成' }))
    await user.click(await screen.findByRole('button', { name: /① RFD3 骨架生成/ }))
    await user.click(await screen.findByRole('button', { name: /送入MPNN/ }))

    expect(await screen.findByText('Sequence 1')).toBeInTheDocument()
    expect(mockRunMPNN).toHaveBeenCalledWith({
      backbone_pdb_content: 'RFD3_0',
      batch_size: 10,
      fixed_chains: ['A'],
      preview_only: true,
    })

    await user.click(screen.getByRole('button', { name: /送入RF3验证/ }))
    expect(await screen.findByText('验证通过 ✅')).toBeInTheDocument()
    expect(mockRunRF3).toHaveBeenCalledWith({
      mpnn_pdb_content: 'MPNN_PREVIEW',
      rfd3_pdb_content: 'RFD3_0',
      example_id: 'binder_design',
      preview_only: true,
    })
  })

  it('keeps protein-to-protein pipeline completion when navigating away during the run', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    useAppStore.getState().setPdbContent('TARGET_PDB')
    useAppStore.getState().setRfd3Config({
      targetStructure: 'A/1-100',
      hotspots: 'A/42',
    })
    let resolvePipeline: (value: any) => void = () => {}
    mockRunPipeline.mockReturnValue(new Promise((resolve) => { resolvePipeline = resolve }))

    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: '新建任务' }))
    await user.click(screen.getByRole('button', { name: '开始生成' }))
    expect(await screen.findByText('完整流水线正在运行...')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /实验记录/ }))
    await user.click(screen.getByRole('button', { name: /新建设计/ }))
    await user.click(screen.getByRole('button', { name: /① RFD3 骨架生成/ }))
    expect(await screen.findByText('完整流水线正在运行...')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /实验记录/ }))
    resolvePipeline({
      job_id: 'job-running',
      experiment_id: 'exp-running',
      status: 'completed',
      rfd3_results: {
        success: true,
        designs: [
          { index: 0, batch: 0, design_in_batch: 0, name: 'rfd3_done', pdb_path: '', pdb_content: 'RFD3_DONE', plddt: 80 },
        ],
        batches: [
          {
            batch_idx: 0,
            num_structures: 1,
            designs: [
              { index: 0, batch: 0, design_in_batch: 0, name: 'rfd3_done', pdb_path: '', pdb_content: 'RFD3_DONE', plddt: 80 },
            ],
          },
        ],
        num_batches: 1,
        num_designs: 1,
        first_backbone_pdb: 'RFD3_DONE',
        output_dir: '',
      },
    })

    await waitFor(() => {
      expect(useAppStore.getState().rfd3Results?.experiment_id).toBe('exp-running')
    })

    await user.click(screen.getByRole('button', { name: /新建设计/ }))
    await user.click(screen.getByRole('button', { name: /① RFD3 骨架生成/ }))

    expect(await screen.findByText('rfd3_done')).toBeInTheDocument()
    expect(screen.queryByText('完整流水线正在运行...')).not.toBeInTheDocument()
  })

  it('keeps predicted hotspot markers after selecting target range', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    useAppStore.getState().setChains([
      {
        chain_id: 'A',
        sequence: 'ACDEFGHIK',
        length: 9,
        resSeqs: [1, 2, 3, 4, 5, 6, 7, 8, 9],
      },
    ])
    useAppStore.getState().setPdbContent('TARGET_PDB')
    useAppStore.getState().setPredictedHotspots([{ chain: 'A', residue: 3, score: 0.91 }])

    render(<App />)

    expect(screen.getByTitle('A/3 · 预测热点')).toBeInTheDocument()
    const startResidue = screen.getByTitle('A/2 (先拖拽选择范围)')
    const endResidue = screen.getByTitle('A/5 (先拖拽选择范围)')
    fireEvent.mouseDown(startResidue)
    fireEvent.mouseEnter(endResidue)
    fireEvent.mouseMove(endResidue)
    fireEvent.mouseUp(endResidue)

    expect(useAppStore.getState().selectedRange).toEqual({
      chain: 'A',
      startResSeq: 2,
      endResSeq: 5,
    })
    expect(useAppStore.getState().predictedHotspots).toEqual([{ chain: 'A', residue: 3, score: 0.91 }])
    expect(screen.getByTitle('A/3 · 预测热点')).toBeInTheDocument()
  })

  it('loads experiment records through the experiments page', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    mockGetExperiments.mockResolvedValue({
      total: 1,
      items: [
        {
          id: 'exp-1',
          name: '验收实验记录',
          status: 'completed',
          created_at: '2026-05-10T12:00:00',
          updated_at: '2026-05-10T12:10:00',
          target: 'A/1-2',
          hotspots: [{ chain: 'A', residue: 1 }],
          duration_seconds: 12.5,
          gpu_info: 'A100',
          user_id: 'u-researcher',
          num_designs: 1,
        },
      ],
    })

    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /实验记录/ }))

    const table = await screen.findByRole('table')
    expect(within(table).getByText('任务类型')).toBeInTheDocument()
    expect(within(table).getByText('任务ID')).toBeInTheDocument()
    expect(within(table).getByText('任务名称')).toBeInTheDocument()
    expect(within(table).getByText('结束时间')).toBeInTheDocument()
    expect(within(table).getByText('验收实验记录')).toBeInTheDocument()
    expect(within(table).getByText('已完成')).toBeInTheDocument()
    expect(within(table).getByText('protein')).toBeInTheDocument()
    expect(within(table).getByText('exp-1')).toBeInTheDocument()
  })

  it('renders unavailable RF3 RMSD as not calculated in experiment detail', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    mockGetExperiment.mockResolvedValue({
      id: 'exp-rmsd',
      name: 'RMSD 缺失实验',
      status: 'completed',
      created_at: '2026-05-10T12:00:00',
      updated_at: '2026-05-10T12:10:00',
      input_pdb: 'ATOM',
      target: 'A/1-2',
      hotspots: [{ chain: 'A', residue: 1 }],
      rfd3_config: { task_type: 'protein', protein_chain: 'A/1-2' },
      mpnn_config: null,
      rf3_config: null,
      rfd3_results: null,
      mpnn_results: null,
      rf3_results: {
        success: true,
        avg_plddt: 82.1,
        rmsd: -1,
        rmsd_interpretation: 'N/A',
        passed: false,
        summary: { ptm: 0.6, iptm: 0.5 },
      },
      duration_seconds: 10,
      gpu_info: null,
      user_id: 'u-researcher',
      designs: [],
    })

    useAppStore.getState().setCurrentPage('experiment_exp-rmsd')
    render(<App />)

    expect(await screen.findByText('RMSD 缺失实验')).toBeInTheDocument()
    expect(screen.getByText('未计算')).toBeInTheDocument()
    expect(screen.getByText(/缺少参考结构或 CA 原子匹配失败/)).toBeInTheDocument()
    expect(screen.queryByText('-1.00 Å')).not.toBeInTheDocument()
  })

  it('shows fallback metric provenance in top confidence design cards', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    mockGetExperiment.mockResolvedValue({
      id: 'exp-provenance',
      name: '指标来源实验',
      status: 'completed',
      created_at: '2026-05-10T12:00:00',
      updated_at: '2026-05-10T12:10:00',
      input_pdb: 'ATOM',
      target: 'A/1-2',
      hotspots: [{ chain: 'A', residue: 1 }],
      rfd3_config: { task_type: 'protein', protein_chain: 'A/1-2' },
      mpnn_config: null,
      rf3_config: null,
      rfd3_results: null,
      mpnn_results: null,
      rf3_results: null,
      duration_seconds: 10,
      gpu_info: null,
      user_id: 'u-researcher',
      designs: [
        {
          id: 'd1',
          design_name: 'rf3_validated',
          sequence: 'ACDEFG',
          pdb_content: 'RF3_PDB',
          plddt: 91.2,
          rmsd: 1.2,
          ranking_score: 0.85,
          passed_validation: true,
          plddt_source: 'rf3',
          ranking_source: 'rf3',
          validation_status: 'validated',
        },
        {
          id: 'd2',
          design_name: 'rfd3_fallback',
          sequence: 'HIKLMN',
          pdb_content: 'RFD3_PDB',
          plddt: 82.4,
          rmsd: null,
          ranking_score: -1.2,
          passed_validation: false,
          plddt_source: 'rfd3',
          ranking_source: 'mpnn',
          validation_status: 'not_validated',
        },
        {
          id: 'd3',
          design_name: 'missing_metrics',
          sequence: 'PQRSTV',
          pdb_content: null,
          plddt: null,
          rmsd: null,
          ranking_score: null,
          passed_validation: false,
          plddt_source: 'none',
          ranking_source: 'none',
          validation_status: 'not_validated',
        },
      ],
    })

    useAppStore.getState().setCurrentPage('experiment_exp-provenance')
    const user = userEvent.setup()
    render(<App />)

    expect(await screen.findByText('指标来源实验')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /设计结果/ }))

    const topSection = screen.getByText('置信度最高的三个设计结果').closest('div')
    expect(topSection).not.toBeNull()
    expect(within(topSection as HTMLElement).getByText('rf3_validated')).toBeInTheDocument()
    expect(within(topSection as HTMLElement).getByText('rfd3_fallback')).toBeInTheDocument()
    expect(within(topSection as HTMLElement).getByText('pLDDT 来源: RF3 验证')).toBeInTheDocument()
    expect(within(topSection as HTMLElement).getByText('pLDDT 来源: RFD3 预筛选')).toBeInTheDocument()
    expect(within(topSection as HTMLElement).getByText('Ranking 来源: MPNN score')).toBeInTheDocument()
    expect(within(topSection as HTMLElement).getAllByText('未验证')[0]).toBeInTheDocument()

    const table = screen.getByRole('table')
    expect(within(table).getByText('指标来源')).toBeInTheDocument()
    expect(within(table).getByText('验证状态')).toBeInTheDocument()
    expect(within(table).getByText('RFD3 预筛选 / MPNN score')).toBeInTheDocument()
  })

  it('shows a login-required message instead of not-found when experiment detail returns 401', async () => {
    useAppStore.getState().setAuth('expired-token', researcher)
    mockGetExperiment.mockRejectedValue({
      response: {
        status: 401,
        data: { message: '请先登录' },
      },
    })

    useAppStore.getState().setCurrentPage('experiment_exp-auth')
    render(<App />)

    expect(await screen.findByText('请先登录后查看实验记录')).toBeInTheDocument()
    expect(screen.getByText('当前登录已失效，请重新登录后再试。')).toBeInTheDocument()
    expect(screen.queryByText('实验记录不存在')).not.toBeInTheDocument()
    expect(useAppStore.getState().token).toBeNull()
    expect(screen.getByRole('heading', { name: '登录 DeepBinder' })).toBeInTheDocument()
  })

  it('documents roles and ProteinMPNN visualization limits in help page', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)

    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /帮助/ }))

    expect(screen.getByText('用户角色与功能')).toBeInTheDocument()
    expect(screen.getByText('研究人员')).toBeInTheDocument()
    expect(screen.getByText('管理员')).toBeInTheDocument()
    expect(screen.getByText('MPNN可视化')).toBeInTheDocument()
    expect(screen.getByText(/ProteinMPNN负责序列设计/)).toBeInTheDocument()
  })
})
