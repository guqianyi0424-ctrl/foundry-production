import { render, screen, waitFor, within } from '@testing-library/react'
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
const mockGetExperiments = vi.fn()
const mockGetExperiment = vi.fn()

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
    getExperiments: (...args: unknown[]) => mockGetExperiments(...args),
    getExperiment: (...args: unknown[]) => mockGetExperiment(...args),
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
  mockGetExperiments.mockResolvedValue({ total: 0, items: [] })
  mockGetExperiment.mockReset()
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
  })

  it('runs the De Novo RFD3 to MPNN to RF3 page flow with mocked APIs', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    mockRunDeNovoRFD3.mockResolvedValue({
      success: true,
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
      ],
      batches: [],
      num_batches: 1,
      num_designs: 1,
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
    await user.click(screen.getByRole('button', { name: /运行 RFD3/ }))

    await screen.findByText('denovo_0')
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
    })
    expect(screen.getByText(/ProteinMPNN 生成的是氨基酸序列/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /送入 RF3/ }))
    await screen.findByText('验证通过')
    expect(mockRunRF3).toHaveBeenCalledWith({
      mpnn_pdb_content: 'MPNN_PDB',
      rfd3_pdb_content: 'RFD3_PDB',
      example_id: 'denovo_design',
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

  it('sends the clicked protein-to-protein RFD3 design into MPNN', async () => {
    useAppStore.getState().setAuth('researcher-token', researcher)
    useAppStore.getState().setPdbContent('TARGET_PDB')
    useAppStore.getState().setRfd3Config({
      targetStructure: 'A/1-100',
      hotspots: 'A/42',
    })
    mockRunRFD3.mockResolvedValue({
      success: true,
      designs: [
        { index: 0, batch: 0, design_in_batch: 0, name: 'rfd3_0', pdb_path: '', pdb_content: 'RFD3_0', plddt: 80 },
        { index: 1, batch: 0, design_in_batch: 1, name: 'rfd3_1', pdb_path: '', pdb_content: 'RFD3_1', plddt: 81 },
        { index: 2, batch: 0, design_in_batch: 2, name: 'rfd3_2', pdb_path: '', pdb_content: 'RFD3_2', plddt: 82 },
        { index: 3, batch: 0, design_in_batch: 3, name: 'rfd3_3', pdb_path: '', pdb_content: 'RFD3_3', plddt: 83 },
      ],
      batches: [
        {
          batch_idx: 0,
          num_structures: 4,
          designs: [
            { index: 0, batch: 0, design_in_batch: 0, name: 'rfd3_0', pdb_path: '', pdb_content: 'RFD3_0', plddt: 80 },
            { index: 1, batch: 0, design_in_batch: 1, name: 'rfd3_1', pdb_path: '', pdb_content: 'RFD3_1', plddt: 81 },
            { index: 2, batch: 0, design_in_batch: 2, name: 'rfd3_2', pdb_path: '', pdb_content: 'RFD3_2', plddt: 82 },
            { index: 3, batch: 0, design_in_batch: 3, name: 'rfd3_3', pdb_path: '', pdb_content: 'RFD3_3', plddt: 83 },
          ],
        },
      ],
      num_batches: 1,
      num_designs: 4,
      first_backbone_pdb: 'RFD3_0',
      output_dir: '',
    })
    mockRunMPNN.mockResolvedValue({
      success: true,
      sequences: [
        { index: 0, name: 'seq_0', sequence: 'ACDEFGHIK', pdb_path: '', pdb_content: 'MPNN_FOR_RFD3_2', score: -0.3 },
      ],
      num_sequences: 1,
      first_sequence_pdb: 'MPNN_FOR_RFD3_2',
      output_dir: '',
    })

    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: '新建任务' }))
    await user.click(screen.getByRole('button', { name: '开始生成' }))
    await user.click(await screen.findByRole('button', { name: /① RFD3 骨架生成/ }))
    await screen.findByText('rfd3_2')

    await user.click(screen.getAllByRole('button', { name: /送入MPNN/ })[2])

    await screen.findByText('Sequence 1')
    expect(mockRunMPNN).toHaveBeenCalledWith({
      backbone_pdb_content: 'RFD3_2',
      batch_size: 10,
      fixed_chains: ['A'],
    })
    expect(screen.queryAllByRole('button', { name: /送入MPNN/ })).toHaveLength(0)
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
