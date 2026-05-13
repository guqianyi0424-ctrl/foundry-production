import { useMemo, useState } from 'react'
import { runDeNovoRFD3, runMPNN, runRF3 } from '@/api'
import type { MPNNResponse, MPNNSequence, RFD3Design, RFD3Response, RF3Response } from '@/api'
import { SilentStructureViewer } from '@/components/SilentStructureViewer'
import { AlertTriangle, CheckCircle2, Dna, FlaskConical, Play, RotateCcw, Shield } from 'lucide-react'

const clampNumber = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max)

const shortSequence = (sequence: string, max = 140) =>
  sequence.length > max ? `${sequence.slice(0, max)}...` : sequence

const hasRmsd = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0

const PREVIEW_DESIGN_LIMIT = 2

export function DeNovoDesignPage() {
  const [length, setLength] = useState(80)
  const [diffusionBatchSize, setDiffusionBatchSize] = useState(2)
  const [nBatches, setNBatches] = useState(1)
  const [mpnnBatchSize, setMpnnBatchSize] = useState(10)
  const [rfd3Results, setRfd3Results] = useState<RFD3Response | null>(null)
  const [mpnnResults, setMpnnResults] = useState<MPNNResponse | null>(null)
  const [rf3Results, setRf3Results] = useState<RF3Response | null>(null)
  const [selectedBackbone, setSelectedBackbone] = useState<RFD3Design | null>(null)
  const [selectedSequence, setSelectedSequence] = useState<MPNNSequence | null>(null)
  const [runningStep, setRunningStep] = useState<'rfd3' | 'mpnn' | 'rf3' | null>(null)

  const expectedBackbones = useMemo(() => diffusionBatchSize * nBatches, [diffusionBatchSize, nBatches])
  const isRunning = runningStep !== null
  const visibleBackbones = useMemo(
    () => (rfd3Results?.designs ?? []).slice(0, PREVIEW_DESIGN_LIMIT),
    [rfd3Results?.designs],
  )

  const handleRunRFD3 = async () => {
    setRunningStep('rfd3')
    setRfd3Results(null)
    setMpnnResults(null)
    setRf3Results(null)
    setSelectedBackbone(null)
    setSelectedSequence(null)
    try {
      const result = await runDeNovoRFD3({
        length: clampNumber(length, 40, 200),
        diffusion_batch_size: clampNumber(diffusionBatchSize, 1, 10),
        n_batches: clampNumber(nBatches, 1, 10),
      })
      setRfd3Results(result)
      setSelectedBackbone(result.designs?.[0] ?? null)
    } catch (err) {
      alert('De Novo RFD3 运行失败: ' + String(err))
    } finally {
      setRunningStep(null)
    }
  }

  const handleRunMPNN = async () => {
    if (!selectedBackbone?.pdb_content) return
    setRunningStep('mpnn')
    setMpnnResults(null)
    setRf3Results(null)
    setSelectedSequence(null)
    try {
      const result = await runMPNN({
        backbone_pdb_content: selectedBackbone.pdb_content,
        batch_size: clampNumber(mpnnBatchSize, 1, 50),
        ...(rfd3Results?.experiment_id ? { experiment_id: rfd3Results.experiment_id } : {}),
      })
      setMpnnResults(result)
      setSelectedSequence(result.sequences?.[0] ?? null)
    } catch (err) {
      alert('MPNN 序列设计失败: ' + String(err))
    } finally {
      setRunningStep(null)
    }
  }

  const handleRunRF3 = async () => {
    if (!selectedSequence?.pdb_content) return
    setRunningStep('rf3')
    setRf3Results(null)
    try {
      const result = await runRF3({
        mpnn_pdb_content: selectedSequence.pdb_content,
        rfd3_pdb_content: selectedBackbone?.pdb_content,
        example_id: 'denovo_design',
        ...(rfd3Results?.experiment_id ? { experiment_id: rfd3Results.experiment_id } : {}),
      })
      setRf3Results(result)
    } catch (err) {
      alert('RF3 验证失败: ' + String(err))
    } finally {
      setRunningStep(null)
    }
  }

  const handleReset = () => {
    setRfd3Results(null)
    setMpnnResults(null)
    setRf3Results(null)
    setSelectedBackbone(null)
    setSelectedSequence(null)
    setRunningStep(null)
  }

  return (
    <div className="p-6 max-w-[1400px] mx-auto space-y-5">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <Dna size={22} className="text-blue-600" />
            De Novo Protein Design
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            无靶点结构输入：RFD3 生成新骨架，MPNN 设计序列，RF3 验证设计可折叠性。
          </p>
        </div>
        <button
          onClick={handleReset}
          disabled={isRunning}
          className="inline-flex items-center gap-2 rounded-lg border border-gray-200 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <RotateCcw size={16} />
          重置
        </button>
      </div>

      <section className="bg-white rounded-xl border border-gray-200 p-5">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <NumberField label="Protein length" value={length} min={40} max={200} onChange={setLength} />
          <NumberField label="Diffusion batch size" value={diffusionBatchSize} min={1} max={10} onChange={setDiffusionBatchSize} />
          <NumberField label="N batches" value={nBatches} min={1} max={10} onChange={setNBatches} />
          <NumberField label="MPNN batch size" value={mpnnBatchSize} min={1} max={50} onChange={setMpnnBatchSize} />
        </div>
        <div className="mt-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="rounded-lg bg-blue-50 px-3 py-2 text-xs text-blue-700">
            将生成 {nBatches} 个 batch × {diffusionBatchSize} 个骨架，共 {expectedBackbones} 个 de novo 结构。
          </div>
          <button
            onClick={handleRunRFD3}
            disabled={isRunning}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-400"
          >
            <Play size={16} />
            {runningStep === 'rfd3' ? 'RFD3 生成中...' : '新建任务'}
          </button>
        </div>
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_420px] gap-5">
        <div className="space-y-5">
          <WorkflowBlock
            title="1. RFD3 Backbone Generation"
            icon={<Dna size={18} className="text-blue-600" />}
            action={
              <button
                onClick={handleRunMPNN}
                disabled={!selectedBackbone?.pdb_content || isRunning}
                className="inline-flex items-center gap-2 rounded-lg bg-green-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-green-700 disabled:cursor-not-allowed disabled:bg-gray-300"
              >
                <FlaskConical size={14} />
                {runningStep === 'mpnn' ? 'MPNN 运行中...' : '送入 MPNN'}
              </button>
            }
          >
            {runningStep === 'rfd3' && <LoadingRow text="RFD3 正在生成 de novo 骨架..." />}
            {!rfd3Results && runningStep !== 'rfd3' && <EmptyRow text="设置参数后运行 RFD3，无需上传靶点结构。" />}
            {rfd3Results?.mock && <MockNotice label="RFD3" />}
            {rfd3Results?.designs && rfd3Results.designs.length > 0 && (
              <>
                <div className="rounded-lg bg-blue-50 px-3 py-2 text-xs text-blue-700">
                  生成 {rfd3Results.num_designs} 个骨架结构，当前仅展示 {visibleBackbones.length} 个预览 design，不影响后台完整生成。
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {visibleBackbones.map((design) => (
                    <button
                      key={`${design.batch}-${design.index}`}
                      onClick={() => setSelectedBackbone(design)}
                      className={`text-left rounded-xl border p-3 transition-all ${
                        selectedBackbone?.index === design.index
                          ? 'border-blue-400 bg-blue-50 ring-1 ring-blue-200'
                          : 'border-gray-100 hover:border-gray-200 hover:bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-semibold text-gray-900 truncate">{design.name}</span>
                        <span className="shrink-0 rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-700">#{design.index + 1}</span>
                      </div>
                      <div className="mt-2 text-xs text-gray-500">Batch {design.batch} · pLDDT {(design.plddt ?? 0).toFixed(1)}</div>
                    </button>
                  ))}
                </div>
              </>
            )}
          </WorkflowBlock>

          <WorkflowBlock
            title="2. MPNN Sequence Design"
            icon={<FlaskConical size={18} className="text-green-600" />}
            action={
              <button
                onClick={handleRunRF3}
                disabled={!selectedSequence?.pdb_content || isRunning}
                className="inline-flex items-center gap-2 rounded-lg bg-orange-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-orange-700 disabled:cursor-not-allowed disabled:bg-gray-300"
              >
                <Shield size={14} />
                {runningStep === 'rf3' ? 'RF3 验证中...' : '送入 RF3'}
              </button>
            }
          >
            <div className="rounded-lg border border-blue-100 bg-blue-50 px-3 py-2 text-xs text-blue-700">
              ProteinMPNN 生成的是氨基酸序列；只有当后端返回 PDB 时，这里才显示“设计序列对应骨架”，最终折叠结构由 RF3 验证。
            </div>
            {runningStep === 'mpnn' && <LoadingRow text="MPNN 正在为选中骨架设计序列..." />}
            {!mpnnResults && runningStep !== 'mpnn' && <EmptyRow text="先在 RFD3 结果中选择一个骨架，再运行 MPNN。" />}
            {mpnnResults?.mock && <MockNotice label="MPNN" />}
            {mpnnResults?.sequences && mpnnResults.sequences.length > 0 && (
              <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1">
                {mpnnResults.sequences.map((sequence) => (
                  <button
                    key={sequence.index}
                    onClick={() => setSelectedSequence(sequence)}
                    className={`w-full text-left rounded-xl border p-3 transition-all ${
                      selectedSequence?.index === sequence.index
                        ? 'border-green-400 bg-green-50 ring-1 ring-green-200'
                        : 'border-gray-100 hover:border-gray-200 hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm font-semibold text-gray-900">Sequence {sequence.index + 1}</span>
                      <span className="shrink-0 text-xs text-gray-500">{sequence.sequence.length} aa</span>
                    </div>
                    <div className="mt-2 rounded-lg bg-white/70 p-2 font-mono text-xs leading-relaxed text-gray-700 break-all">
                      {shortSequence(sequence.sequence)}
                    </div>
                    <div className="mt-2 text-xs text-gray-500">Score {(sequence.score ?? 0).toFixed(2)}</div>
                  </button>
                ))}
              </div>
            )}
          </WorkflowBlock>

          <WorkflowBlock title="3. RF3 Structure Validation" icon={<Shield size={18} className="text-orange-600" />}>
            {runningStep === 'rf3' && <LoadingRow text="RF3 正在预测结构并计算 RMSD..." />}
            {!rf3Results && runningStep !== 'rf3' && <EmptyRow text="选择一条带 PDB 的 MPNN 序列后运行 RF3 验证。" />}
            {rf3Results?.mock && <MockNotice label="RF3" />}
            {rf3Results && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <MetricCard label="RMSD" value={hasRmsd(rf3Results.rmsd) ? `${rf3Results.rmsd.toFixed(2)} Å` : '未计算'} tone={!hasRmsd(rf3Results.rmsd) ? 'gray' : rf3Results.rmsd < 2 ? 'green' : rf3Results.rmsd < 5 ? 'amber' : 'red'} />
                  <MetricCard label="pLDDT" value={rf3Results.avg_plddt.toFixed(1)} tone={rf3Results.avg_plddt > 80 ? 'green' : rf3Results.avg_plddt > 60 ? 'amber' : 'red'} />
                  <MetricCard label="pTM" value={(rf3Results.summary?.ptm ?? 0).toFixed(3)} tone="blue" />
                  <MetricCard label="Ranking" value={(rf3Results.summary?.ranking_score ?? 0).toFixed(3)} tone="gray" />
                </div>
                <div className={`rounded-xl border px-4 py-3 text-sm ${rf3Results.passed ? 'border-green-200 bg-green-50 text-green-800' : 'border-amber-200 bg-amber-50 text-amber-800'}`}>
                  <div className="flex items-center gap-2 font-semibold">
                    {rf3Results.passed ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
                    {rf3Results.passed ? '验证通过' : '需要优化'}
                  </div>
                  <div className="mt-1 text-xs text-gray-600">
                    {hasRmsd(rf3Results.rmsd)
                      ? `RMSD ${rf3Results.rmsd.toFixed(2)} Å，${rf3Results.rmsd_interpretation}`
                      : '缺少参考结构或 CA 原子匹配失败，RMSD 未计算；这不等于设计失败'}
                  </div>
                </div>
              </div>
            )}
          </WorkflowBlock>
        </div>

        <aside className="space-y-5">
          <PreviewCard title="RFD3 生成骨架">
            <SilentStructureViewer pdbContent={selectedBackbone?.pdb_content} title="Selected RFD3 backbone" heightClassName="h-72" emptyText="选择 RFD3 骨架后显示" />
          </PreviewCard>
          <PreviewCard title="MPNN 序列映射骨架">
            <SilentStructureViewer pdbContent={selectedSequence?.pdb_content} title="Selected MPNN sequence structure" heightClassName="h-72" emptyText="当前 MPNN 结果没有 PDB 结构" />
          </PreviewCard>
          <PreviewCard title="RFD3 / RF3 结构对比">
            <SilentStructureViewer
              pdbContent={selectedBackbone?.pdb_content}
              secondPdbContent={rf3Results?.predicted_pdb}
              title="RFD3 and RF3 comparison"
              heightClassName="h-80"
              emptyText="运行 RF3 后显示叠合对比"
            />
          </PreviewCard>
        </aside>
      </section>
    </div>
  )
}

function NumberField({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  onChange: (value: number) => void
}) {
  return (
    <label className="block">
      <span className="block text-xs font-medium text-gray-600 mb-1.5">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(event) => onChange(clampNumber(Number(event.target.value), min, max))}
        className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none"
      />
      <span className="mt-1 block text-[11px] text-gray-400">{min}-{max}</span>
    </label>
  )
}

function WorkflowBlock({
  title,
  icon,
  action,
  children,
}: {
  title: string
  icon: React.ReactNode
  action?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <section className="bg-white rounded-xl border border-gray-200 p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 text-base font-bold text-gray-900">
          {icon}
          {title}
        </h2>
        {action}
      </div>
      <div className="space-y-3">{children}</div>
    </section>
  )
}

function PreviewCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="overflow-hidden rounded-xl border border-gray-200 bg-white">
      <div className="border-b border-gray-100 px-4 py-2.5 text-sm font-semibold text-gray-700">{title}</div>
      {children}
    </section>
  )
}

function LoadingRow({ text }: { text: string }) {
  return (
    <div className="flex items-center justify-center rounded-xl bg-gray-50 py-8 text-sm text-gray-600">
      <div className="mr-3 h-6 w-6 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
      {text}
    </div>
  )
}

function EmptyRow({ text }: { text: string }) {
  return <div className="rounded-xl border border-dashed border-gray-200 bg-gray-50 py-8 text-center text-sm text-gray-400">{text}</div>
}

function MockNotice({ label }: { label: string }) {
  return <div className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700">{label} 当前使用 mock/fallback 结果。</div>
}

function MetricCard({ label, value, tone }: { label: string; value: string; tone: 'green' | 'amber' | 'red' | 'blue' | 'gray' }) {
  const colorMap = {
    green: 'text-green-600',
    amber: 'text-amber-600',
    red: 'text-red-600',
    blue: 'text-blue-600',
    gray: 'text-gray-900',
  }
  return (
    <div className="rounded-xl border border-gray-100 bg-white p-4">
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`mt-1 text-xl font-bold ${colorMap[tone]}`}>{value}</div>
    </div>
  )
}
