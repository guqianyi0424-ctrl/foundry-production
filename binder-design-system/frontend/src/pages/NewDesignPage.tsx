import { useCallback, useState } from 'react'
import { useAppStore } from '@/store/useAppStore'
import { SequenceViewer } from '@/components/SequenceViewer'
import { MolstarViewer } from '@/components/MolstarViewer'
import { DesignPanel } from '@/components/DesignPanel'
import { uploadPdb, predictHotspot, runRFD3, runMPNN, runRF3 } from '@/api'
import type { PredictHotspotResponse, RFD3Design, MPNNSequence, RF3Response } from '@/api'
import type { HotspotResidue } from '@/types'
import { parseStructureFile } from '@/utils/pdbParser'
import { Upload, Play, RotateCcw, Sparkles, Scissors, Target, RefreshCw, X, CheckCircle2, Dna, FlaskConical, ChevronRight, Shield, TrendingUp, AlertTriangle } from 'lucide-react'

const splitHotspotConfig = (input: string): string[] =>
  input
    .split(',')
    .map(item => item.trim())
    .filter(Boolean)

const toRfd3HotspotToken = (input: string): string =>
  input.replace('/', '').replace(/\s+/g, '')

const hotspotChainFromConfig = (input: string): string | null => {
  const trimmed = input.trim()
  if (!trimmed) return null
  return trimmed.includes('/') ? trimmed.split('/')[0] : trimmed.match(/^[A-Za-z0-9]+/)?.[0] ?? null
}

const mergeHotspotSelections = (...groups: HotspotResidue[][]): HotspotResidue[] => {
  const seen = new Set<string>()
  const merged: HotspotResidue[] = []

  for (const group of groups) {
    for (const hotspot of group) {
      const key = `${hotspot.chain}/${hotspot.residue}`
      if (seen.has(key)) continue
      seen.add(key)
      merged.push(hotspot)
    }
  }

  return merged
}

interface HotspotPredictionModalProps {
  open: boolean
  onClose: () => void
  prediction: PredictHotspotResponse | null
}

function HotspotPredictionModal({ open, onClose, prediction }: HotspotPredictionModalProps) {
  if (!open || !prediction) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-2xl shadow-2xl w-[520px] max-h-[80vh] overflow-hidden">
        <div className="p-6 border-b border-gray-100">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center">
                <CheckCircle2 size={22} className="text-green-600" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-gray-900">热点预测完成</h3>
                <p className="text-sm text-gray-500 mt-0.5">
                  预测了 <span className="text-green-600 font-semibold">{prediction.num_hotspots}</span> 个热点残基
                  {prediction.model_loaded
                    ? <span className="ml-1 text-blue-500">(DL模型)</span>
                    : <span className="ml-1 text-amber-500">(规则预测)</span>
                  }
                </p>
              </div>
            </div>
            <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors">
              <X size={18} className="text-gray-400" />
            </button>
          </div>
        </div>
        <div className="p-6 max-h-[50vh] overflow-y-auto">
          <div className="space-y-2">
            {prediction.hotspots.map((h, i) => (
              <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-gray-50 hover:bg-blue-50 transition-colors">
                <div className="flex items-center gap-3">
                  <span className="w-7 h-7 rounded-full bg-blue-100 text-blue-700 text-xs font-bold flex items-center justify-center">{i + 1}</span>
                  <div>
                    <span className="text-sm font-medium text-gray-900">{h.residue_name} {h.residue}</span>
                    <span className="ml-2 text-xs text-gray-400">链 {h.chain}</span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-gradient-to-r from-blue-400 to-green-400" style={{ width: `${Math.min(h.score * 100, 100)}%` }} />
                  </div>
                  <span className="text-xs font-mono text-gray-600 w-12 text-right">{h.score.toFixed(3)}</span>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 p-3 bg-amber-50 rounded-xl text-xs text-amber-700">
            预测热点已在序列下方以红点标记；双击可移除，点击「指定热点」才会写入热点参数框。
          </div>
        </div>
        <div className="p-4 border-t border-gray-100 flex justify-end gap-3">
          <button onClick={onClose} className="px-5 py-2 rounded-lg bg-green-600 text-white text-sm font-medium hover:bg-green-700 transition-colors">知道了</button>
        </div>
      </div>
    </div>
  )
}

interface RFD3ModalProps {
  open: boolean
  onClose: () => void
  onRun: (params: { binder_length: number; diffusion_batch_size: number; n_batches: number }) => void
  target: string
  hotspots: string[]
  isRunning: boolean
}

function RFD3Modal({ open, onClose, onRun, target, hotspots, isRunning }: RFD3ModalProps) {
  const [binderLength, setBinderLength] = useState(80)
  const [batchSize, setBatchSize] = useState(2)
  const [nBatches, setNBatches] = useState(2)

  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-2xl shadow-2xl w-[560px] overflow-hidden">
        <div className="p-6 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
              <Dna size={22} className="text-blue-600" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-gray-900">RFD3 骨架生成</h3>
              <p className="text-sm text-gray-500">配置参数并生成蛋白质骨架结构</p>
            </div>
          </div>
        </div>
        <div className="p-6 space-y-4">
          <div className="p-3 bg-gray-50 rounded-xl space-y-1.5">
            <div className="flex justify-between text-sm"><span className="text-gray-500">靶点结构</span><span className="font-medium text-gray-900">{target || '未设置'}</span></div>
            <div className="flex justify-between text-sm"><span className="text-gray-500">热点残基</span><span className="font-medium text-gray-900">{hotspots.length > 0 ? hotspots.join(', ') : '未设置'}</span></div>
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Binder 长度</label>
              <input type="number" value={binderLength} onChange={e => setBinderLength(Number(e.target.value))} min={40} max={200} className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-400" />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Batch 大小</label>
              <input type="number" value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} min={1} max={10} className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-400" />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Batch 数量</label>
              <input type="number" value={nBatches} onChange={e => setNBatches(Number(e.target.value))} min={1} max={10} className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-400" />
            </div>
          </div>
          <div className="p-3 bg-blue-50 rounded-xl text-xs text-blue-700">
            🧬 将生成 {nBatches} 个 batch × {batchSize} 个骨架 = 共 {nBatches * batchSize} 个结构
          </div>
        </div>
        <div className="p-4 border-t border-gray-100 flex justify-end gap-3">
          <button onClick={onClose} className="px-5 py-2 rounded-lg border border-gray-200 text-gray-600 text-sm hover:bg-gray-50 transition-colors">取消</button>
          <button onClick={() => onRun({ binder_length: binderLength, diffusion_batch_size: batchSize, n_batches: nBatches })} disabled={isRunning} className={`px-5 py-2 rounded-lg text-white text-sm font-medium transition-colors ${isRunning ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}>
            {isRunning ? '生成中...' : '开始生成'}
          </button>
        </div>
      </div>
    </div>
  )
}

export function NewDesignPage() {
  const targetFile = useAppStore((s) => s.targetFile)
  const setTargetFile = useAppStore((s) => s.setTargetFile)
  const pdbContent = useAppStore((s) => s.pdbContent)
  const setPdbContent = useAppStore((s) => s.setPdbContent)
  const chains = useAppStore((s) => s.chains)
  const setChains = useAppStore((s) => s.setChains)
  const selectedRange = useAppStore((s) => s.selectedRange)
  const setSelectedRange = useAppStore((s) => s.setSelectedRange)
  const selectedHotspots = useAppStore((s) => s.selectedHotspots)
  const setSelectedHotspots = useAppStore((s) => s.setSelectedHotspots)
  const predictedHotspots = useAppStore((s) => s.predictedHotspots)
  const setPredictedHotspots = useAppStore((s) => s.setPredictedHotspots)
  const setRfd3Config = useAppStore((s) => s.setRfd3Config)
  const binderLength = useAppStore((s) => s.binderLength)
  const rfd3Config = useAppStore((s) => s.rfd3Config)
  const rfd3Results = useAppStore((s) => s.rfd3Results)
  const setRfd3Results = useAppStore((s) => s.setRfd3Results)
  const mpnnResults = useAppStore((s) => s.mpnnResults)
  const setMpnnResults = useAppStore((s) => s.setMpnnResults)
  const rf3Results = useAppStore((s) => s.rf3Results)
  const setRf3Results = useAppStore((s) => s.setRf3Results)
  const isRunning = useAppStore((s) => s.isRunning)
  const setIsRunning = useAppStore((s) => s.setIsRunning)
  const addJob = useAppStore((s) => s.addJob)
  const resetAll = useAppStore((s) => s.resetAll)

  const [isPredicting, setIsPredicting] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictHotspotResponse | null>(null)
  const [showPredictionModal, setShowPredictionModal] = useState(false)
  const [showRFD3Modal, setShowRFD3Modal] = useState(false)
  const [isRFD3Running, setIsRFD3Running] = useState(false)
  const [isMPNNRunning, setIsMPNNRunning] = useState(false)
  const [runningMPNNDesignIdx, setRunningMPNNDesignIdx] = useState<number | null>(null)
  const [isRF3Running, setIsRF3Running] = useState(false)
  const [selectedRFD3Design, setSelectedRFD3Design] = useState<RFD3Design | null>(null)
  const [selectedMPNNSeq, setSelectedMPNNSeq] = useState<MPNNSequence | null>(null)
  const [activeStep, setActiveStep] = useState(0)
  const pendingHotspots = mergeHotspotSelections(predictedHotspots, selectedHotspots)
  const committedHotspotItems = splitHotspotConfig(rfd3Config.hotspots)
  const committedHotspotTokens = committedHotspotItems.map(toRfd3HotspotToken)

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setTargetFile(file)
    const content = await file.text()
    try {
      const chains = parseStructureFile(content, file.name)
      if (chains.length === 0) { alert('文件解析失败'); return }
      setChains(chains)
      setPdbContent(content)
    } catch {
      try {
        const res = await uploadPdb(file)
        setChains(res.chains.map(c => ({ ...c, resSeqs: c.resSeqs ?? Array.from({ length: c.length }, (_, i) => i + 1) })))
        setPdbContent(res.pdb_content)
      } catch { alert('文件解析失败') }
    }
  }, [setTargetFile, setChains, setPdbContent])

  const handlePredictHotspot = useCallback(async () => {
    if (!pdbContent) return
    setIsPredicting(true)
    try {
      const res = await predictHotspot(pdbContent)
      const hotspots = res.hotspots.map(h => ({ chain: h.chain, residue: h.residue, score: h.score }))
      setPredictedHotspots(hotspots)
      setPredictionResult(res)
      setShowPredictionModal(true)
    } catch { alert('热点预测失败') }
    finally { setIsPredicting(false) }
  }, [pdbContent, setPredictedHotspots])

  const handleTrimTarget = () => {
    if (!selectedRange) { alert('请先选择残基范围'); return }
    const targetStr = `${selectedRange.chain}/${selectedRange.startResSeq}-${selectedRange.endResSeq}`
    if (confirm(`裁剪结构为 ${targetStr}，是否确认？`)) setRfd3Config({ targetStructure: targetStr })
  }

  const handleSpecifyHotspot = () => {
    if (pendingHotspots.length === 0) { alert('请先选择热点残基'); return }
    const hotspotStr = pendingHotspots.map(h => `${h.chain}/${h.residue}`).join(', ')
    if (confirm(`指定热点: ${hotspotStr}，是否确认？`)) setRfd3Config({ hotspots: hotspotStr })
  }

  const handleResetView = useCallback(() => {
    resetAll()
    setRfd3Results(null)
    setMpnnResults(null)
  }, [resetAll, setRfd3Results, setMpnnResults])

  const handleRunRFD3 = useCallback(async (params: { binder_length: number; diffusion_batch_size: number; n_batches: number }) => {
    if (!pdbContent) return
    setIsRFD3Running(true)
    setShowRFD3Modal(false)
    try {
      const res = await runRFD3({
        pdb_content: pdbContent,
        target: rfd3Config.targetStructure || undefined,
        hotspots: committedHotspotTokens.length > 0 ? committedHotspotTokens : undefined,
        binder_length: params.binder_length,
        diffusion_batch_size: params.diffusion_batch_size,
        n_batches: params.n_batches,
      })
      setRfd3Results(res)
      setActiveStep(1)
    } catch (err) { alert('RFD3运行失败: ' + String(err)) }
    finally { setIsRFD3Running(false) }
  }, [pdbContent, rfd3Config, committedHotspotTokens, setRfd3Results])

  const handleRunMPNN = useCallback(async (backbonePdb: string, designIdx: number) => {
    setIsMPNNRunning(true)
    setRunningMPNNDesignIdx(designIdx)
    try {
      const targetChains = [...new Set(committedHotspotItems.map(hotspotChainFromConfig).filter((chain): chain is string => Boolean(chain)))]
      const res = await runMPNN({
        backbone_pdb_content: backbonePdb,
        batch_size: 10,
        fixed_chains: targetChains.length > 0 ? targetChains : undefined,
      })
      setMpnnResults(res)
      setActiveStep(2)
    } catch (err) { alert('MPNN运行失败: ' + String(err)) }
    finally { setIsMPNNRunning(false); setRunningMPNNDesignIdx(null) }
  }, [committedHotspotItems, setMpnnResults])

  const handleRunRF3 = useCallback(async (mpnnPdb: string) => {
    setIsRF3Running(true)
    try {
      const rfd3Pdb = selectedRFD3Design?.pdb_content || undefined
      const res = await runRF3({
        mpnn_pdb_content: mpnnPdb,
        rfd3_pdb_content: rfd3Pdb,
        example_id: 'binder_design',
      })
      setRf3Results(res)
      setActiveStep(2)
    } catch (err) { alert('RF3运行失败: ' + String(err)) }
    finally { setIsRF3Running(false) }
  }, [selectedRFD3Design, setRf3Results])

  return (
    <div className="p-6 max-w-[1400px] mx-auto space-y-5">
      <HotspotPredictionModal open={showPredictionModal} onClose={() => setShowPredictionModal(false)} prediction={predictionResult} />
      <RFD3Modal open={showRFD3Modal} onClose={() => setShowRFD3Modal(false)} onRun={handleRunRFD3} target={rfd3Config.targetStructure} hotspots={committedHotspotItems} isRunning={isRFD3Running} />

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex bg-gray-100 rounded-lg p-0.5">
            {['表单模式', 'JSON模式'].map(mode => (
              <button key={mode} className="px-4 py-1.5 text-xs font-medium rounded-md transition-all text-gray-600 hover:text-gray-900">{mode}</button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleResetView} className="flex items-center gap-2 px-4 py-2 rounded-lg border border-gray-200 text-gray-700 text-sm hover:bg-gray-50 transition-all">
            <RotateCcw size={16} />重置
          </button>
        </div>
      </div>

      {/* Target Structure */}
      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-red-500">*</span><span className="text-sm font-semibold">目标结构</span>
        </div>
        <div className="flex items-center gap-3 mb-2">
          <label className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${!targetFile ? 'border-primary-400 bg-primary-50 text-primary-700' : 'border-gray-200 text-gray-600'} cursor-pointer text-sm transition-all`}>
            <Upload size={16} /><span>上传文件</span>
            <input type="file" accept=".pdb,.cif,.ent" onChange={handleUpload} className="hidden" />
          </label>
          <span className="text-xs text-gray-400">支持pdb/cif格式，最大200MB</span>
        </div>
        {targetFile && (
          <div className="flex items-center gap-2 mt-3 p-2.5 bg-gray-50 rounded-lg border border-gray-100 w-fit">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14,2 14,8 20,8"/></svg>
            <span className="text-sm text-gray-700 font-medium">{targetFile.name}</span>
            <button onClick={() => { setTargetFile(null); setPdbContent(''); setChains([]) }} className="ml-2 text-red-500 hover:text-red-700"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>
          </div>
        )}
      </div>

      {/* Structure & Sequence */}
      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-bold text-gray-900 flex items-center gap-2"><span>🧬</span> 结构与序列</h3>
          <div className="flex gap-2">
            <button onClick={handlePredictHotspot} disabled={!pdbContent || isPredicting} title="AI预测热点 (Top-5)" className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition-all ${!pdbContent || isPredicting ? 'border-gray-100 text-gray-300 cursor-not-allowed' : 'border-purple-200 text-purple-600 hover:bg-purple-50 bg-purple-50/50'}`}>
              <Sparkles size={14} />{isPredicting ? '预测中...' : '预测热点'}
            </button>
            <button onClick={handleTrimTarget} disabled={!selectedRange} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition-all ${!selectedRange ? 'border-gray-100 text-gray-300 cursor-not-allowed' : 'border-gray-200 text-gray-600 hover:bg-orange-50 hover:text-orange-600'}`}>
              <Scissors size={14} />裁剪靶点
            </button>
            <button onClick={handleSpecifyHotspot} disabled={pendingHotspots.length === 0} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition-all ${pendingHotspots.length === 0 ? 'border-gray-100 text-gray-300 cursor-not-allowed' : 'border-gray-200 text-gray-600 hover:bg-green-50 hover:text-green-600'}`}>
              <Target size={14} />指定热点
            </button>
            <button onClick={() => { setSelectedRange(null); setSelectedHotspots([]); setPredictedHotspots([]) }} disabled={selectedHotspots.length === 0 && predictedHotspots.length === 0 && !selectedRange} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition-all ${selectedHotspots.length === 0 && predictedHotspots.length === 0 && !selectedRange ? 'border-gray-100 text-gray-300 cursor-not-allowed' : 'border-gray-200 text-gray-600 hover:bg-gray-50'}`}>
              <RefreshCw size={14} />重置
            </button>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-5">
          <SequenceViewer />
          <MolstarViewer />
        </div>
      </div>

      <DesignPanel onOpenRFD3={() => setShowRFD3Modal(true)} isRFD3Running={isRFD3Running} />

      {/* Step Pipeline: RFD3 → MPNN → RF3 */}
      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <div className="flex items-center gap-3 mb-4">
          {['① RFD3 骨架生成', '② MPNN 序列设计', '③ RF3 结构验证'].map((step, i) => (
            <button key={step} onClick={() => setActiveStep(i)} className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeStep === i ? 'bg-blue-50 text-blue-700 border border-blue-200' : 'text-gray-500 hover:bg-gray-50'}`}>
              {step}
              {i < 2 && <ChevronRight size={14} className="ml-1 text-gray-300" />}
            </button>
          ))}
        </div>

        {/* RFD3 Results */}
        {activeStep === 0 && (
          <div>
            {isRFD3Running && (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin w-8 h-8 border-3 border-blue-500 border-t-transparent rounded-full mr-3" />
                <span className="text-gray-600">RFD3 正在生成骨架结构...</span>
              </div>
            )}
            {rfd3Results && !isRFD3Running && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-600">
                    生成 <span className="font-semibold text-gray-900">{rfd3Results.num_designs}</span> 个骨架结构
                    ({rfd3Results.num_batches} batch × {rfd3Results.batches?.[0]?.num_structures || '?'} 结构)
                    {rfd3Results.mock && <span className="ml-2 text-amber-500">(mock模式)</span>}
                  </div>
                  <button onClick={() => setShowRFD3Modal(true)} className="px-3 py-1.5 rounded-lg border border-blue-200 text-blue-600 text-xs hover:bg-blue-50 transition-all">
                    重新生成
                  </button>
                </div>
                {rfd3Results.batches?.map((batch) => (
                  <div key={batch.batch_idx} className="border border-gray-100 rounded-xl p-4">
                    <h4 className="text-sm font-semibold text-gray-700 mb-3">Batch {batch.batch_idx}</h4>
                    <div className="grid grid-cols-2 gap-3">
                      {batch.designs.map((design) => (
                        <div key={design.index} onClick={() => setSelectedRFD3Design(design)} className={`p-3 rounded-xl border cursor-pointer transition-all ${selectedRFD3Design?.index === design.index ? 'border-blue-400 bg-blue-50 ring-1 ring-blue-200' : 'border-gray-100 hover:border-gray-200 hover:bg-gray-50'}`}>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-gray-900">{design.name}</span>
                            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700">Design #{design.index + 1}</span>
                          </div>
                          {design.pdb_content && (
                            <div className="h-32 rounded-lg overflow-hidden border border-gray-100 bg-gray-50">
                              <iframe srcDoc={`<!DOCTYPE html><html><head><script src="https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/viewer/molstar.js"></script><style>body{margin:0;padding:0;overflow:hidden}#app{width:100%;height:100%}</style></head><body><div id="app"></div><script>var noop=function(){};console.log=noop;console.warn=noop;console.info=noop;console.debug=noop;molstar.Viewer.create('app',{layoutIsExpanded:false}).then(v=>{const pdbData=\`${design.pdb_content.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`;v.loadStructureFromData(pdbData,'pdb')})</script></body></html>`} className="w-full h-full border-0" title={design.name} />
                            </div>
                          )}
                          <div className="mt-2 flex justify-end">
                            <button onClick={(e) => { e.stopPropagation(); handleRunMPNN(design.pdb_content, design.index) }} disabled={isMPNNRunning} className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs bg-green-50 text-green-700 hover:bg-green-100 transition-all disabled:opacity-50">
                              <FlaskConical size={12} />{isMPNNRunning && runningMPNNDesignIdx === design.index ? 'MPNN运行中...' : '送入MPNN'}
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {!rfd3Results && !isRFD3Running && (
              <div className="text-center py-12 text-gray-400">
                <Dna size={48} className="mx-auto mb-3 opacity-30" />
                <p>点击上方「RFD3 骨架生成」按钮开始</p>
              </div>
            )}
          </div>
        )}

        {/* MPNN Results */}
        {activeStep === 1 && (
          <div>
            {isMPNNRunning && (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin w-8 h-8 border-3 border-green-500 border-t-transparent rounded-full mr-3" />
                <span className="text-gray-600">MPNN 正在设计序列...</span>
              </div>
            )}
            {mpnnResults && !isMPNNRunning && (
              <div className="space-y-3">
                <div className="text-sm text-gray-600 mb-2">
                  生成 <span className="font-semibold text-gray-900">{mpnnResults.num_sequences}</span> 条设计序列
                  {mpnnResults.mock && <span className="ml-2 text-amber-500">(mock模式)</span>}
                </div>
                <div className="max-h-[500px] overflow-y-auto space-y-2">
                  {mpnnResults.sequences.map((seq) => (
                    <div key={seq.index} onClick={() => setSelectedMPNNSeq(seq)} className={`p-3 rounded-xl border cursor-pointer transition-all ${selectedMPNNSeq?.index === seq.index ? 'border-green-400 bg-green-50 ring-1 ring-green-200' : 'border-gray-100 hover:border-gray-200'}`}>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-sm font-medium text-gray-900">Sequence {seq.index + 1}</span>
                        <span className="text-xs text-gray-400">{seq.sequence.length} aa</span>
                      </div>
                      <div className="text-xs font-mono text-gray-600 break-all leading-relaxed bg-gray-50 p-2 rounded-lg">
                        {seq.sequence.length > 120 ? seq.sequence.slice(0, 120) + '...' : seq.sequence}
                      </div>
                      {seq.pdb_content && selectedMPNNSeq?.index === seq.index && (
                        <div className="mt-3 h-48 rounded-lg overflow-hidden border border-gray-100">
                          <iframe srcDoc={`<!DOCTYPE html><html><head><script src="https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/viewer/molstar.js"></script><style>body{margin:0;padding:0;overflow:hidden}#app{width:100%;height:100%}</style></head><body><div id="app"></div><script>var noop=function(){};console.log=noop;console.warn=noop;console.info=noop;console.debug=noop;molstar.Viewer.create('app',{layoutIsExpanded:false}).then(v=>{const pdbData=\`${seq.pdb_content.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`;v.loadStructureFromData(pdbData,'pdb')})</script></body></html>`} className="w-full h-full border-0" title={`Sequence ${seq.index + 1}`} />
                        </div>
                      )}
                      {seq.pdb_content && selectedMPNNSeq?.index === seq.index && (
                        <div className="mt-2 flex justify-end">
                          <button onClick={() => handleRunRF3(seq.pdb_content)} disabled={isRF3Running} className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs bg-orange-50 text-orange-700 hover:bg-orange-100 transition-all disabled:opacity-50">
                            <Shield size={12} />{isRF3Running ? 'RF3验证中...' : '送入RF3验证'}
                          </button>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {!mpnnResults && !isMPNNRunning && (
              <div className="text-center py-12 text-gray-400">
                <FlaskConical size={48} className="mx-auto mb-3 opacity-30" />
                <p>请先在 RFD3 步骤中选择一个骨架，点击「送入MPNN」</p>
              </div>
            )}
          </div>
        )}

        {/* RF3 Results */}
        {activeStep === 2 && (
          <div>
            {isRF3Running && (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin w-8 h-8 border-3 border-orange-500 border-t-transparent rounded-full mr-3" />
                <span className="text-gray-600">RF3 正在验证结构...</span>
              </div>
            )}
            {rf3Results && !isRF3Running && (
              <div className="space-y-5">
                {/* Summary Cards */}
                <div className="grid grid-cols-4 gap-4">
                  <div className="p-4 rounded-xl border border-gray-100 bg-white">
                    <div className="text-xs text-gray-500 mb-1">Backbone RMSD</div>
                    <div className="flex items-baseline gap-1">
                      <span className={`text-2xl font-bold ${rf3Results.rmsd < 2 ? 'text-green-600' : rf3Results.rmsd < 5 ? 'text-amber-600' : 'text-red-600'}`}>{rf3Results.rmsd.toFixed(2)}</span>
                      <span className="text-xs text-gray-400">Å</span>
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${rf3Results.rmsd_interpretation === 'Excellent' ? 'bg-green-100 text-green-700' : rf3Results.rmsd_interpretation === 'Good' ? 'bg-blue-100 text-blue-700' : 'bg-amber-100 text-amber-700'}`}>
                      {rf3Results.rmsd_interpretation}
                    </span>
                  </div>
                  <div className="p-4 rounded-xl border border-gray-100 bg-white">
                    <div className="text-xs text-gray-500 mb-1">pLDDT</div>
                    <div className="flex items-baseline gap-1">
                      <span className={`text-2xl font-bold ${rf3Results.avg_plddt > 80 ? 'text-green-600' : rf3Results.avg_plddt > 60 ? 'text-amber-600' : 'text-red-600'}`}>{rf3Results.avg_plddt.toFixed(1)}</span>
                    </div>
                    <span className="text-xs text-gray-400">平均置信度</span>
                  </div>
                  <div className="p-4 rounded-xl border border-gray-100 bg-white">
                    <div className="text-xs text-gray-500 mb-1">pTM / ipTM</div>
                    <div className="flex items-baseline gap-2">
                      <span className="text-lg font-bold text-blue-600">{(rf3Results.summary?.ptm ?? 0).toFixed(3)}</span>
                      <span className="text-xs text-gray-300">/</span>
                      <span className="text-lg font-bold text-purple-600">{(rf3Results.summary?.iptm ?? 0).toFixed(3)}</span>
                    </div>
                  </div>
                  <div className="p-4 rounded-xl border border-gray-100 bg-white">
                    <div className="text-xs text-gray-500 mb-1">Ranking Score</div>
                    <div className="text-2xl font-bold text-gray-900">{(rf3Results.summary?.ranking_score ?? 0).toFixed(3)}</div>
                    <div className="flex items-center gap-1 mt-1">
                      {rf3Results.summary?.has_clash ? (
                        <span className="text-xs text-red-600 flex items-center gap-0.5"><AlertTriangle size={10} />Clash</span>
                      ) : (
                        <span className="text-xs text-green-600">No Clash</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Pass/Fail Banner */}
                <div className={`p-4 rounded-xl flex items-center gap-3 ${rf3Results.passed ? 'bg-green-50 border border-green-200' : 'bg-amber-50 border border-amber-200'}`}>
                  {rf3Results.passed ? (
                    <CheckCircle2 size={24} className="text-green-600 shrink-0" />
                  ) : (
                    <AlertTriangle size={24} className="text-amber-600 shrink-0" />
                  )}
                  <div>
                    <span className={`font-semibold ${rf3Results.passed ? 'text-green-800' : 'text-amber-800'}`}>
                      {rf3Results.passed ? '验证通过 ✅' : '需要优化 ⚠️'}
                    </span>
                    <p className="text-sm text-gray-600 mt-0.5">
                      {rf3Results.passed
                        ? `RMSD ${rf3Results.rmsd.toFixed(2)} Å < 2.0 Å，设计序列很可能折叠成预期结构`
                        : `RMSD ${rf3Results.rmsd.toFixed(2)} Å ≥ 2.0 Å，建议调整参数重新设计`}
                    </p>
                  </div>
                </div>

                {/* 3D Structure Comparison */}
                <div className="border border-gray-100 rounded-xl overflow-hidden">
                  <div className="px-4 py-2 border-b border-gray-100 bg-gray-50 flex items-center justify-between">
                    <span className="text-sm font-semibold text-gray-700">RFD3 vs RF3 结构叠合对比</span>
                    <div className="flex items-center gap-3 text-xs">
                      <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm inline-block" style={{background:'#f59e0b'}} />RFD3 骨架</span>
                      <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm inline-block" style={{background:'#3b82f6'}} />RF3 预测</span>
                    </div>
                  </div>
                  {selectedRFD3Design?.pdb_content && rf3Results.predicted_pdb ? (
                    <div className="h-96">
                      <iframe srcDoc={`<!DOCTYPE html><html><head><script src="https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/viewer/molstar.js"></script><style>body{margin:0;padding:0;overflow:hidden}#app{width:100%;height:100%}.legend{position:absolute;bottom:12px;left:12px;z-index:100;background:rgba(0,0,0,0.75);color:#fff;padding:8px 12px;border-radius:6px;font-family:sans-serif;font-size:12px;display:flex;gap:12px}</style></head><body><div id="app"></div><script>var noop=function(){};console.log=noop;console.warn=noop;console.info=noop;console.debug=noop;molstar.Viewer.create('app',{layoutIsExpanded:false,layoutShowControls:false}).then(v=>{const pdb1=\`${selectedRFD3Design.pdb_content.replace(/`/g,'\\\\`').replace(/\$/g,'\\\\$')}\`;const pdb2=\`${rf3Results.predicted_pdb.replace(/`/g,'\\\\`').replace(/\$/g,'\\\\$')}\`;Promise.all([v.loadStructureFromData(pdb1,'pdb'),v.loadStructureFromData(pdb2,'pdb')]).then(([s1,s2])=>{v.visual.update({structure:s1},{type:'cartoon',color:{r:245,g:158,b:11},opacity:0.7});v.visual.update({structure:s2},{type:'cartoon',color:{r:59,g:130,b:246},opacity:0.7})})})</script></body></html>`} className="w-full h-full border-0" title="RFD3 vs RF3 Comparison" />
                    </div>
                  ) : (
                    <div className="h-64 flex items-center justify-center text-gray-400 text-sm">
                      {!selectedRFD3Design?.pdb_content ? '请先在RFD3结果中选择一个设计' : '请先运行RF3验证'}
                    </div>
                  )}
                </div>

                {/* Per-residue RMSD Chart */}
                {rf3Results.per_res_rmsd && rf3Results.per_res_rmsd.length > 0 && (
                  <div className="bg-white border border-gray-200 rounded-xl p-5">
                    <div className="flex justify-between items-center mb-4">
                      <span className="text-sm font-semibold text-gray-700">逐残基 RMSD</span>
                      <span className="text-xs text-gray-400">阈值: 2.0 Å</span>
                    </div>
                    <div className="flex items-end gap-[1px] h-[100px] pb-1 border-b border-gray-200">
                      {rf3Results.per_res_rmsd.map((val, i) => {
                        const color = val < 1 ? '#059669' : val < 2 ? '#3b82f6' : val < 4 ? '#d97706' : '#dc2626'
                        const height = Math.min(val / 8 * 100, 100)
                        return (
                          <div key={i} title={`Res ${i + 1}: ${val.toFixed(2)} Å`} style={{ width: `${Math.max(2, 600 / rf3Results.per_res_rmsd.length)}px`, height: `${height}%`, backgroundColor: color, borderRadius: '1px 1px 0 0', minHeight: '1px' }} />
                        )
                      })}
                    </div>
                    <div className="flex justify-between mt-1 text-[10px] text-gray-400">
                      <span>N-term</span><span>C-term</span>
                    </div>
                    <div className="flex gap-4 mt-3 text-[11px] text-gray-500">
                      <span><span className="inline-block w-2 h-2 rounded-sm mr-1" style={{ background: '#059669' }} />&lt;1 Å</span>
                      <span><span className="inline-block w-2 h-2 rounded-sm mr-1" style={{ background: '#3b82f6' }} />1-2 Å</span>
                      <span><span className="inline-block w-2 h-2 rounded-sm mr-1" style={{ background: '#d97706' }} />2-4 Å</span>
                      <span><span className="inline-block w-2 h-2 rounded-sm mr-1" style={{ background: '#dc2626' }} />&gt;4 Å</span>
                    </div>
                  </div>
                )}

                {/* Confidence Details */}
                <div className="bg-white border border-gray-200 rounded-xl p-5">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">置信度详情</h4>
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { label: 'Overall pLDDT', value: (rf3Results.summary?.overall_plddt ?? 0).toFixed(3) },
                      { label: 'Overall PAE', value: (rf3Results.summary?.overall_pae ?? 0).toFixed(2) },
                      { label: 'Overall PDE', value: (rf3Results.summary?.overall_pde ?? 0).toFixed(2) },
                      { label: 'pTM', value: (rf3Results.summary?.ptm ?? 0).toFixed(4) },
                      { label: 'ipTM', value: (rf3Results.summary?.iptm ?? 0).toFixed(4) },
                      { label: 'Ranking Score', value: (rf3Results.summary?.ranking_score ?? 0).toFixed(4) },
                      { label: 'Chain pTM', value: (rf3Results.summary?.chain_ptm ?? []).map((v: number) => v.toFixed(2)).join(', ') || 'N/A' },
                      { label: 'Has Clash', value: rf3Results.summary?.has_clash ? 'Yes ⚠️' : 'No ✅' },
                    ].map(({ label, value }) => (
                      <div key={label} className="flex justify-between p-2.5 bg-gray-50 rounded-lg">
                        <span className="text-xs text-gray-500">{label}</span>
                        <span className="text-xs font-mono font-medium text-gray-900">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {rf3Results.mock && (
                  <div className="p-3 bg-amber-50 rounded-xl text-xs text-amber-700">
                    ⚠️ 当前为 mock 模式，数据为模拟生成。安装 RF3 模型后可获取真实预测结果。
                  </div>
                )}
              </div>
            )}
            {!rf3Results && !isRF3Running && (
              <div className="text-center py-12 text-gray-400">
                <Shield size={48} className="mx-auto mb-3 opacity-30" />
                <p>请先在 MPNN 步骤中选择一条序列，点击「送入RF3验证」</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
