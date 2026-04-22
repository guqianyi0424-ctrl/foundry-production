import { useCallback, useState } from 'react'
import { useAppStore } from '@/store/useAppStore'
import { SequenceViewer } from '@/components/SequenceViewer'
import { MolstarViewer } from '@/components/MolstarViewer'
import { DesignPanel } from '@/components/DesignPanel'
import { uploadPdb, predictHotspot, runRFD3, runMPNN } from '@/api'
import type { PredictHotspotResponse, RFD3Design, MPNNSequence } from '@/api'
import { parseStructureFile } from '@/utils/pdbParser'
import { Upload, Play, RotateCcw, Sparkles, Scissors, Target, RefreshCw, X, CheckCircle2, Dna, FlaskConical, ChevronRight } from 'lucide-react'

interface HotspotPredictionModalProps {
  open: boolean
  onClose: () => void
  onConfirm: () => void
  prediction: PredictHotspotResponse | null
}

function HotspotPredictionModal({ open, onClose, onConfirm, prediction }: HotspotPredictionModalProps) {
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
            💡 预测的热点残基已同步到3D结构中高亮显示，点击「确认」将自动填入热点框
          </div>
        </div>
        <div className="p-4 border-t border-gray-100 flex justify-end gap-3">
          <button onClick={onClose} className="px-5 py-2 rounded-lg border border-gray-200 text-gray-600 text-sm hover:bg-gray-50 transition-colors">取消</button>
          <button onClick={onConfirm} className="px-5 py-2 rounded-lg bg-green-600 text-white text-sm font-medium hover:bg-green-700 transition-colors">确认填入</button>
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
  const setRfd3Config = useAppStore((s) => s.setRfd3Config)
  const binderLength = useAppStore((s) => s.binderLength)
  const rfd3Config = useAppStore((s) => s.rfd3Config)
  const rfd3Results = useAppStore((s) => s.rfd3Results)
  const setRfd3Results = useAppStore((s) => s.setRfd3Results)
  const mpnnResults = useAppStore((s) => s.mpnnResults)
  const setMpnnResults = useAppStore((s) => s.setMpnnResults)
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
  const [selectedRFD3Design, setSelectedRFD3Design] = useState<RFD3Design | null>(null)
  const [selectedMPNNSeq, setSelectedMPNNSeq] = useState<MPNNSequence | null>(null)
  const [activeStep, setActiveStep] = useState(0)

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
      setPredictionResult(res)
      setShowPredictionModal(true)
    } catch { alert('热点预测失败') }
    finally { setIsPredicting(false) }
  }, [pdbContent])

  const handleConfirmPrediction = useCallback(() => {
    if (!predictionResult) return
    setSelectedHotspots(predictionResult.hotspots.map(h => ({ chain: h.chain, residue: h.residue, score: h.score })))
    setRfd3Config({ hotspots: predictionResult.hotspots.map(h => `${h.chain}/${h.residue}`).join(', ') })
    setShowPredictionModal(false)
  }, [predictionResult, setSelectedHotspots, setRfd3Config])

  const handleTrimTarget = () => {
    if (!selectedRange) { alert('请先选择残基范围'); return }
    const targetStr = `${selectedRange.chain}/${selectedRange.startResSeq}-${selectedRange.endResSeq}`
    if (confirm(`裁剪结构为 ${targetStr}，是否确认？`)) setRfd3Config({ targetStructure: targetStr })
  }

  const handleSpecifyHotspot = () => {
    if (selectedHotspots.length === 0) { alert('请先选择热点残基'); return }
    const hotspotStr = selectedHotspots.map(h => `${h.chain}/${h.residue}`).join(', ')
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
        hotspots: selectedHotspots.map(h => `${h.chain}${h.residue}`) || undefined,
        binder_length: params.binder_length,
        diffusion_batch_size: params.diffusion_batch_size,
        n_batches: params.n_batches,
      })
      setRfd3Results(res)
      setActiveStep(1)
    } catch (err) { alert('RFD3运行失败: ' + String(err)) }
    finally { setIsRFD3Running(false) }
  }, [pdbContent, rfd3Config, selectedHotspots, setRfd3Results])

  const handleRunMPNN = useCallback(async (backbonePdb: string) => {
    setIsMPNNRunning(true)
    try {
      const targetChains = [...new Set(selectedHotspots.map(h => h.chain))]
      const res = await runMPNN({
        backbone_pdb_content: backbonePdb,
        batch_size: 10,
        fixed_chains: targetChains.length > 0 ? targetChains : undefined,
      })
      setMpnnResults(res)
      setActiveStep(2)
    } catch (err) { alert('MPNN运行失败: ' + String(err)) }
    finally { setIsMPNNRunning(false) }
  }, [selectedHotspots, setMpnnResults])

  return (
    <div className="p-6 max-w-[1400px] mx-auto space-y-5">
      <HotspotPredictionModal open={showPredictionModal} onClose={() => setShowPredictionModal(false)} onConfirm={handleConfirmPrediction} prediction={predictionResult} />
      <RFD3Modal open={showRFD3Modal} onClose={() => setShowRFD3Modal(false)} onRun={handleRunRFD3} target={rfd3Config.targetStructure} hotspots={selectedHotspots.map(h => `${h.chain}/${h.residue}`)} isRunning={isRFD3Running} />

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
          <button onClick={() => setShowRFD3Modal(true)} disabled={!pdbContent || isRFD3Running} className={`flex items-center gap-2 px-5 py-2 rounded-lg text-white text-sm font-medium transition-all ${!pdbContent || isRFD3Running ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 shadow-sm'}`}>
            <Dna size={16} />{isRFD3Running ? 'RFD3生成中...' : 'RFD3 骨架生成'}
          </button>
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
            <button onClick={handleSpecifyHotspot} disabled={selectedHotspots.length === 0} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition-all ${selectedHotspots.length === 0 ? 'border-gray-100 text-gray-300 cursor-not-allowed' : 'border-gray-200 text-gray-600 hover:bg-green-50 hover:text-green-600'}`}>
              <Target size={14} />指定热点
            </button>
            <button onClick={() => { setSelectedRange(null); setSelectedHotspots([]) }} disabled={selectedHotspots.length === 0 && !selectedRange} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition-all ${selectedHotspots.length === 0 && !selectedRange ? 'border-gray-100 text-gray-300 cursor-not-allowed' : 'border-gray-200 text-gray-600 hover:bg-gray-50'}`}>
              <RefreshCw size={14} />重置
            </button>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-5">
          <SequenceViewer />
          <MolstarViewer />
        </div>
      </div>

      <DesignPanel />

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
                            <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700">pLDDT: {design.plddt?.toFixed(1) || 'N/A'}</span>
                          </div>
                          {design.pdb_content && (
                            <div className="h-32 rounded-lg overflow-hidden border border-gray-100 bg-gray-50">
                              <iframe srcDoc={`<!DOCTYPE html><html><head><script src="https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/viewer/molstar.js"></script><style>body{margin:0;padding:0;overflow:hidden}#app{width:100%;height:100%}</style></head><body><div id="app"></div><script>molstar.Viewer.create('app',{layoutIsExpanded:false}).then(v=>{const pdbData=\`${design.pdb_content.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`;v.loadStructureFromData(pdbData,'pdb')})</script></body></html>`} className="w-full h-full border-0" title={design.name} />
                            </div>
                          )}
                          <div className="mt-2 flex justify-end">
                            <button onClick={(e) => { e.stopPropagation(); handleRunMPNN(design.pdb_content) }} disabled={isMPNNRunning} className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs bg-green-50 text-green-700 hover:bg-green-100 transition-all disabled:opacity-50">
                              <FlaskConical size={12} />{isMPNNRunning ? 'MPNN运行中...' : '送入MPNN'}
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
                          <iframe srcDoc={`<!DOCTYPE html><html><head><script src="https://cdn.jsdelivr.net/npm/molstar@4.4.0/build/viewer/molstar.js"></script><style>body{margin:0;padding:0;overflow:hidden}#app{width:100%;height:100%}</style></head><body><div id="app"></div><script>molstar.Viewer.create('app',{layoutIsExpanded:false}).then(v=>{const pdbData=\`${seq.pdb_content.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`;v.loadStructureFromData(pdbData,'pdb')})</script></body></html>`} className="w-full h-full border-0" title={`Sequence ${seq.index + 1}`} />
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
          <div className="text-center py-12 text-gray-400">
            <p>RF3 结构验证步骤待实现</p>
          </div>
        )}
      </div>
    </div>
  )
}
