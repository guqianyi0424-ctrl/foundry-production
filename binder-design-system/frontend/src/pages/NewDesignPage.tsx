import { useCallback, useState } from 'react'
import { useAppStore } from '@/store/useAppStore'
import { SequenceViewer } from '@/components/SequenceViewer'
import { MolstarViewer } from '@/components/MolstarViewer'
import { DesignPanel } from '@/components/DesignPanel'
import { RFD3Panel } from '@/components/ResultTabs/RFD3Panel'
import { MPNNPanel } from '@/components/ResultTabs/MPNNPanel'
import { RF3Panel } from '@/components/ResultTabs/RF3Panel'
import { uploadPdb, predictHotspot, runPipeline } from '@/api'
import { parseStructureFile } from '@/utils/pdbParser'
import { Upload, Play, RotateCcw, Sparkles, Scissors, Target, RefreshCw } from 'lucide-react'

const TABS = ['② RFD3 主链生成', '③ MPNN 序列设计', '④ RF3 结构验证'] as const

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
  const setBinderLength = useAppStore((s) => s.setBinderLength)
  const rfd3Results = useAppStore((s) => s.rfd3Results)
  const isRunning = useAppStore((s) => s.isRunning)
  const setIsRunning = useAppStore((s) => s.setIsRunning)
  const setRfd3Results = useAppStore((s) => s.setRfd3Results)
  const setMpnnResults = useAppStore((s) => s.setMpnnResults)
  const setRf3Results = useAppStore((s) => s.setRf3Results)
  const addJob = useAppStore((s) => s.addJob)
  const resetAll = useAppStore((s) => s.resetAll)

  const [activeTab, setActiveTab] = useState(0)
  const [taskType, setTaskType] = useState('蛋白')
  const [inputMode, setInputMode] = useState<'upload' | 'input'>('upload')

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (file.size > 200 * 1024 * 1024) {
      alert('文件大小不能超过200MB')
      return
    }
    setTargetFile(file)

    const content = await file.text()

    try {
      const chains = parseStructureFile(content, file.name)
      if (chains.length === 0) {
        alert('文件解析失败：未找到有效的氨基酸残基，请检查文件格式')
        return
      }
      setChains(chains)
      setPdbContent(content)
    } catch {
      try {
        const res = await uploadPdb(file)
        const chainsWithSeqs = res.chains.map(c => ({
          ...c,
          resSeqs: c.resSeqs ?? Array.from({ length: c.length }, (_, i) => i + 1),
        }))
        setChains(chainsWithSeqs)
        setPdbContent(res.pdb_content)
      } catch {
        alert('文件解析失败，请检查格式是否为有效的 PDB/CIF 文件')
      }
    }
  }, [setTargetFile, setChains, setPdbContent])

  const handlePredictHotspot = useCallback(async () => {
    if (!pdbContent) return
    try {
      const res = await predictHotspot(pdbContent)
      setSelectedHotspots(res.hotspots.map(h => ({ chain: h.chain, residue: h.residue, score: h.score })))
    } catch {
      alert('热点预测失败，请检查环境配置')
    }
  }, [pdbContent, setSelectedHotspots])

  const handleTrimTarget = () => {
    if (!selectedRange) {
      alert('请先在序列视图中拖拽选择一个残基范围')
      return
    }
    const targetStr = `${selectedRange.chain}/${selectedRange.startResSeq}-${selectedRange.endResSeq}`
    const confirmed = confirm(`裁剪结构为 ${targetStr}，是否确认？`)
    if (confirmed) {
      setRfd3Config({ targetStructure: targetStr })
    }
  }

  const handleSpecifyHotspot = () => {
    if (selectedHotspots.length === 0) {
      alert('请先在序列视图中单击选择热点残基')
      return
    }
    const hotspotStr = selectedHotspots.map(h => `${h.chain}/${h.residue}`).join(', ')
    const confirmed = confirm(`指定热点: ${hotspotStr}，是否确认？`)
    if (confirmed) {
      setRfd3Config({ hotspots: hotspotStr })
    }
  }

  const handleResetView = () => {
    resetAll()
  }

  const handleRun = useCallback(async () => {
    if (!pdbContent || selectedHotspots.length === 0) {
      alert('请上传目标蛋白并选择热点残基')
      return
    }
    setIsRunning(true)
    try {
      const res = await runPipeline({
        pdb_content: pdbContent,
        hotspots: selectedHotspots.map(h => ({ chain: h.chain, residue: h.residue })),
        binder_length: binderLength,
      })
      if (res.rfd3_results?.success) {
        setRfd3Results(res.rfd3_results.designs)
      }
      setMpnnResults(res.mpnn_results ?? null)
      setRf3Results(res.rf3_results ?? null)
      addJob({ id: res.job_id || Date.now().toString(), name: targetFile?.name ?? 'Design', status: 'completed', time: new Date().toLocaleString() })
      setActiveTab(0)
    } catch (err) {
      alert('运行失败: ' + String(err))
    } finally {
      setIsRunning(false)
    }
  }, [pdbContent, selectedHotspots, binderLength, setIsRunning, setRfd3Results, setMpnnResults, setRf3Results, addJob, targetFile])

  return (
    <div className="p-6 max-w-[1400px] mx-auto space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex bg-gray-100 rounded-lg p-0.5">
            {['表单模式', 'JSON模式'].map(mode => (
              <button key={mode} className="px-4 py-1.5 text-xs font-medium rounded-md transition-all text-gray-600 hover:text-gray-900">{mode}</button>
            ))}
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-red-500">*</span><span>任务类型</span>
            <select value={taskType} onChange={e => setTaskType(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-primary-400">
              <option value="蛋白">蛋白</option>
            </select>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleRun} disabled={isRunning} className={`flex items-center gap-2 px-5 py-2 rounded-lg text-white text-sm font-medium transition-all ${isRunning ? 'bg-gray-400 cursor-not-allowed' : 'bg-primary-600 hover:bg-primary-700 shadow-sm'}`}>
            <Play size={16} />{isRunning ? '运行中...' : '运行'}
          </button>
          <button onClick={handleResetView} className="flex items-center gap-2 px-4 py-2 rounded-lg border border-gray-200 text-gray-700 text-sm hover:bg-gray-50 transition-all">
            <RotateCcw size={16} />重置
          </button>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-red-500">*</span><span className="text-sm font-semibold">目标结构</span>
        </div>
        <div className="flex items-center gap-3 mb-2">
          <label className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${inputMode === 'upload' ? 'border-primary-400 bg-primary-50 text-primary-700' : 'border-gray-200 text-gray-600'} cursor-pointer text-sm transition-all`}>
            <Upload size={16} />
            <span>上传文件</span>
            <input type="file" accept=".pdb,.cif,.ent" onChange={handleUpload} className="hidden" />
          </label>
          <button
            onClick={() => setInputMode(inputMode === 'upload' ? 'input' : 'upload')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm transition-all ${inputMode === 'input' ? 'border-primary-400 bg-primary-50 text-primary-700' : 'border-gray-200 text-gray-600 hover:bg-gray-50'}`}
          >
            输入
          </button>
          <span className="text-xs text-gray-400">支持pdb/cif格式文件，文件不得超过200MB</span>
        </div>

        {targetFile && (
          <div className="flex items-center gap-2 mt-3 p-2.5 bg-gray-50 rounded-lg border border-gray-100 w-fit">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14,2 14,8 20,8"/></svg>
            <span className="text-sm text-gray-700 font-medium">{targetFile.name}</span>
            <button onClick={() => { setTargetFile(null); setPdbContent(''); setChains([]) }} className="ml-2 text-red-500 hover:text-red-700"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>
          </div>
        )}
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-bold text-gray-900 flex items-center gap-2">
            <span>🧬</span> 结构与序列
          </h3>
          <div className="flex gap-2">
            <button onClick={handlePredictHotspot} disabled={!pdbContent || isRunning} title="预测热点" className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition-all ${!pdbContent || isRunning ? 'border-gray-100 text-gray-300 cursor-not-allowed' : 'border-gray-200 text-gray-600 hover:bg-blue-50 hover:border-blue-200 hover:text-blue-600'}`}>
              <Sparkles size={14} />预测热点
            </button>
            <button onClick={handleTrimTarget} disabled={!selectedRange} title="裁剪靶点" className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition-all ${!selectedRange ? 'border-gray-100 text-gray-300 cursor-not-allowed' : 'border-gray-200 text-gray-600 hover:bg-orange-50 hover:border-orange-200 hover:text-orange-600'}`}>
              <Scissors size={14} />裁剪靶点
            </button>
            <button onClick={handleSpecifyHotspot} disabled={selectedHotspots.length === 0} title="指定热点" className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition-all ${selectedHotspots.length === 0 ? 'border-gray-100 text-gray-300 cursor-not-allowed' : 'border-gray-200 text-gray-600 hover:bg-green-50 hover:border-green-200 hover:text-green-600'}`}>
              <Target size={14} />指定热点
            </button>
            <button onClick={() => { setSelectedRange(null); setSelectedHotspots([]) }} disabled={selectedHotspots.length === 0 && !selectedRange} title="重置" className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition-all ${selectedHotspots.length === 0 && !selectedRange ? 'border-gray-100 text-gray-300 cursor-not-allowed' : 'border-gray-200 text-gray-600 hover:bg-gray-50'}`}>
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

      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-bold text-gray-900 flex items-center gap-2">
            <span>📊</span> 设计结果
          </h3>
        </div>

        <div className="flex border-b border-gray-200">
          {TABS.map((tab, i) => (
            <button
              key={tab}
              onClick={() => setActiveTab(i)}
              className={`px-6 py-3 text-sm transition-all ${activeTab === i ? 'tab-active' : 'tab-inactive'}`}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className="p-5">
          {activeTab === 0 && <RFD3Panel />}
          {activeTab === 1 && <MPNNPanel />}
          {activeTab === 2 && <RF3Panel />}
        </div>
      </div>
    </div>
  )
}
