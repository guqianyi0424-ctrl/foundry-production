import { useAppStore } from '@/store/useAppStore'
import { Dna } from 'lucide-react'

interface DesignPanelProps {
  onOpenRFD3?: () => void
  isRFD3Running?: boolean
}

export function DesignPanel({ onOpenRFD3, isRFD3Running }: DesignPanelProps) {
  const rfd3Config = useAppStore((s) => s.rfd3Config)
  const setRfd3Config = useAppStore((s) => s.setRfd3Config)
  const selectedRange = useAppStore((s) => s.selectedRange)

  const syncTargetFromRange = () => {
    if (!selectedRange) return
    const val = `${selectedRange.chain}/${selectedRange.startResSeq}-${selectedRange.endResSeq}`
    setRfd3Config({ targetStructure: val })
  }

  if (selectedRange && !rfd3Config.targetStructure) {
    syncTargetFromRange()
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-gray-900">RFD3 Binder 设计参数</span>
      </div>

      <div className="grid grid-cols-[100px_1fr] gap-y-4 gap-x-4 items-start">
        <label className="text-sm text-gray-700 pt-2 flex items-center gap-1 shrink-0">
          <span className="text-red-500">*</span>目标实体
        </label>
        <div className="flex items-center gap-3">
          <select value={rfd3Config.targetEntityType} onChange={(e) => setRfd3Config({ targetEntityType: e.target.value })} className="border border-gray-200 rounded-lg px-3 py-2 text-sm w-28 focus:outline-none focus:border-primary-400 bg-white">
            <option value="蛋白">蛋白</option>
          </select>
          <input type="text" value={rfd3Config.targetStructure} onChange={(e) => setRfd3Config({ targetStructure: e.target.value })} placeholder="结构域*（例如：A/6-36）" className={`flex-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary-400 ${!rfd3Config.targetStructure ? 'border-red-300' : 'border-gray-200'}`} />
          {selectedRange && (
            <button onClick={syncTargetFromRange} className="shrink-0 px-3 py-2 text-xs bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition-colors whitespace-nowrap">同步选择</button>
          )}
        </div>

        <label className="text-sm text-gray-700 pt-2 flex items-center gap-1 shrink-0">
          热点
        </label>
        <input type="text" value={rfd3Config.hotspots} onChange={(e) => setRfd3Config({ hotspots: e.target.value })} placeholder='输入链残基编号：A/1 表示 A 链上的残基 1（例如：A/1, A/2, A/3）' className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary-400" />

        <label className="text-sm text-gray-700 pt-2 flex items-center gap-1 shrink-0">
          条件原子
        </label>
        <input type="text" value={rfd3Config.conditionAtoms} onChange={(e) => setRfd3Config({ conditionAtoms: e.target.value })} placeholder='输入残基索引和分离的原子（例如：{"A/1":["NE2","CD2","CE1"]}）' className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary-400 font-mono text-xs" />

        <label className="text-sm text-gray-700 pt-2 flex items-center gap-1 shrink-0">
          <span className="text-red-500">*</span>设计实体长度
        </label>
        <div className="space-y-1.5">
          <input type="text" value={`${rfd3Config.lengthMin}-${rfd3Config.lengthMax}`} onChange={(e) => {
            const m = e.target.value.match(/^(\d+)\s*-\s*(\d+)$/)
            if (m) {
              setRfd3Config({ lengthMin: parseInt(m[1]), lengthMax: parseInt(m[2]) })
            }
          }} placeholder="长度范围*（例如：20-20 表示固定长度 20；1-100 表示在范围内随机）" className={`w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary-400`} />
          <p className="text-xs text-gray-400">长度范围（例如：20-20 表示固定长度 20；1-100 表示在范围内随机）</p>
        </div>

        <label className="text-sm text-gray-700 pt-2 flex items-center gap-1 shrink-0">
          <span className="text-red-500">*</span>生成参数
        </label>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-600 whitespace-nowrap">n_batches:</label>
            <input type="number" min={1} max={50} value={rfd3Config.nBatches} onChange={(e) => setRfd3Config({ nBatches: Math.max(1, parseInt(e.target.value) || 1) })} className="w-20 border border-gray-200 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:border-primary-400" />
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-600 whitespace-nowrap">diffusion_batch_size:</label>
            <input type="number" min={1} max={64} value={rfd3Config.diffusionBatchSize} onChange={(e) => setRfd3Config({ diffusionBatchSize: Math.max(1, parseInt(e.target.value) || 16) })} className="w-20 border border-gray-200 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:border-primary-400" />
          </div>
        </div>

        <div></div>
        <div className="pt-2 flex items-center gap-3">
          {onOpenRFD3 && (
            <button onClick={onOpenRFD3} disabled={!rfd3Config.targetStructure || isRFD3Running} className={`flex items-center gap-2 px-5 py-2 rounded-lg text-white text-sm font-medium transition-all ${!rfd3Config.targetStructure || isRFD3Running ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 shadow-sm'}`}>
              <Dna size={16} />{isRFD3Running ? '任务运行中...' : '新建任务'}
            </button>
          )}
          <button onClick={() => {
            alert(`目标实体: ${rfd3Config.targetEntityType} / ${rfd3Config.targetStructure || '(未填)'}\n热点: ${rfd3Config.hotspots || '(未填)'}\n条件原子: ${rfd3Config.conditionAtoms || '(未填)'}\n长度: ${rfd3Config.lengthMin}-${rfd3Config.lengthMax}\nn_batches: ${rfd3Config.nBatches}\ndiffusion_batch_size: ${rfd3Config.diffusionBatchSize}\n\n总生成数: ${rfd3Config.nBatches * rfd3Config.diffusionBatchSize} 个`)
          }} disabled={!rfd3Config.targetStructure} className={`px-5 py-2 rounded-lg text-white text-sm font-medium transition-all ${!rfd3Config.targetStructure ? 'bg-gray-300 cursor-not-allowed' : 'bg-primary-600 hover:bg-primary-700 shadow-sm'}`}>
            验证配置
          </button>
        </div>
      </div>
    </div>
  )
}
