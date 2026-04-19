import { useAppStore } from '@/store/useAppStore'
import { Upload, Link2, RotateCcw } from 'lucide-react'

const AA_COLORS: Record<string, string> = {
  A: '#4ade80', R: '#f87171', N: '#60a5fa', D: '#f87171',
  C: '#facc15', Q: '#60a5fa', E: '#f87171', G: '#4ade80',
  H: '#f87171', I: '#a3e635', L: '#a3e635', K: '#f87171',
  M: '#a3e635', F: '#c084fc', P: '#fb923c', S: '#4ade80',
  T: '#4ade80', W: '#c084fc', Y: '#c084fc', V: '#a3e635',
}

export function SequenceViewer() {
  const chains = useAppStore((s) => s.chains)
  const selectedHotspots = useAppStore((s) => s.selectedHotspots)
  const toggleHotspot = useAppStore((s) => s.toggleHotspot)

  const isHotspot = (chainId: string, resIdx: number) =>
    selectedHotspots.some(h => h.chain === chainId && h.residue === resIdx + 1)

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 flex flex-col" style={{ minHeight: '420px' }}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700">序列视图</h3>
        <div className="flex gap-1.5">
          <button title="裁剪热点" className="p-1.5 hover:bg-gray-100 rounded text-gray-500 transition-colors"><Link2 size={14} /></button>
          <button title="指定热点" className="p-1.5 hover:bg-gray-100 rounded text-gray-500 transition-colors"><Upload size={14} /></button>
          <button title="重置" className="p-1.5 hover:bg-gray-100 rounded text-gray-500 transition-colors"><RotateCcw size={14} /></button>
        </div>
      </div>

      <div className="flex-1 overflow-auto border border-gray-100 rounded-lg p-3 bg-white space-y-4">
        {chains.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            请上传目标结构文件
          </div>
        ) : (
          chains.map((chain) => (
            <div key={chain.chain_id}>
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-xs font-bold text-primary-600 bg-primary-50 px-1.5 py-0.5 rounded">{chain.chain_id}</span>
                <span className="text-xs text-gray-400">protein</span>
              </div>
              <div className="leading-relaxed font-mono text-sm tracking-wider flex flex-wrap gap-x-[1px]">
                {[...chain.sequence].map((aa, i) => {
                  const idx = i + 1
                  const hot = isHotspot(chain.chain_id, idx)
                  return (
                    <span
                      key={`${chain.chain_id}-${idx}`}
                      onClick={() => toggleHotspot(chain.chain_id, idx)}
                      className={hot
                        ? 'seq-hotspot'
                        : 'seq-residue'
                      }
                      style={!hot ? { color: AA_COLORS[aa] ?? '#64748b' } : {}}
                    >
                      {aa}
                    </span>
                  )
                })}
              </div>
            </div>
          ))
        )}
      </div>

      <div className="mt-3 pt-3 border-t border-gray-100 flex items-center gap-2">
        <span className="text-xs font-semibold text-gray-600">热点</span>
        <input
          type="text"
          placeholder="输入残基编号：A/1 表示 A 链上的残基 1（例如：A/1,A/2,A/3）"
          className="flex-1 text-xs px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-primary-400 focus:ring-1 focus:ring-primary-100"
        />
      </div>
    </div>
  )
}
