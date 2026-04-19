import { useState } from 'react'
import { useAppStore } from '@/store/useAppStore'

export function RFD3Panel() {
  const rfd3Results = useAppStore((s) => s.rfd3Results)
  const [selectedIdx, setSelectedIdx] = useState(0)

  if (!rfd3Results || rfd3Results.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
        暂无RFD3结果
      </div>
    )
  }

  const selected = rfd3Results[selectedIdx]
  const plddtColor = selected.plddt > 85 ? '#059669' : selected.plddt > 70 ? '#d97706' : '#dc2626'

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-4">
        {rfd3Results.map((d, i) => (
          <button
            key={i}
            onClick={() => setSelectedIdx(i)}
            className={`design-card text-left ${selectedIdx === i ? 'border-primary-400 ring-1 ring-primary-200' : ''}`}
          >
            <div className="flex justify-between items-center mb-2">
              <span className="font-bold text-sm text-gray-900">Design {d.index + 1}</span>
              {d.rank && <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">Rank #{d.rank}</span>}
            </div>
            <div className="text-xs text-gray-500 mb-2">pLDDT</div>
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-bold" style={{ color: plddtColor }}>{selected.plddt.toFixed(1)}</span>
              <span className="text-xs text-gray-400">/100</span>
            </div>
            <div className="mt-2 w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{ width: `${Math.min(selected.plddt, 100)}%`, backgroundColor: plddtColor }}
              />
            </div>
          </button>
        ))}
      </div>

      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden" style={{ height: '450px' }}>
        <div className="px-4 py-2.5 border-b border-gray-100 flex items-center justify-between">
          <span className="text-sm font-semibold text-gray-700">Design {selected.index + 1} - 3D结构</span>
          <span className="text-xs text-gray-400">Molstar Viewer</span>
        </div>
        <div className="h-[395px] bg-[#fafbfc] flex items-center justify-center text-gray-400 text-sm">
          3D结构查看器 (加载PDB后显示)
        </div>
      </div>

      <RFD3PLDDTChart designIndex={selected.index} />
    </div>
  )
}

function RFD3PLDDTChart({ designIndex }: { designIndex: number }) {
  const data = Array.from({ length: 80 }, () => Math.random() * 30 + 70)
  const avg = data.reduce((a, b) => a + b, 0) / data.length
  const avgColor = avg > 85 ? '#059669' : avg > 70 ? '#d97706' : '#dc2626'
  const barWidth = Math.max(4, Math.min(8, 600 / data.length))

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5">
      <div className="flex justify-between items-center mb-4">
        <span className="text-sm font-semibold text-gray-700">Design {designIndex + 1} - pLDDT置信度</span>
        <span className="text-sm font-semibold" style={{ color: avgColor }}>平均: {avg.toFixed(1)}</span>
      </div>
      <div className="flex items-end gap-[1px] h-[120px] pb-1 border-b border-gray-200">
        {data.map((val, i) => {
          const color = val > 85 ? '#059669' : val > 70 ? '#d97706' : '#dc2626'
          return (
            <div
              key={i}
              title={`Res ${i + 1}: ${val.toFixed(1)}`}
              style={{
                width: `${barWidth}px`,
                height: `${val}%`,
                backgroundColor: color,
                borderRadius: '1px 1px 0 0',
                minHeight: '1px',
              }}
            />
          )
        })}
      </div>
      <div className="flex justify-between mt-1 text-[10px] text-gray-400">
        <span>N-term</span><span>C-term</span>
      </div>
      <div className="flex gap-4 mt-3 text-[11px] text-gray-500">
        <span><span className="inline-block w-2 h-2 rounded-sm mr-1" style={{ background: '#059669' }} />&gt;85 高置信</span>
        <span><span className="inline-block w-2 h-2 rounded-sm mr-1" style={{ background: '#d97706' }} />70-85 中等</span>
        <span><span className="inline-block w-2 h-2 rounded-sm mr-1" style={{ background: '#dc2626' }} />&lt;70 低置信</span>
      </div>
    </div>
  )
}
