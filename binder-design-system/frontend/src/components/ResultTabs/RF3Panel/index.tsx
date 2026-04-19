import { useState } from 'react'
import { useAppStore } from '@/store/useAppStore'

export function RF3Panel() {
  const rf3Results = useAppStore((s) => s.rf3Results)
  const [selectedIdx, setSelectedIdx] = useState(0)

  if (!rf3Results || rf3Results.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
        暂无RF3验证结果
      </div>
    )
  }

  const total = rf3Results.length
  const passed = rf3Results.filter(r => r.passed).length
  const passRate = total > 0 ? ((passed / total) * 100).toFixed(0) : '0'
  const bestRmsd = Math.min(...rf3Results.filter(r => r.rmsd >= 0).map(r => r.rmsd), Infinity)
  const avgPlddt = rf3Results.reduce((a, r) => a + (r.avg_plddt || 0), 0) / total

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: '总验证数', value: String(total) },
          { label: '通过率', value: `${passRate}%`, sub: `${passed}/${total}` },
          { label: '最佳RMSD', value: bestRmsd === Infinity ? 'N/A' : `${bestRmsd.toFixed(3)}Å` },
          { label: '平均pLDDT', value: avgPlddt.toFixed(1) },
        ].map(({ label, value, sub }) => (
          <div key={label} className="metric-card">
            <div className="text-xs text-gray-500 mb-1">{label}</div>
            <div className="text-xl font-bold text-gray-900">{value}</div>
            {sub && <div className="text-[11px] text-gray-400 mt-0.5">{sub}</div>}
          </div>
        ))}
      </div>

      <select
        className="w-full text-sm border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:border-primary-400"
        onChange={(e) => setSelectedIdx(Number(e.target.value))}
        value={selectedIdx}
      >
        {rf3Results.map((r, i) => (
          <option key={i} value={i}>Design {(r.design_idx ?? i) + 1}</option>
        ))}
      </select>

      {(() => {
        const r = rf3Results[selectedIdx]
        if (!r) return null

        const passed = r.passed
        const statusBg = passed ? '#ecfdf5' : '#fef2f2'
        const statusColor = passed ? '#059669' : '#dc2626'
        const statusIcon = passed ? '✅' : '❌'
        const rmsdColor = r.rmsd < 2 ? '#059669' : r.rmsd < 3 ? '#d97706' : '#dc2626'
        const plddtColor = (r.avg_plddt || 0) > 85 ? '#059669' : (r.avg_plddt || 0) > 70 ? '#d97706' : '#dc2626'

        return (
          <>
            <div className="design-card">
              <div className="flex justify-between items-center mb-3">
                <span className="font-bold text-sm text-gray-900">Design {(r.design_idx ?? selectedIdx) + 1}</span>
                <span className="text-xs font-semibold px-2.5 py-1 rounded-full" style={{ color: statusColor, backgroundColor: statusBg }}>
                  {statusIcon} {passed ? '通过' : '未通过'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3 mb-3">
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-[11px] text-gray-500">RMSD</div>
                  <div className="text-lg font-bold mt-1" style={{ color: rmsdColor }}>
                    {r.rmsd >= 0 ? `${r.rmsd.toFixed(3)}Å` : 'N/A'}
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-[11px] text-gray-500">pLDDT</div>
                  <div className="text-lg font-bold mt-1" style={{ color: plddtColor }}>{(r.avg_plddt ?? 0).toFixed(1)}</div>
                </div>
              </div>
              <div className="text-[11px] text-gray-500 mb-1.5">RMSD进度</div>
              <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${Math.max(0, Math.min(100, (1 - (r.rmsd || 0) / 5) * 100))}%`,
                    backgroundColor: rmsdColor,
                  }}
                />
              </div>
            </div>

            <RF3PLDDTChart />
            <RF3PAEHeatmap />
            <RF3RMSDChart />
          </>
        )
      })()}
    </div>
  )
}

function RF3PLDDTChart() {
  const data = Array.from({ length: 80 }, () => Math.random() * 30 + 70)
  const avg = data.reduce((a, b) => a + b, 0) / data.length
  const barWidth = Math.max(2, Math.min(8, 600 / data.length))
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5">
      <div className="flex justify-between items-center mb-4">
        <span className="text-sm font-semibold text-gray-700">pLDDT置信度</span>
        <span className="text-sm font-semibold text-primary-600">{avg.toFixed(1)}</span>
      </div>
      <div className="flex items-end gap-[1px] h-[120px] pb-1 border-b border-gray-200">
        {data.map((val, i) => (
          <div
            key={i}
            title={`Res ${i + 1}: ${val.toFixed(1)}`}
            style={{
              width: `${barWidth}px`,
              height: `${val}%`,
              backgroundColor: val > 85 ? '#059669' : val > 70 ? '#d97706' : '#dc2626',
              borderRadius: '1px 1px 0 0',
              minHeight: '1px',
            }}
          />
        ))}
      </div>
    </div>
  )
}

function RF3PAEHeatmap() {
  const n = 30
  const cellSize = Math.max(2, Math.min(8, 300 / n))
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5">
      <div className="flex justify-between items-center mb-4">
        <span className="text-sm font-semibold text-gray-700">PAE预测误差</span>
        <span className="text-xs text-gray-400">{n}×{n} 残基</span>
      </div>
      <div className="inline-grid gap-0 leading-none" style={{ gridTemplateColumns: `repeat(${n}, ${cellSize}px)` }}>
        {Array.from({ length: n * n }, (_, idx) => {
          const i = Math.floor(idx / n)
          const j = idx % n
          const norm = Math.min(Math.abs(i - j) / n, 1)
          const r = Math.round(norm * 220)
          const g = Math.round((1 - norm) * 100)
          const b = Math.round((1 - norm) * 50)
          return <div key={idx} title={`PAE(${i+1},${j+1})`} style={{ width: `${cellSize}px`, height: `${cellSize}px`, backgroundColor: `rgb(${r},${g},${b})` }} />
        })}
      </div>
    </div>
  )
}

function RF3RMSDChart() {
  const data = Array.from({ length: 80 }, () => Math.random() * 4)
  const avg = data.reduce((a, b) => a + b, 0) / data.length
  const barWidth = Math.max(2, Math.min(8, 600 / data.length))
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5">
      <div className="flex justify-between items-center mb-4">
        <span className="text-sm font-semibold text-gray-700">残基RMSD</span>
        <span className="text-sm font-semibold text-primary-600">{avg.toFixed(3)}Å</span>
      </div>
      <div className="relative flex items-end gap-[1px] h-[120px] pb-1 border-b border-gray-200">
        <div className="absolute left-0 right-0 border-t-2 border-dashed border-red-500 z-10" style={{ top: `${(1 - 2/5) * 100}%` }} />
        {data.map((val, i) => (
          <div
            key={i}
            title={`Res ${i + 1}: ${val.toFixed(3)}Å`}
            style={{
              width: `${barWidth}px`,
              height: `${Math.min(val / 5 * 100, 100)}%`,
              backgroundColor: val < 2 ? '#059669' : val < 3 ? '#d97706' : '#dc2626',
              borderRadius: '1px 1px 0 0',
              minHeight: '1px',
            }}
          />
        ))}
      </div>
    </div>
  )
}
