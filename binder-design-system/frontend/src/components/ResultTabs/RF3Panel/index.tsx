import { useAppStore } from '@/store/useAppStore'

export function RF3Panel() {
  const rf3Results = useAppStore((s) => s.rf3Results)

  if (!rf3Results || !rf3Results.success) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
        暂无RF3验证结果
      </div>
    )
  }

  const rmsdColor = rf3Results.rmsd < 2 ? '#059669' : rf3Results.rmsd < 5 ? '#d97706' : '#dc2626'
  const plddtColor = rf3Results.avg_plddt > 80 ? '#059669' : rf3Results.avg_plddt > 60 ? '#d97706' : '#dc2626'

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-4 gap-4">
        <div className="metric-card">
          <div className="text-xs text-gray-500 mb-1">RMSD</div>
          <div className="text-xl font-bold" style={{ color: rmsdColor }}>{rf3Results.rmsd.toFixed(2)} Å</div>
          <div className="text-[11px] mt-0.5" style={{ color: rmsdColor }}>{rf3Results.rmsd_interpretation}</div>
        </div>
        <div className="metric-card">
          <div className="text-xs text-gray-500 mb-1">pLDDT</div>
          <div className="text-xl font-bold" style={{ color: plddtColor }}>{rf3Results.avg_plddt.toFixed(1)}</div>
        </div>
        <div className="metric-card">
          <div className="text-xs text-gray-500 mb-1">pTM</div>
          <div className="text-xl font-bold text-blue-600">{(rf3Results.summary?.ptm ?? 0).toFixed(3)}</div>
        </div>
        <div className="metric-card">
          <div className="text-xs text-gray-500 mb-1">Ranking</div>
          <div className="text-xl font-bold text-gray-900">{(rf3Results.summary?.ranking_score ?? 0).toFixed(3)}</div>
        </div>
      </div>
      <div className={`p-3 rounded-xl ${rf3Results.passed ? 'bg-green-50 border border-green-200' : 'bg-amber-50 border border-amber-200'}`}>
        <span className={`text-sm font-semibold ${rf3Results.passed ? 'text-green-800' : 'text-amber-800'}`}>
          {rf3Results.passed ? '✅ 验证通过' : '⚠️ 需要优化'}
        </span>
        <span className="ml-2 text-xs text-gray-600">
          RMSD {rf3Results.rmsd.toFixed(2)} Å {rf3Results.passed ? '<' : '≥'} 2.0 Å
        </span>
      </div>
      {rf3Results.per_res_rmsd && rf3Results.per_res_rmsd.length > 0 && (
        <RF3RMSDChart data={rf3Results.per_res_rmsd} />
      )}
    </div>
  )
}

function RF3RMSDChart({ data }: { data: number[] }) {
  const avg = data.reduce((a, b) => a + b, 0) / data.length
  const barWidth = Math.max(2, Math.min(8, 600 / data.length))
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5">
      <div className="flex justify-between items-center mb-4">
        <span className="text-sm font-semibold text-gray-700">残基RMSD</span>
        <span className="text-sm font-semibold text-primary-600">{avg.toFixed(2)} Å</span>
      </div>
      <div className="relative flex items-end gap-[1px] h-[120px] pb-1 border-b border-gray-200">
        <div className="absolute left-0 right-0 border-t-2 border-dashed border-red-400 z-10" style={{ top: `${(1 - 2 / 8) * 100}%` }} />
        {data.map((val, i) => {
          const color = val < 1 ? '#059669' : val < 2 ? '#3b82f6' : val < 4 ? '#d97706' : '#dc2626'
          const height = Math.min(val / 8 * 100, 100)
          return (
            <div key={i} title={`Res ${i + 1}: ${val.toFixed(2)} Å`}
              style={{ width: `${barWidth}px`, height: `${height}%`, backgroundColor: color, borderRadius: '1px 1px 0 0', minHeight: '1px' }} />
          )
        })}
      </div>
      <div className="flex justify-between mt-1 text-[10px] text-gray-400"><span>N-term</span><span>C-term</span></div>
      <div className="flex gap-4 mt-3 text-[11px] text-gray-500">
        <span><span className="inline-block w-2 h-2 rounded-sm mr-1" style={{ background: '#059669' }} />&lt;1 Å</span>
        <span><span className="inline-block w-2 h-2 rounded-sm mr-1" style={{ background: '#3b82f6' }} />1-2 Å</span>
        <span><span className="inline-block w-2 h-2 rounded-sm mr-1" style={{ background: '#d97706' }} />2-4 Å</span>
        <span><span className="inline-block w-2 h-2 rounded-sm mr-1" style={{ background: '#dc2626' }} />&gt;4 Å</span>
      </div>
    </div>
  )
}
