import { useAppStore } from '@/store/useAppStore'

const AA_COLORS: Record<string, string> = {
  A: '#4ade80', R: '#f87171', N: '#60a5fa', D: '#f87171',
  C: '#facc15', Q: '#60a5fa', E: '#f87171', G: '#4ade80',
  H: '#f87171', I: '#a3e635', L: '#a3e635', K: '#f87171',
  M: '#a3e635', F: '#c084fc', P: '#fb923c', S: '#4ade80',
  T: '#4ade80', W: '#c084fc', Y: '#c084fc', V: '#a3e635',
}

export function MPNNPanel() {
  const mpnnResults = useAppStore((s) => s.mpnnResults)

  if (!mpnnResults || mpnnResults.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
        暂无MPNN结果
      </div>
    )
  }

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        {mpnnResults.map((m, i) => {
          const scoreColor = m.score > 0.8 ? '#059669' : m.score > 0.5 ? '#d97706' : '#dc2626'
          const seqDisplay = m.sequence.length > 80 ? m.sequence.slice(0, 77) + '...' : m.sequence

          return (
            <div key={i} className="design-card">
              <div className="flex justify-between items-center mb-2.5">
                <span className="text-sm font-bold text-gray-900">Design {m.design_idx + 1} - Seq {m.seq_idx + 1}</span>
                <span
                  className="text-xs font-semibold px-2.5 py-1 rounded-full"
                  style={{ color: scoreColor, backgroundColor: scoreColor + '18' }}
                >
                  得分: {m.score.toFixed(3)}
                </span>
              </div>
              <div className="leading-relaxed font-mono text-[13px] tracking-wide break-all">
                {[...seqDisplay].map((aa, j) => (
                  <span key={j} style={{ color: AA_COLORS[aa] ?? '#64748b', fontWeight: 600 }}>{aa}</span>
                ))}
              </div>
              <div className="mt-2 text-xs text-gray-400">长度: {m.sequence.length} aa</div>
            </div>
          )
        })}
      </div>

      <MPNNScoreChart results={mpnnResults} />
      <MPNNLegend />
    </div>
  )
}

function MPNNScoreChart({ results }: { results: Array<{ sequence: string; score: number }> }) {
  const maxScore = Math.max(...results.map(r => r.score), 1)
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5">
      <h4 className="text-sm font-semibold text-gray-700 mb-4">得分分布</h4>
      <div className="flex items-end gap-2 h-[100px]">
        {results.map((r, i) => {
          const pct = (r.score / maxScore) * 100
          const color = r.score > 0.8 ? '#059669' : r.score > 0.5 ? '#d97706' : '#dc2626'
          return (
            <div key={i} className="flex flex-col items-center gap-1 flex-1">
              <span className="text-[10px] font-semibold" style={{ color }}>{r.score.toFixed(2)}</span>
              <div
                className="w-full rounded-t-md transition-all"
                style={{ height: `${pct}%`, backgroundColor: color }}
              />
              <span className="text-[10px] text-gray-500">D{r.sequence ? '1' : i + 1}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function MPNNLegend() {
  const legendItems = [
    { label: '疏水 (AIVLMPW)', color: '#4ade80' },
    { label: '极性 (STNQ)', color: '#60a5fa' },
    { label: '正电 (KRH)', color: '#f87171' },
    { label: '负电 (DE)', color: '#f87171' },
    { label: '特殊 (CG)', color: '#facc15' },
    { label: '芳香 (FY)', color: '#c084fc' },
  ]
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5">
      <h4 className="text-sm font-semibold text-gray-700 mb-3">氨基酸类型图例</h4>
      <div className="flex flex-wrap gap-2">
        {legendItems.map(({ label, color }) => (
          <span key={label} className="flex items-center gap-1.5 text-xs text-gray-600 bg-gray-50 px-2 py-1 rounded-md">
            <span className="w-2 h-2 rounded-sm" style={{ background: color }} />{label}
          </span>
        ))}
      </div>
    </div>
  )
}
