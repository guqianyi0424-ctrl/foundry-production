import { useCallback, useRef, useState } from 'react'
import { useAppStore } from '@/store/useAppStore'
import { Link2, RotateCcw, Upload } from 'lucide-react'

export function SequenceViewer() {
  const chains = useAppStore((s) => s.chains)
  const selectedHotspots = useAppStore((s) => s.selectedHotspots)
  const selectedRange = useAppStore((s) => s.selectedRange)
  const setSelectedRange = useAppStore((s) => s.setSelectedRange)
  const toggleHotspot = useAppStore((s) => s.toggleHotspot)
  const setFocusedResidue = useAppStore((s) => s.setFocusedResidue)
  const setHoveredResidue = useAppStore((s) => s.setHoveredResidue)

  const [dragState, setDragState] = useState<{
    chainId: string | null;
    startIdx: number;
    currentIdx: number;
    dragging: boolean;
  }>({ chainId: null, startIdx: -1, currentIdx: -1, dragging: false })

  const dragStartRef = useRef<{ chainId: string; idx: number; resSeq: number; moved: boolean } | null>(null)

  const isHotspot = useCallback((chainId: string, resSeq: number) =>
    selectedHotspots.some(h => h.chain === chainId && h.residue === resSeq), [selectedHotspots])

  const isInSelectedRange = useCallback((chainId: string, resSeq: number): boolean => {
    if (!selectedRange || selectedRange.chain !== chainId) return false
    return resSeq >= selectedRange.startResSeq && resSeq <= selectedRange.endResSeq
  }, [selectedRange])

  const isInDragRange = useCallback((chainId: string, idx: number): boolean => {
    if (!dragState.dragging || dragState.chainId !== chainId) return false
    const start = Math.min(dragState.startIdx, dragState.currentIdx)
    const end = Math.max(dragState.startIdx, dragState.currentIdx)
    return idx >= start && idx <= end
  }, [dragState])

  const handleMouseDown = (chainId: string, idx: number, resSeq: number) => {
    dragStartRef.current = { chainId, idx, resSeq, moved: false }
    setDragState({ chainId, startIdx: idx, currentIdx: idx, dragging: true })
  }

  const handleMouseMove = (chainId: string, idx: number) => {
    if (!dragState.dragging || !dragStartRef.current || dragState.chainId !== chainId) return
    dragStartRef.current.moved = true
    setDragState(prev => ({ ...prev, currentIdx: idx }))
  }

  const handleMouseEnter = (chainId: string, resSeq: number, idx: number) => {
    setHoveredResidue({ chain: chainId, resSeq })
    if (dragState.dragging && dragState.chainId === chainId) {
      setDragState(prev => ({ ...prev, currentIdx: idx }))
    }
  }

  const handleMouseUp = () => {
    if (!dragState.dragging || !dragStartRef.current) {
      setDragState({ chainId: null, startIdx: -1, currentIdx: -1, dragging: false })
      dragStartRef.current = null
      return
    }
    const { chainId, idx: startIdx, resSeq: startResSeq, moved } = dragStartRef.current
    const endIdx = dragState.currentIdx

    if (!moved) {
      const chain = chains.find(c => c.chain_id === chainId)
      if (chain) {
        const r = startResSeq
        if (isInSelectedRange(chainId, r)) {
          toggleHotspot(chainId, r)
          setFocusedResidue({ chain: chainId, resSeq: r })
        }
      }
    } else {
      const chain = chains.find(c => c.chain_id === chainId)
      if (chain) {
        const minIdx = Math.min(startIdx, endIdx)
        const maxIdx = Math.max(startIdx, endIdx)
        const startR = chain.resSeqs?.[minIdx] ?? (minIdx + 1)
        const endR = chain.resSeqs?.[maxIdx] ?? (maxIdx + 1)
        setSelectedRange({ chain: chainId, startResSeq: startR, endResSeq: endR })
      }
    }

    setDragState({ chainId: null, startIdx: -1, currentIdx: -1, dragging: false })
    dragStartRef.current = null
  }

  const handleMouseLeave = () => {
    setHoveredResidue(null)
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 flex flex-col" style={{ minHeight: '420px', maxHeight: '420px' }}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => { handleMouseUp(); handleMouseLeave() }}>
      <div className="flex items-center justify-between mb-3 shrink-0">
        <h3 className="text-sm font-semibold text-gray-700">序列视图</h3>
        <div className="flex gap-1.5">
          <button title="裁剪热点" className="p-1.5 hover:bg-gray-100 rounded text-gray-500 transition-colors"><Link2 size={14} /></button>
          <button title="指定热点" className="p-1.5 hover:bg-gray-100 rounded text-gray-500 transition-colors"><Upload size={14} /></button>
          <button title="重置" onClick={() => { useAppStore.getState().setSelectedRange(null); useAppStore.getState().setSelectedHotspots([]) }} className="p-1.5 hover:bg-gray-100 rounded text-gray-500 transition-colors"><RotateCcw size={14} /></button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto overflow-x-hidden border border-gray-100 rounded-lg p-3 bg-white space-y-4">
        {chains.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            请上传目标结构文件
          </div>
        ) : (
          chains.map((chain) => (
            <div key={chain.chain_id}>
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-xs font-bold text-primary-600 bg-primary-50 px-1.5 py-0.5 rounded">{chain.chain_id}</span>
                <span className="text-xs text-gray-400">{chain.length} residues</span>
                {selectedRange && selectedRange.chain === chain.chain_id && (
                  <span className="text-xs font-medium text-green-600 bg-green-50 px-2 py-0.5 rounded">
                    已选: {selectedRange.startResSeq}-{selectedRange.endResSeq}
                  </span>
                )}
              </div>
              <div
                className="leading-relaxed font-mono text-base tracking-wider flex flex-wrap gap-x-[2px]"
                onMouseMove={(e) => {
                  const target = e.target as HTMLElement
                  if (target.dataset.idx !== undefined) {
                    handleMouseMove(chain.chain_id, parseInt(target.dataset.idx))
                  }
                }}
              >
                {[...chain.sequence].map((aa, i) => {
                  const resSeq = chain.resSeqs?.[i] ?? (i + 1)
                  const inRange = isInSelectedRange(chain.chain_id, resSeq)
                  const hot = isHotspot(chain.chain_id, resSeq)
                  const inDrag = isInDragRange(chain.chain_id, i)
                  const residueClassName = [
                    'seq-residue relative inline-flex h-7 w-[1.15rem] items-center justify-center rounded px-0.5 pb-2 pt-1 select-none cursor-pointer transition-colors',
                    inDrag ? 'bg-amber-100 text-amber-800 ring-1 ring-amber-300' : '',
                    inRange ? 'bg-green-50 text-green-700 hover:bg-green-100' : 'text-gray-400 hover:bg-gray-50',
                    hot ? 'font-semibold text-red-700' : '',
                  ].filter(Boolean).join(' ')
                  return (
                    <span
                      key={`${chain.chain_id}-${resSeq}`}
                      data-idx={i}
                      onMouseDown={(e) => { e.preventDefault(); handleMouseDown(chain.chain_id, i, resSeq) }}
                      onMouseEnter={() => handleMouseEnter(chain.chain_id, resSeq, i)}
                      onMouseLeave={handleMouseLeave}
                      className={residueClassName}
                      title={
                        inRange
                          ? `${chain.chain_id}/${resSeq}${hot ? ' ✓ 热点' : ''}`
                          : `${chain.chain_id}/${resSeq} (先拖拽选择范围)`
                        }
                    >
                      <span className="seq-residue-letter">{aa}</span>
                      {hot && <span className="seq-hotspot-dot absolute bottom-0.5 left-1/2 h-1.5 w-1.5 -translate-x-1/2 rounded-full bg-red-500 shadow-[0_0_0_2px_rgba(255,255,255,0.95)]" aria-hidden="true" />}
                    </span>
                  )
                })}
              </div>
            </div>
          ))
        )}
      </div>

      <div className="mt-2 text-xs text-gray-400 shrink-0">
        💡 拖拽选择残基范围 → 单击范围内残基选择热点 → 点击上方按钮确认
      </div>
    </div>
  )
}
