import { useEffect, useRef, useState } from 'react'
import { useAppStore } from '@/store/useAppStore'
import 'molstar/build/viewer/molstar.css'

export function MolstarViewer() {
  const containerRef = useRef<HTMLDivElement>(null)
  const pdbContent = useAppStore((s) => s.pdbContent)
  const focusedResidue = useAppStore((s) => s.focusedResidue)
  const hoveredResidue = useAppStore((s) => s.hoveredResidue)
  const selectedRange = useAppStore((s) => s.selectedRange)
  const selectedHotspots = useAppStore((s) => s.selectedHotspots)
  const predictedHotspots = useAppStore((s) => s.predictedHotspots)
  const [initDone, setInitDone] = useState(false)
  const [structureVersion, setStructureVersion] = useState(0)
  const pluginRef = useRef<any>(null)

  useEffect(() => {
    if (!containerRef.current || initDone) return
    let destroyed = false

    const initViewer = async () => {
      try {
        const { createPluginUI } = await import('molstar/lib/mol-plugin-ui')
        const { DefaultPluginUISpec } = await import('molstar/lib/mol-plugin-ui/spec')
        const { renderReact18 } = await import('molstar/lib/mol-plugin-ui/react18')

        if (destroyed || !containerRef.current) return

        const spec = DefaultPluginUISpec()
        spec.layout = {
          initial: {
            isExpanded: false,
            showControls: false,
            regionState: {
              left: 'hidden' as any, top: 'hidden' as any,
              right: 'hidden' as any, bottom: 'hidden' as any,
            },
          },
        }
        spec.components = {
          ...spec.components,
          controls: {
            top: 'none' as any, left: 'none' as any,
            right: 'none' as any, bottom: 'none' as any,
          },
        }

        const plugin = await createPluginUI({
          target: containerRef.current,
          render: renderReact18,
          spec,
        })

        if (destroyed) return
        pluginRef.current = plugin
        setInitDone(true)
      } catch (err) {
        console.error('Molstar init error:', err)
      }
    }
    initViewer()
    return () => { destroyed = true }
  }, [initDone])

  useEffect(() => {
    const plugin = pluginRef.current
    if (!plugin || !pdbContent) return
    let cancelled = false
    const loadStructure = async () => {
      try {
        const data = await plugin.builders.data.rawData({ data: pdbContent }, { state: { isGhost: true } })
        if (cancelled) return
        const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb')
        if (cancelled) return
        await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default')
        if (!cancelled) setStructureVersion(version => version + 1)
      } catch (err) {
        console.warn('Load structure failed:', err)
      }
    }
    loadStructure()
    return () => { cancelled = true }
  }, [pdbContent])

  const buildResiduesLoci = async (targets: Array<{ chain: string; resSeq: number }>): Promise<any> => {
    const plugin = pluginRef.current
    if (!plugin || !targets.length) return null

    const structures = plugin.managers.structure.hierarchy.current.structures
    if (!structures.length) return null
    const structure = structures[0].cell.obj.data
    if (!structure) return null

    try {
      const SE = (await import('molstar/lib/mol-model/structure')).StructureElement
      const SP = (await import('molstar/lib/mol-model/structure')).StructureProperties

      const loc = SE.Location.create(structure)
      const elements: any[] = []

      for (const unit of structure.units) {
        if (!unit.model.atomicHierarchy) continue
        const unitElements = unit.elements
        const elementCount = unitElements.length ?? 0
        if (elementCount === 0) continue

        loc.unit = unit
        const matchingIndices: number[] = []

        for (let i = 0; i < elementCount; i++) {
          loc.element = unitElements[i]
          const asymId = SP.chain.auth_asym_id(loc)
          const seqId = SP.residue.auth_seq_id(loc)

          for (const t of targets) {
            if (asymId === t.chain && seqId === t.resSeq) {
              matchingIndices.push(i)
              break
            }
          }
        }

        if (matchingIndices.length > 0) {
          matchingIndices.sort((a, b) => a - b)
          elements.push({
            unit,
            indices: new Int32Array(matchingIndices),
          })
        }
      }

      if (elements.length === 0) {
        console.warn('[buildResiduesLoci] no matching elements for', targets)
        return null
      }

      const loci = SE.Loci(structure, elements)
      console.log('[buildResiduesLoci] success! size:', SE.Loci.size(loci))
      return loci
    } catch (e) {
      console.error('[buildResiduesLoci] error:', e)
      return null
    }
  }

  const buildRangeLoci = async (chainId: string, startResSeq: number, endResSeq: number): Promise<any> => {
    const plugin = pluginRef.current
    if (!plugin) return null

    const structures = plugin.managers.structure.hierarchy.current.structures
    if (!structures.length) return null
    const structure = structures[0].cell.obj.data
    if (!structure) return null

    try {
      const SE = (await import('molstar/lib/mol-model/structure')).StructureElement
      const SP = (await import('molstar/lib/mol-model/structure')).StructureProperties

      const loc = SE.Location.create(structure)
      const elements: any[] = []

      for (const unit of structure.units) {
        if (!unit.model.atomicHierarchy) continue
        const unitElements = unit.elements
        const elementCount = unitElements.length ?? 0
        if (elementCount === 0) continue

        loc.unit = unit
        const matchingIndices: number[] = []

        for (let i = 0; i < elementCount; i++) {
          loc.element = unitElements[i]
          const asymId = SP.chain.auth_asym_id(loc)
          const seqId = SP.residue.auth_seq_id(loc)

          if (asymId === chainId && seqId >= startResSeq && seqId <= endResSeq) {
            matchingIndices.push(i)
          }
        }

        if (matchingIndices.length > 0) {
          matchingIndices.sort((a, b) => a - b)
          elements.push({
            unit,
            indices: new Int32Array(matchingIndices),
          })
        }
      }

      if (elements.length === 0) {
        console.warn('[buildRangeLoci] no matching elements for', chainId, startResSeq, '-', endResSeq)
        return null
      }

      const loci = SE.Loci(structure, elements)
      console.log('[buildRangeLoci] success! size:', SE.Loci.size(loci))
      return loci
    } catch (e) {
      console.error('[buildRangeLoci] error:', e)
      return null
    }
  }

  useEffect(() => {
    const plugin = pluginRef.current
    if (!plugin || !focusedResidue) return
    const focusResidue = async () => {
      try {
        const loci = await buildResiduesLoci([{ chain: focusedResidue.chain, resSeq: focusedResidue.resSeq }])
        if (!loci) return
        plugin.managers.camera.focusLoci(loci, { durationMs: 500 })
      } catch (err) {
        console.warn('Focus residue failed:', err)
      }
    }
    focusResidue()
  }, [focusedResidue])

  useEffect(() => {
    const plugin = pluginRef.current
    if (!plugin) return

    const updateHighlights = async () => {
      try {
        plugin.managers.interactivity.lociHighlights.clearHighlights()
        plugin.managers.structure.selection.clear()

        let combinedSelectionLoci: any = null

        const highlightLoci = (loci: any, color?: number) => {
          if (!loci) return
          ;(plugin.managers.interactivity.lociHighlights as any).highlight({ loci, color })
        }

        const addPersistentSelection = async (loci: any) => {
          if (!loci) return
          combinedSelectionLoci = combinedSelectionLoci
            ? (await import('molstar/lib/mol-model/structure')).StructureElement.Loci.union(combinedSelectionLoci, loci)
            : loci
          plugin.managers.structure.selection.fromLoci('add', loci, false)
        }

        if (selectedRange) {
          const rangeLoci = await buildRangeLoci(
            selectedRange.chain,
            selectedRange.startResSeq,
            selectedRange.endResSeq
          )
          highlightLoci(rangeLoci, 0xF59E0B)
          await addPersistentSelection(rangeLoci)
        }

        const visibleHotspots = [
          ...predictedHotspots,
          ...selectedHotspots.filter(selected => !predictedHotspots.some(predicted =>
            predicted.chain === selected.chain && predicted.residue === selected.residue
          )),
        ]

        if (visibleHotspots.length > 0) {
          const hotspotLoci = await buildResiduesLoci(
            visibleHotspots.map(h => ({ chain: h.chain, resSeq: h.residue }))
          )
          highlightLoci(hotspotLoci, 0xEF4444)
          await addPersistentSelection(hotspotLoci)
        }

        if (hoveredResidue) {
          const hoverLoci = await buildResiduesLoci([{ chain: hoveredResidue.chain, resSeq: hoveredResidue.resSeq }])
          highlightLoci(hoverLoci, 0x60A5FA)
        }
      } catch (err) {
        console.warn('Update highlights failed:', err)
      }
    }

    updateHighlights()
  }, [hoveredResidue, focusedResidue, selectedHotspots, predictedHotspots, selectedRange, structureVersion])

  const handleResetCamera = () => {
    const plugin = pluginRef.current
    if (plugin?.canvas3d) plugin.canvas3d.requestCameraReset()
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 flex flex-col" style={{ minHeight: '420px', maxHeight: '420px' }}>
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100 shrink-0">
        <h3 className="text-sm font-semibold text-gray-700">3D 结构</h3>
        <div className="flex gap-1.5">
          <button onClick={handleResetCamera} title="重置视角" className="p-1.5 hover:bg-gray-100 rounded text-gray-500 transition-colors">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>
          </button>
          <button title="旋转" className="p-1.5 hover:bg-gray-100 rounded text-gray-500 transition-colors">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21.5 2v6h-6M2.5 22v-6h6"/><path d="M11 19a8 8 0 010-16 7.99 7.99 0 015.29 2l5.21 3M18.71 13.71A8 8 0 0113 20l-6-4.21"/></svg>
          </button>
          <button title="适应窗口" className="p-1.5 hover:bg-gray-100 rounded text-gray-500 transition-colors">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6v6H9z"/></svg>
          </button>
          <button title="截图" className="p-1.5 hover:bg-gray-100 rounded text-gray-500 transition-colors">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>
          </button>
        </div>
      </div>
      <div ref={containerRef} style={{ flex: 1, minHeight: '340px', position: 'relative', background: '#fafbfc' }} />
      {!initDone && (
        <div style={{ position: 'absolute', top: '44px', left: 0, right: 0, bottom: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#9ca3af', fontSize: '14px', background: '#fff', zIndex: 10, pointerEvents: 'none' }}>
          正在加载3D引擎...
        </div>
      )}
    </div>
  )
}
