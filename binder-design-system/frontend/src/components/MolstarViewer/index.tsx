import { useEffect, useRef } from 'react'
import { useAppStore } from '@/store/useAppStore'
import { PluginSpec } from 'molstar/lib/mol-plugin/specs'

declare global {
  interface Window {
    MolstarPlugin: any
    Viewer: any
  }
}

export function MolstarViewer() {
  const containerRef = useRef<HTMLDivElement>(null)
  const pluginRef = useRef<any>(null)
  const pdbContent = useAppStore((s) => s.pdbContent)

  useEffect(() => {
    if (!containerRef.current || !pdbContent) return

    let plugin: any
    const initMolstar = async () => {
      if (typeof window.MolstarPlugin === 'undefined') {
        await import('@molstar/molstar/build/viewer/molstar-plugin')
      }

      const { PluginSpec: PS } = await import('molstar/lib/mol-plugin/specs')
      const { createPlugin } = await import('molstar/lib/mol-plugin/plugin')

      plugin = createPlugin(containerRef.current!, new PS({
        layout: { initial: { isExpanded: false } },
        layoutControls: { display: true }
      }))

      await plugin.loadStructureFromData(pdbContent, 'pdb', { representation: { type: 'cartoon' as const } })

      pluginRef.current = plugin
    }

    initMolstar()

    return () => {
      if (pluginRef.current) {
        try { pluginRef.current.dispose?.() } catch {}
        pluginRef.current = null
      }
    }
  }, [pdbContent])

  return (
    <div className="bg-white rounded-xl border border-gray-200 flex flex-col" style={{ minHeight: '420px' }}>
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-700">3D 结构</h3>
        <div className="flex gap-1.5">
          <button title="重置视角" className="p-1.5 hover:bg-gray-100 rounded text-gray-500 transition-colors">
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
      <div ref={containerRef} className="flex-1 molstar-container bg-[#fafbfc]" style={{ minHeight: '340px' }} />
      {!pdbContent && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-sm pointer-events-none">
          请上传目标结构文件查看3D视图
        </div>
      )}
    </div>
  )
}
