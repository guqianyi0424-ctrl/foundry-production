import { render, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { MolstarViewer } from '@/components/MolstarViewer'
import { useAppStore } from '@/store/useAppStore'

const rawData = vi.fn(async () => ({ kind: 'data' }))
const parseTrajectory = vi.fn(async () => ({ kind: 'trajectory' }))
const applyPreset = vi.fn(async () => undefined)

vi.mock('molstar/lib/mol-plugin-ui', () => ({
  createPluginUI: vi.fn(async () => ({
    builders: {
      data: { rawData },
      structure: {
        parseTrajectory,
        hierarchy: { applyPreset },
      },
    },
    managers: {
      structure: {
        hierarchy: { current: { structures: [] } },
        selection: { clear: vi.fn(), fromLoci: vi.fn() },
      },
      interactivity: {
        lociHighlights: { clearHighlights: vi.fn(), highlight: vi.fn() },
      },
      camera: { focusLoci: vi.fn() },
    },
    canvas3d: { requestCameraReset: vi.fn() },
  })),
}))

vi.mock('molstar/lib/mol-plugin-ui/spec', () => ({
  DefaultPluginUISpec: vi.fn(() => ({ components: {} })),
}))

vi.mock('molstar/lib/mol-plugin-ui/react18', () => ({
  renderReact18: vi.fn(),
}))

describe('MolstarViewer', () => {
  beforeEach(() => {
    useAppStore.getState().resetAll()
    rawData.mockClear()
    parseTrajectory.mockClear()
    applyPreset.mockClear()
  })

  it('loads existing PDB content after the viewer initializes', async () => {
    useAppStore.getState().setPdbContent('TARGET_PDB')

    render(<MolstarViewer />)

    await waitFor(() => {
      expect(rawData).toHaveBeenCalledWith(
        { data: 'TARGET_PDB' },
        { state: { isGhost: true } },
      )
    })
    expect(parseTrajectory).toHaveBeenCalled()
    expect(applyPreset).toHaveBeenCalled()
  })
})
