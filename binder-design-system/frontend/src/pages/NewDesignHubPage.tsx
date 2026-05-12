import { useState } from 'react'
import { Dna, FlaskConical } from 'lucide-react'
import { DeNovoDesignPage } from '@/pages/DeNovoDesignPage'
import { NewDesignPage } from '@/pages/NewDesignPage'

type DesignMode = 'protein-to-protein' | 'de-novo'

const modes: Array<{
  key: DesignMode
  label: string
  icon: typeof FlaskConical
}> = [
  { key: 'de-novo', label: 'de novo protein', icon: Dna },
  { key: 'protein-to-protein', label: 'protein to protein', icon: FlaskConical },
]

export function NewDesignHubPage() {
  const [mode, setMode] = useState<DesignMode>('protein-to-protein')

  return (
    <div>
      <div className="sticky top-0 z-20 border-b border-gray-200 bg-deepbinder-bg/95 px-6 py-4 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-4">
          <h1 className="text-xl font-bold text-gray-900">新建设计</h1>
          <div className="inline-flex rounded-lg border border-gray-200 bg-white p-1 shadow-sm">
            {modes.map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setMode(key)}
                className={`inline-flex min-w-40 items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                  mode === key
                    ? 'bg-primary-600 text-white shadow-sm'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`}
              >
                <Icon size={16} />
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
      {mode === 'de-novo' ? <DeNovoDesignPage /> : <NewDesignPage />}
    </div>
  )
}
