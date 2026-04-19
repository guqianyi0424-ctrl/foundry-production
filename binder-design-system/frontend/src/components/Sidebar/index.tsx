import { useAppStore } from '@/store/useAppStore'
import { FlaskConical, ClipboardList, HelpCircle } from 'lucide-react'

const navItems = [
  { label: '新建设计', icon: FlaskConical },
  { label: '作业中心', icon: ClipboardList },
  { label: '帮助', icon: HelpCircle },
]

export function Sidebar() {
  const currentPage = useAppStore((s) => s.currentPage)
  const setCurrentPage = useAppStore((s) => s.setCurrentPage)

  return (
    <aside className="w-[200px] min-w-[200px] bg-white border-r border-gray-200 flex flex-col h-full shadow-sm">
      <div className="flex items-center gap-2 px-5 py-4 border-b border-gray-100">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center text-white font-bold text-sm">
          O
        </div>
        <span className="font-bold text-lg text-gray-900">ODesign</span>
      </div>

      <nav className="flex-1 py-3 px-3 space-y-1">
        {navItems.map(({ label, icon: Icon }) => {
          const isActive = currentPage === label
          return (
            <button
              key={label}
              onClick={() => setCurrentPage(label)}
              className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? 'bg-primary-50 text-primary-700 shadow-sm'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
              }`}
            >
              <Icon size={18} />
              <span>{label}</span>
            </button>
          )
        })}
      </nav>

      <div className="p-4 border-t border-gray-100">
        <div className="text-xs text-gray-400 text-center">ODesign v1.0</div>
      </div>
    </aside>
  )
}
