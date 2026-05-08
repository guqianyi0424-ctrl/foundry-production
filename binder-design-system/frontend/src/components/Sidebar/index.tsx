import { useAppStore } from '@/store/useAppStore'
import { FlaskConical, ClipboardList, HelpCircle, Beaker, Activity, LogOut, User } from 'lucide-react'

const navItems = [
  { label: '新建设计', icon: FlaskConical },
  { label: '实验记录', icon: Beaker },
  { label: '作业中心', icon: ClipboardList },
  { label: '系统监控', icon: Activity },
  { label: '帮助', icon: HelpCircle },
]

export function Sidebar() {
  const currentPage = useAppStore((s) => s.currentPage)
  const setCurrentPage = useAppStore((s) => s.setCurrentPage)
  const user = useAppStore((s) => s.user)
  const clearAuth = useAppStore((s) => s.clearAuth)

  return (
    <aside className="w-[200px] min-w-[200px] bg-white border-r border-gray-200 flex flex-col h-full shadow-sm">
      <div className="flex items-center gap-2 px-5 py-4 border-b border-gray-100">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center text-white font-bold text-sm">
          D
        </div>
        <span className="font-bold text-lg text-gray-900">DeepBinder</span>
      </div>

      <nav className="flex-1 py-3 px-3 space-y-1">
        {navItems.map(({ label, icon: Icon }) => {
          const isActive = currentPage === label || currentPage.startsWith(`experiment_`) && label === '实验记录'
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
        {user ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <User size={14} className="text-gray-400" />
              <span className="text-gray-700 font-medium truncate">{user.username}</span>
              <span className="text-xs px-1.5 py-0.5 bg-gray-100 text-gray-500 rounded">{user.role === 'admin' ? '管理员' : '研究员'}</span>
            </div>
            <button
              onClick={() => { clearAuth(); setCurrentPage('新建设计') }}
              className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
            >
              <LogOut size={12} /> 退出登录
            </button>
          </div>
        ) : (
          <div className="text-xs text-gray-400 text-center">未登录</div>
        )}
        <div className="text-xs text-gray-400 text-center mt-2">DeepBinder v2.0</div>
      </div>
    </aside>
  )
}
