import { useEffect, useState } from 'react'
import { getUsers, type UserSummary } from '@/api'
import { useAppStore } from '@/store/useAppStore'
import { RefreshCw, Users, Shield, User } from 'lucide-react'

const formatDate = (dateStr?: string | null) => {
  if (!dateStr) return '-'
  try {
    return new Date(dateStr).toLocaleString('zh-CN')
  } catch {
    return dateStr
  }
}

export function UserManagementPage() {
  const setCurrentPage = useAppStore((s) => s.setCurrentPage)
  const [users, setUsers] = useState<UserSummary[]>([])
  const [loading, setLoading] = useState(true)

  const fetchUsers = async () => {
    setLoading(true)
    try {
      const res = await getUsers()
      setUsers(res.users || [])
    } catch (err) {
      console.error('获取用户列表失败:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchUsers()
  }, [])

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">用户管理</h1>
          <p className="text-sm text-gray-500 mt-1">按用户查看账号信息和实验记录</p>
        </div>
        <button
          onClick={fetchUsers}
          className="flex items-center gap-1.5 px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
        >
          <RefreshCw size={14} /> 刷新
        </button>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-100 flex items-center gap-2">
          <Users size={18} className="text-primary-600" />
          <span className="font-medium text-gray-900">用户列表</span>
          <span className="text-sm text-gray-400">共 {users.length} 个账号</span>
        </div>

        {loading ? (
          <div className="text-center py-16 text-gray-400">
            <RefreshCw size={28} className="animate-spin mx-auto mb-3" />
            加载中...
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="text-left px-4 py-3 font-medium text-gray-600">用户名</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">邮箱</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">角色</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">创建时间</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">最近登录</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">操作</th>
              </tr>
            </thead>
            <tbody>
              {users.map((item) => {
                const isAdmin = item.role === 'admin'
                const RoleIcon = isAdmin ? Shield : User
                return (
                  <tr key={item.id} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-3 font-medium text-gray-900">{item.username}</td>
                    <td className="px-4 py-3 text-gray-500">{item.email || '-'}</td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${isAdmin ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700'}`}>
                        <RoleIcon size={12} />
                        {isAdmin ? '管理员' : '研究员'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-500 text-xs">{formatDate(item.created_at)}</td>
                    <td className="px-4 py-3 text-gray-500 text-xs">{formatDate(item.last_login)}</td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => setCurrentPage(`user_${item.id}`)}
                        aria-label={`查看 ${item.username} 实验`}
                        className="px-2 py-1 text-xs text-primary-600 hover:bg-primary-50 rounded transition-colors"
                      >
                        查看实验
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
