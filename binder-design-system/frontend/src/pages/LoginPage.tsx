import { useState, useEffect } from 'react'
import { useAppStore } from '@/store/useAppStore'
import { authLogin, authRegister } from '@/api'
import { LogIn, UserPlus, Lock, Mail, User, AlertCircle } from 'lucide-react'

export function LoginPage() {
  const setAuth = useAppStore((s) => s.setAuth)
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      if (mode === 'login') {
        const res = await authLogin(username, password)
        setAuth(res.access_token, res.user)
      } else {
        await authRegister(username, password, email || undefined)
        const res = await authLogin(username, password)
        setAuth(res.access_token, res.user)
      }
    } catch (err: any) {
      const detail = err.response?.data?.detail || '操作失败'
      setError(typeof detail === 'string' ? detail : JSON.stringify(detail))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 text-white font-bold text-2xl mb-4 shadow-lg">
            O
          </div>
          <h1 className="text-2xl font-bold text-gray-900">ODesign</h1>
          <p className="text-gray-500 mt-1">蛋白质Binder设计系统</p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="flex gap-2 mb-6 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => { setMode('login'); setError('') }}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                mode === 'login' ? 'bg-white shadow text-primary-700' : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              <LogIn size={14} className="inline mr-1" />登录
            </button>
            <button
              onClick={() => { setMode('register'); setError('') }}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                mode === 'register' ? 'bg-white shadow text-primary-700' : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              <UserPlus size={14} className="inline mr-1" />注册
            </button>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700 text-sm">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">用户名</label>
              <div className="relative">
                <User size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
                  placeholder="请输入用户名"
                  required
                />
              </div>
            </div>

            {mode === 'register' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">邮箱 (可选)</label>
                <div className="relative">
                  <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
                    placeholder="请输入邮箱"
                  />
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">密码</label>
              <div className="relative">
                <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
                  placeholder="请输入密码"
                  required
                  minLength={6}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 bg-gradient-to-r from-primary-500 to-primary-700 text-white rounded-lg font-medium text-sm hover:from-primary-600 hover:to-primary-800 transition-all disabled:opacity-50 shadow-md"
            >
              {loading ? '处理中...' : mode === 'login' ? '登录' : '注册'}
            </button>
          </form>

          {mode === 'login' && (
            <div className="mt-4 p-3 bg-blue-50 rounded-lg text-xs text-blue-700">
              <p className="font-medium mb-1">默认管理员账号：</p>
              <p>用户名: admin / 密码: admin123</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
