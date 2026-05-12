import { useState } from 'react'
import { useAppStore } from '@/store/useAppStore'
import { authLogin, authRegister } from '@/api'
import { X, Eye, EyeOff } from 'lucide-react'

const formatBackendError = (err: any): string => {
  const data = err?.response?.data
  if (!data) return '操作失败'

  const message = typeof data.message === 'string'
    ? data.message
    : typeof data.detail === 'string'
      ? data.detail
      : '操作失败'

  const details = Array.isArray(data.details)
    ? data.details
        .map((item: any) => {
          if (typeof item === 'string') return item
          if (typeof item?.msg === 'string') return item.msg
          if (typeof item?.message === 'string') return item.message
          return null
        })
        .filter((item: string | null): item is string => Boolean(item))
    : []

  return details.length > 0 ? `${message}：${details.join('；')}` : message
}

export function LoginModal() {
  const setAuth = useAppStore((s) => s.setAuth)
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [showPwd, setShowPwd] = useState(false)
  const [rememberMe, setRememberMe] = useState(false)

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
      setError(formatBackendError(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div className="relative bg-white rounded-2xl shadow-2xl w-[420px] p-0 overflow-hidden animate-in fade-in zoom-in duration-200">
        <div className="px-10 pt-8 pb-6">
          <button
            onClick={() => {}}
            className="absolute right-5 top-4 text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X size={20} />
          </button>

          <h2 className="text-xl font-bold text-center text-gray-900 mb-6">
            {mode === 'login' ? '登录 DeepBinder' : '注册 DeepBinder'}
          </h2>

          {error && (
            <div className="mb-4 px-3 py-2.5 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm flex items-start gap-2">
              <span className="mt-0.5">•</span>
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === 'register' && (
            <div>
              <label htmlFor="auth-email" className="block text-sm font-medium text-gray-700 mb-1.5">
                邮箱 <span className="text-red-500">*</span>
              </label>
              <input
                id="auth-email"
                type="email"
                value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-3.5 py-2.5 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
                  placeholder="Your@example.com"
                  required={mode === 'register'}
                />
              </div>
            )}

            <div>
              <label htmlFor="auth-username" className="block text-sm font-medium text-gray-700 mb-1.5">
                用户名 <span className="text-red-500">*</span>
              </label>
              <input
                id="auth-username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-3.5 py-2.5 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
                placeholder="请输入用户名"
                required
              />
            </div>

            <div>
              <label htmlFor="auth-password" className="block text-sm font-medium text-gray-700 mb-1.5">
                密码 <span className="text-red-500">*</span>
              </label>
              <div className="relative">
                <input
                  id="auth-password"
                  type={showPwd ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full pl-3.5 pr-10 py-2.5 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
                  placeholder="Password"
                  required
                  minLength={6}
                />
                <button
                  type="button"
                  onClick={() => setShowPwd(!showPwd)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPwd ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {mode === 'login' && (
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-1.5 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  <span className="text-sm text-gray-600">记住我</span>
                </label>
                <button type="button" className="text-sm text-indigo-600 hover:text-indigo-800 font-medium">
                  忘记密码
                </button>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50 shadow-md mt-2"
            >
              {loading ? '处理中...' : mode === 'login' ? '登 录' : '注 册'}
            </button>

            <div className="text-center pt-1">
              {mode === 'login' ? (
                <p className="text-sm text-gray-500">
                  还没有账号？{' '}
                  <button
                    type="button"
                    onClick={() => { setMode('register'); setError('') }}
                    className="text-indigo-600 hover:text-indigo-800 font-medium"
                  >
                    注册
                  </button>
                </p>
              ) : (
                <p className="text-sm text-gray-500">
                  已有账号？{' '}
                  <button
                    type="button"
                    onClick={() => { setMode('login'); setError('') }}
                    className="text-indigo-600 hover:text-indigo-800 font-medium"
                  >
                    登录
                  </button>
                </p>
              )}
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
