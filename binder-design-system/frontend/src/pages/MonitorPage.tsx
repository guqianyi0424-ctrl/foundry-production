import { useState, useEffect } from 'react'
import { getMonitorStatus, getMonitorTasks } from '@/api'
import { Cpu, Server, HardDrive, Activity, Clock, CheckCircle, XCircle } from 'lucide-react'

const formatGpuMemory = (gpu: any) => {
  if (!gpu) return ''
  if (gpu.memory && gpu.memory !== 'N/A') return gpu.memory
  if (typeof gpu.memory_used_mib === 'number' && typeof gpu.memory_total_mib === 'number') {
    return `${gpu.memory_used_mib} / ${gpu.memory_total_mib} MiB`
  }
  return ''
}

export function MonitorPage() {
  const [status, setStatus] = useState<any>(null)
  const [tasks, setTasks] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetch = async () => {
      try {
        const [s, t] = await Promise.all([getMonitorStatus(), getMonitorTasks()])
        setStatus(s)
        setTasks(t.tasks || [])
      } catch (err) {
        console.error('获取监控数据失败:', err)
      } finally {
        setLoading(false)
      }
    }
    fetch()
    const interval = setInterval(fetch, 10000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return <div className="flex items-center justify-center h-full text-gray-400"><Activity className="animate-spin mr-2" /> 加载中...</div>
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">系统监控</h1>

      {status && (
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <div className="flex items-center gap-2 mb-2">
              <Cpu size={18} className="text-blue-500" />
              <span className="text-sm text-gray-500">CPU</span>
            </div>
            <div className="text-2xl font-bold">{status.cpu_percent}%</div>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <div className="flex items-center gap-2 mb-2">
              <Server size={18} className="text-green-500" />
              <span className="text-sm text-gray-500">内存</span>
            </div>
            <div className="text-2xl font-bold">{status.memory?.percent}%</div>
            <div className="text-xs text-gray-400 mt-1">{status.memory?.used_gb} / {status.memory?.total_gb} GB</div>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <div className="flex items-center gap-2 mb-2">
              <HardDrive size={18} className="text-orange-500" />
              <span className="text-sm text-gray-500">磁盘</span>
            </div>
            <div className="text-2xl font-bold">{status.disk?.percent}%</div>
            <div className="text-xs text-gray-400 mt-1">{status.disk?.used_gb} / {status.disk?.total_gb} GB</div>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <div className="flex items-center gap-2 mb-2">
              <Activity size={18} className="text-purple-500" />
              <span className="text-sm text-gray-500">GPU</span>
            </div>
            <div className="text-lg font-bold">{status.gpu?.name || 'N/A'}</div>
            <div className="text-xs text-gray-400 mt-1">{formatGpuMemory(status.gpu)}</div>
            {typeof status.gpu?.utilization_percent === 'number' && (
              <div className="text-xs text-gray-400 mt-0.5">利用率 {status.gpu.utilization_percent}%</div>
            )}
          </div>
        </div>
      )}

      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <h3 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
          <Clock size={16} /> 最近任务
        </h3>
        {tasks.length === 0 ? (
          <p className="text-gray-400 text-sm">暂无任务记录</p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-gray-500">
                <th className="text-left py-2">任务</th>
                <th className="text-left py-2">状态</th>
                <th className="text-left py-2">耗时</th>
                <th className="text-left py-2">时间</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((t, i) => (
                <tr key={i} className="border-b border-gray-100">
                  <td className="py-2 font-mono text-xs">{t.name}</td>
                  <td className="py-2">
                    {t.status === 'success' ? (
                      <span className="inline-flex items-center gap-1 text-green-600"><CheckCircle size={12} /> 成功</span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-red-500"><XCircle size={12} /> 失败</span>
                    )}
                  </td>
                  <td className="py-2 text-gray-600">{t.duration}</td>
                  <td className="py-2 text-gray-400 text-xs">{t.timestamp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
