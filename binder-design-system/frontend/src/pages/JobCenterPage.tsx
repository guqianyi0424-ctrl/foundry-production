import { useAppStore } from '@/store/useAppStore'
import { CheckCircle2, XCircle, Clock } from 'lucide-react'

export function JobCenterPage() {
  const jobHistory = useAppStore((s) => s.jobHistory)

  const statusConfig: Record<string, { icon: any; color: string; bg: string }> = {
    completed: { icon: CheckCircle2, color: '#059669', bg: '#ecfdf5' },
    failed: { icon: XCircle, color: '#dc2626', bg: '#fef2f2' },
    running: { icon: Clock, color: '#d97706', bg: '#fffbeb' },
    pending: { icon: Clock, color: '#94a3b8', bg: '#f8fafc' },
  }

  return (
    <div className="p-6 max-w-[1200px] mx-auto">
      <h1 className="text-xl font-bold text-gray-900 mb-6">作业中心</h1>

      {jobHistory.length === 0 ? (
        <div className="bg-white rounded-xl border border-gray-200 p-16 text-center">
          <div className="text-5xl mb-4">📋</div>
          <p className="text-gray-500 text-sm">暂无历史记录</p>
          <p className="text-gray-400 text-xs mt-2">运行设计任务后，记录将显示在这里</p>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50/50">
                <th className="text-left text-xs font-semibold text-gray-600 px-5 py-3">任务名称</th>
                <th className="text-left text-xs font-semibold text-gray-600 px-5 py-3">状态</th>
                <th className="text-left text-xs font-semibold text-gray-600 px-5 py-3">创建时间</th>
                <th className="text-right text-xs font-semibold text-gray-600 px-5 py-3">操作</th>
              </tr>
            </thead>
            <tbody>
              {jobHistory.map((job) => {
                const config = statusConfig[job.status] ?? statusConfig.pending
                const Icon = config.icon
                return (
                  <tr key={job.id} className="border-b border-gray-50 hover:bg-gray-50/50 transition-colors">
                    <td className="px-5 py-3.5 text-sm font-medium text-gray-900">{job.name}</td>
                    <td className="px-5 py-3.5">
                      <span className="inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full" style={{ color: config.color, backgroundColor: config.bg }}>
                        <Icon size={14} />
                        {job.status === 'completed' ? '完成' : job.status === 'failed' ? '失败' : job.status === 'running' ? '运行中' : '等待中'}
                      </span>
                    </td>
                    <td className="px-5 py-3.5 text-sm text-gray-500">{job.time}</td>
                    <td className="px-5 py-3.5 text-right">
                      <button className="text-xs text-primary-600 hover:text-primary-800 font-medium mr-3">查看</button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
