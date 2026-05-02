import { useState, useEffect, useCallback } from 'react'
import { useAppStore } from '@/store/useAppStore'
import { getExperiments, deleteExperiment, exportExperiment, type ExperimentItem } from '@/api'
import { FlaskConical, Trash2, Download, Search, RefreshCw, Clock, Cpu, CheckCircle, XCircle, Loader, FileText } from 'lucide-react'

const statusConfig: Record<string, { label: string; color: string; icon: any }> = {
  created: { label: '已创建', color: 'bg-gray-100 text-gray-700', icon: FileText },
  running: { label: '运行中', color: 'bg-blue-100 text-blue-700', icon: Loader },
  rfd3_completed: { label: 'RFD3完成', color: 'bg-yellow-100 text-yellow-700', icon: FlaskConical },
  mpnn_completed: { label: 'MPNN完成', color: 'bg-orange-100 text-orange-700', icon: FlaskConical },
  completed: { label: '已完成', color: 'bg-green-100 text-green-700', icon: CheckCircle },
  failed: { label: '失败', color: 'bg-red-100 text-red-700', icon: XCircle },
}

export function ExperimentsPage() {
  const setCurrentPage = useAppStore((s) => s.setCurrentPage)
  const [experiments, setExperiments] = useState<ExperimentItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [keyword, setKeyword] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

  const fetchExperiments = useCallback(async () => {
    setLoading(true)
    try {
      const res = await getExperiments({ page, page_size: 20, keyword: keyword || undefined, status: statusFilter || undefined })
      setExperiments(res.items || [])
      setTotal(res.total || 0)
    } catch (err) {
      console.error('获取实验列表失败:', err)
    } finally {
      setLoading(false)
    }
  }, [page, keyword, statusFilter])

  useEffect(() => {
    fetchExperiments()
  }, [fetchExperiments])

  const handleDelete = async (id: string) => {
    if (!confirm('确定删除此实验记录？')) return
    try {
      await deleteExperiment(id)
      fetchExperiments()
    } catch (err) {
      console.error('删除失败:', err)
    }
  }

  const handleExport = async (id: string, name: string) => {
    try {
      const report = await exportExperiment(id)
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${name}_report.json`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('导出失败:', err)
    }
  }

  const toggleSelect = (id: string) => {
    const next = new Set(selectedIds)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setSelectedIds(next)
  }

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '-'
    try {
      return new Date(dateStr).toLocaleString('zh-CN')
    } catch {
      return dateStr
    }
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">实验记录</h1>
          <p className="text-sm text-gray-500 mt-1">管理和查看所有蛋白Binder设计实验</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={fetchExperiments}
            className="flex items-center gap-1.5 px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            <RefreshCw size={14} /> 刷新
          </button>
          <button
            onClick={() => setCurrentPage('新建设计')}
            className="flex items-center gap-1.5 px-4 py-2 text-sm bg-primary-600 text-white rounded-lg hover:bg-primary-700"
          >
            <FlaskConical size={14} /> 新建实验
          </button>
        </div>
      </div>

      <div className="flex items-center gap-3 mb-4">
        <div className="relative flex-1 max-w-xs">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            value={keyword}
            onChange={(e) => { setKeyword(e.target.value); setPage(1) }}
            placeholder="搜索实验名称..."
            className="w-full pl-9 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(e) => { setStatusFilter(e.target.value); setPage(1) }}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500"
        >
          <option value="">全部状态</option>
          <option value="created">已创建</option>
          <option value="running">运行中</option>
          <option value="completed">已完成</option>
          <option value="failed">失败</option>
        </select>
        <span className="text-sm text-gray-500">共 {total} 条记录</span>
      </div>

      {loading ? (
        <div className="text-center py-20 text-gray-400">
          <RefreshCw size={32} className="animate-spin mx-auto mb-3" />
          加载中...
        </div>
      ) : experiments.length === 0 ? (
        <div className="text-center py-20 text-gray-400">
          <FileText size={48} className="mx-auto mb-3 opacity-50" />
          <p className="text-lg font-medium">暂无实验记录</p>
          <p className="text-sm mt-1">开始新建设计来创建第一条实验记录</p>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="text-left px-4 py-3 font-medium text-gray-600 w-8">
                  <input type="checkbox" className="rounded" onChange={() => {
                    if (selectedIds.size === experiments.length) setSelectedIds(new Set())
                    else setSelectedIds(new Set(experiments.map(e => e.id)))
                  }} checked={selectedIds.size === experiments.length && experiments.length > 0} />
                </th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">实验名称</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">状态</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">靶点</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">设计数</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">耗时</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">创建时间</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">操作</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp) => {
                const sc = statusConfig[exp.status] || statusConfig.created
                const StatusIcon = sc.icon
                return (
                  <tr key={exp.id} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-3">
                      <input type="checkbox" className="rounded" checked={selectedIds.has(exp.id)} onChange={() => toggleSelect(exp.id)} />
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => setCurrentPage(`experiment_${exp.id}`)}
                        className="text-primary-600 hover:text-primary-800 font-medium hover:underline"
                      >
                        {exp.name}
                      </button>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${sc.color}`}>
                        <StatusIcon size={12} />
                        {sc.label}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-gray-600">{exp.target || '-'}</td>
                    <td className="px-4 py-3 text-gray-600">{exp.num_designs}</td>
                    <td className="px-4 py-3 text-gray-600">
                      {exp.duration_seconds ? `${exp.duration_seconds.toFixed(1)}s` : '-'}
                    </td>
                    <td className="px-4 py-3 text-gray-500 text-xs">{formatDate(exp.created_at)}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => handleExport(exp.id, exp.name)}
                          className="p-1.5 text-gray-400 hover:text-primary-600 hover:bg-primary-50 rounded transition-colors"
                          title="导出报告"
                        >
                          <Download size={14} />
                        </button>
                        <button
                          onClick={() => handleDelete(exp.id)}
                          className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                          title="删除"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {total > 20 && (
        <div className="flex items-center justify-center gap-2 mt-4">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="px-3 py-1.5 text-sm border rounded-lg disabled:opacity-50 hover:bg-gray-50"
          >
            上一页
          </button>
          <span className="text-sm text-gray-500">第 {page} 页</span>
          <button
            onClick={() => setPage(p => p + 1)}
            disabled={page * 20 >= total}
            className="px-3 py-1.5 text-sm border rounded-lg disabled:opacity-50 hover:bg-gray-50"
          >
            下一页
          </button>
        </div>
      )}
    </div>
  )
}
