import { useState, useEffect } from 'react'
import { useAppStore } from '@/store/useAppStore'
import { getExperiment, exportExperiment, type ExperimentDetail } from '@/api'
import { ArrowLeft, Download, Clock, Cpu, CheckCircle, XCircle, FlaskConical, Activity, FileText, Dna, Box } from 'lucide-react'

const statusConfig: Record<string, { label: string; color: string }> = {
  created: { label: '已创建', color: 'bg-gray-100 text-gray-700' },
  running: { label: '运行中', color: 'bg-blue-100 text-blue-700' },
  rfd3_completed: { label: 'RFD3完成', color: 'bg-yellow-100 text-yellow-700' },
  mpnn_completed: { label: 'MPNN完成', color: 'bg-orange-100 text-orange-700' },
  completed: { label: '已完成', color: 'bg-green-100 text-green-700' },
  failed: { label: '失败', color: 'bg-red-100 text-red-700' },
}

export function ExperimentDetailPage({ experimentId }: { experimentId: string }) {
  const setCurrentPage = useAppStore((s) => s.setCurrentPage)
  const [experiment, setExperiment] = useState<ExperimentDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<'overview' | 'rfd3' | 'mpnn' | 'rf3' | 'designs'>('overview')

  useEffect(() => {
    const fetchDetail = async () => {
      setLoading(true)
      try {
        const data = await getExperiment(experimentId)
        setExperiment(data)
      } catch (err) {
        console.error('获取实验详情失败:', err)
      } finally {
        setLoading(false)
      }
    }
    fetchDetail()
  }, [experimentId])

  const handleExport = async () => {
    if (!experiment) return
    try {
      const report = await exportExperiment(experiment.id)
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${experiment.name}_report.json`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('导出失败:', err)
    }
  }

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '-'
    try { return new Date(dateStr).toLocaleString('zh-CN') } catch { return dateStr }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400">
        <Activity size={32} className="animate-spin mr-3" /> 加载中...
      </div>
    )
  }

  if (!experiment) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400">
        <XCircle size={32} className="mr-3" /> 实验记录不存在
      </div>
    )
  }

  const sc = statusConfig[experiment.status] || statusConfig.created
  const rf3Results = experiment.rf3_results || {}
  const rf3Summary = (rf3Results as any).summary || {}

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <button
          onClick={() => setCurrentPage('实验记录')}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <ArrowLeft size={20} />
        </button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-gray-900">{experiment.name}</h1>
          <div className="flex items-center gap-3 mt-1">
            <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${sc.color}`}>
              {sc.label}
            </span>
            <span className="text-xs text-gray-500 flex items-center gap-1">
              <Clock size={12} /> {formatDate(experiment.created_at)}
            </span>
            {experiment.duration_seconds && (
              <span className="text-xs text-gray-500 flex items-center gap-1">
                <Cpu size={12} /> {experiment.duration_seconds.toFixed(1)}s
              </span>
            )}
          </div>
        </div>
        <button
          onClick={handleExport}
          className="flex items-center gap-1.5 px-4 py-2 text-sm bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          <Download size={14} /> 导出报告
        </button>
      </div>

      <div className="flex gap-2 mb-6 border-b border-gray-200">
        {[
          { key: 'overview', label: '概览', icon: FileText },
          { key: 'rfd3', label: 'RFD3', icon: FlaskConical },
          { key: 'mpnn', label: 'MPNN', icon: Dna },
          { key: 'rf3', label: 'RF3', icon: Box },
          { key: 'designs', label: '设计结果', icon: Activity },
        ].map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key as any)}
            className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
              activeTab === key
                ? 'border-primary-600 text-primary-700'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <Icon size={14} /> {label}
          </button>
        ))}
      </div>

      {activeTab === 'overview' && (
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="font-medium text-gray-900 mb-3">基本信息</h3>
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-500">靶点</dt>
                <dd className="text-gray-900">{experiment.target || '-'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">热点残基</dt>
                <dd className="text-gray-900">
                  {experiment.hotspots ? JSON.stringify(experiment.hotspots) : '-'}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">GPU信息</dt>
                <dd className="text-gray-900">{experiment.gpu_info || '-'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">设计数</dt>
                <dd className="text-gray-900">{experiment.designs?.length || 0}</dd>
              </div>
            </dl>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="font-medium text-gray-900 mb-3">RF3 验证结果</h3>
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-500">RMSD</dt>
                <dd className="text-gray-900">{(rf3Results as any).rmsd?.toFixed(2) || '-'} Å</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">pLDDT</dt>
                <dd className="text-gray-900">{(rf3Results as any).avg_plddt?.toFixed(2) || '-'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">pTM</dt>
                <dd className="text-gray-900">{rf3Summary.ptm?.toFixed(3) || '-'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">iPTM</dt>
                <dd className="text-gray-900">{rf3Summary.iptm?.toFixed(3) || '-'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">验证通过</dt>
                <dd>{(rf3Results as any).passed ? <CheckCircle size={16} className="text-green-600" /> : <XCircle size={16} className="text-red-500" />}</dd>
              </div>
            </dl>
          </div>
        </div>
      )}

      {activeTab === 'rfd3' && (
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="font-medium text-gray-900 mb-3">RFD3 骨架生成结果</h3>
          {experiment.rfd3_results ? (
            <pre className="bg-gray-50 rounded-lg p-4 text-xs overflow-auto max-h-96">
              {JSON.stringify(experiment.rfd3_results, null, 2)}
            </pre>
          ) : (
            <p className="text-gray-400">暂无RFD3结果</p>
          )}
        </div>
      )}

      {activeTab === 'mpnn' && (
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="font-medium text-gray-900 mb-3">MPNN 序列设计结果</h3>
          {experiment.mpnn_results ? (
            <pre className="bg-gray-50 rounded-lg p-4 text-xs overflow-auto max-h-96">
              {JSON.stringify(experiment.mpnn_results, null, 2)}
            </pre>
          ) : (
            <p className="text-gray-400">暂无MPNN结果</p>
          )}
        </div>
      )}

      {activeTab === 'rf3' && (
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="font-medium text-gray-900 mb-3">RF3 结构验证结果</h3>
          {experiment.rf3_results ? (
            <pre className="bg-gray-50 rounded-lg p-4 text-xs overflow-auto max-h-96">
              {JSON.stringify(experiment.rf3_results, null, 2)}
            </pre>
          ) : (
            <p className="text-gray-400">暂无RF3结果</p>
          )}
        </div>
      )}

      {activeTab === 'designs' && (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          {experiment.designs && experiment.designs.length > 0 ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b">
                  <th className="text-left px-4 py-3 font-medium text-gray-600">名称</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-600">序列</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-600">pLDDT</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-600">RMSD</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-600">Ranking</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-600">验证</th>
                </tr>
              </thead>
              <tbody>
                {experiment.designs.map((d) => (
                  <tr key={d.id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="px-4 py-3 font-medium">{d.design_name || '-'}</td>
                    <td className="px-4 py-3 text-xs font-mono text-gray-600 max-w-xs truncate">
                      {d.sequence ? (d.sequence.length > 40 ? d.sequence.slice(0, 40) + '...' : d.sequence) : '-'}
                    </td>
                    <td className="px-4 py-3">{d.plddt?.toFixed(1) || '-'}</td>
                    <td className="px-4 py-3">{d.rmsd?.toFixed(2) || '-'} Å</td>
                    <td className="px-4 py-3">{d.ranking_score?.toFixed(3) || '-'}</td>
                    <td className="px-4 py-3">
                      {d.passed_validation ? (
                        <CheckCircle size={16} className="text-green-600" />
                      ) : (
                        <XCircle size={16} className="text-red-400" />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="text-center py-10 text-gray-400">暂无设计结果</div>
          )}
        </div>
      )}
    </div>
  )
}
