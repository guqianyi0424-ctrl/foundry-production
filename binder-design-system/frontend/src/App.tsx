import { Sidebar } from '@/components/Sidebar'
import { NewDesignPage } from '@/pages/NewDesignPage'
import { JobCenterPage } from '@/pages/JobCenterPage'
import { HelpPage } from '@/pages/HelpPage'
import { LoginPage } from '@/pages/LoginPage'
import { ExperimentsPage } from '@/pages/ExperimentsPage'
import { ExperimentDetailPage } from '@/pages/ExperimentDetailPage'
import { MonitorPage } from '@/pages/MonitorPage'
import { useAppStore } from '@/store/useAppStore'

function App() {
  const currentPage = useAppStore((s) => s.currentPage)
  const token = useAppStore((s) => s.token)

  if (!token) {
    return <LoginPage />
  }

  const experimentMatch = currentPage.match(/^experiment_(.+)$/)
  const experimentId = experimentMatch ? experimentMatch[1] : null

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-odesign-bg">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        {currentPage === '新建设计' && <NewDesignPage />}
        {currentPage === '实验记录' && <ExperimentsPage />}
        {currentPage === '作业中心' && <JobCenterPage />}
        {currentPage === '系统监控' && <MonitorPage />}
        {currentPage === '帮助' && <HelpPage />}
        {experimentId && <ExperimentDetailPage experimentId={experimentId} />}
      </main>
    </div>
  )
}

export default App
