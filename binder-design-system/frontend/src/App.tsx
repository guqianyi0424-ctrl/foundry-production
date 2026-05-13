import { Sidebar } from '@/components/Sidebar'
import { NewDesignHubPage } from '@/pages/NewDesignHubPage'
import { HelpPage } from '@/pages/HelpPage'
import { LoginModal } from '@/pages/LoginPage'
import { ExperimentsPage } from '@/pages/ExperimentsPage'
import { ExperimentDetailPage } from '@/pages/ExperimentDetailPage'
import { MonitorPage } from '@/pages/MonitorPage'
import { useAppStore } from '@/store/useAppStore'

function App() {
  const currentPage = useAppStore((s) => s.currentPage)
  const token = useAppStore((s) => s.token)

  const experimentMatch = currentPage.match(/^experiment_(.+)$/)
  const experimentId = experimentMatch ? experimentMatch[1] : null

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-deepbinder-bg">
      <Sidebar />
      <main className="flex-1 overflow-auto relative">
        {(currentPage === '新建设计' || currentPage === 'De Novo Design') && <NewDesignHubPage />}
        {currentPage === '实验记录' && <ExperimentsPage />}
        {currentPage === '系统监控' && <MonitorPage />}
        {currentPage === '帮助' && <HelpPage />}
        {experimentId && <ExperimentDetailPage experimentId={experimentId} />}
      </main>
      {!token && <LoginModal />}
    </div>
  )
}

export default App
