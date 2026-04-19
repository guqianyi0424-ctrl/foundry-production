import { Sidebar } from '@/components/Sidebar'
import { NewDesignPage } from '@/pages/NewDesignPage'
import { JobCenterPage } from '@/pages/JobCenterPage'
import { HelpPage } from '@/pages/HelpPage'
import { useAppStore } from '@/store/useAppStore'

function App() {
  const currentPage = useAppStore((s) => s.currentPage)

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-odesign-bg">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        {currentPage === '新建设计' && <NewDesignPage />}
        {currentPage === '作业中心' && <JobCenterPage />}
        {currentPage === '帮助' && <HelpPage />}
      </main>
    </div>
  )
}

export default App
