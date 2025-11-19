// CAMBIO IMPORTANTE:
// 1. Usamos '../' en lugar de '@/'
// 2. Usamos 'notebook-viewer' en minúsculas (tal como sale en tu foto)
import NotebookViewer from '../components/notebook-viewer'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      <NotebookViewer />
    </main>
  )
}
