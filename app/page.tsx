// Fíjate en el cambio al final de esta línea: '.../NotebookViewer'
import NotebookViewer from '@/components/NotebookViewer'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      <NotebookViewer />
    </main>
  )
}
