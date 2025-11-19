'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Split, Database, Workflow } from 'lucide-react'
import { cn } from '@/lib/utils'

interface NotebookItem {
  id: number
  notebook_id: string
  title: string
  description: string
  order: number
}

interface NotebookMenuProps {
  notebooks: NotebookItem[]
  selectedNotebook: string
  onSelectNotebook: (notebookId: string) => void
}

const notebookIcons = {
  notebook_07: Split,
  notebook_08: Database,
  notebook_09: Workflow,
}

const notebookColors = {
  notebook_07: 'from-purple-500 to-purple-600',
  notebook_08: 'from-green-500 to-green-600',
  notebook_09: 'from-orange-500 to-orange-600',
}

export default function NotebookMenu({
  notebooks,
  selectedNotebook,
  onSelectNotebook,
}: NotebookMenuProps) {
  return (
    <Card className="border-slate-200 dark:border-slate-800 sticky top-24">
      <CardHeader>
        <CardTitle className="text-lg">Notebooks</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {notebooks.map((notebook) => {
          const Icon = notebookIcons[notebook.notebook_id as keyof typeof notebookIcons]
          const colorClass = notebookColors[notebook.notebook_id as keyof typeof notebookColors]
          const isSelected = selectedNotebook === notebook.notebook_id

          return (
            <Button
              key={notebook.id}
              variant={isSelected ? 'secondary' : 'ghost'}
              className={cn(
                'w-full justify-start h-auto p-4 text-left',
                isSelected && 'bg-blue-100 dark:bg-blue-950/30 hover:bg-blue-100 dark:hover:bg-blue-950/30'
              )}
              onClick={() => onSelectNotebook(notebook.notebook_id)}
            >
              <div className="flex items-start gap-3 w-full">
                <div className={cn('p-2 bg-gradient-to-br rounded-lg shrink-0', colorClass)}>
                  <Icon className="h-4 w-4 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-semibold text-sm mb-1">{notebook.title}</div>
                  <div className="text-xs text-slate-600 dark:text-slate-400 line-clamp-2">
                    {notebook.description}
                  </div>
                </div>
              </div>
            </Button>
          )
        })}
      </CardContent>
    </Card>
  )
}
