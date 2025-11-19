'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { BookOpen, Code2, Lightbulb, Database, Split, Workflow, BarChart3, GitCompare } from 'lucide-react'
import CodeBlock from './code-block'
import NotebookMenu from './notebook-menu'
import { StatisticsTable } from './statistics-table'
import { MethodComparisonTable } from './method-comparison'
import { TechnicalDetails } from './technical-details'
import { notebooksData, notebookDetails, type NotebookDetail, type NotebookItem } from '@/lib/notebook-data'

const notebookIcons = {
  notebook_07: Split,
  notebook_08: Database,
  notebook_09: Workflow,
}

export default function NotebookViewer() {
  const [notebooks, setNotebooks] = useState<NotebookItem[]>([])
  const [selectedNotebook, setSelectedNotebook] = useState<string>('')
  const [notebookDetail, setNotebookDetail] = useState<NotebookDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)

  useEffect(() => {
    fetchNotebooks()
  }, [])

  useEffect(() => {
    if (selectedNotebook) {
      fetchNotebookDetail(selectedNotebook)
    }
  }, [selectedNotebook])

  const fetchNotebooks = async () => {
    try {
      await new Promise((resolve) => setTimeout(resolve, 300))
      setNotebooks(notebooksData)
      if (notebooksData.length > 0) {
        setSelectedNotebook(notebooksData[0].notebook_id)
      }
    } catch (error) {
      console.error('Error fetching notebooks:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchNotebookDetail = async (notebookId: string) => {
    setDetailLoading(true)
    try {
      await new Promise((resolve) => setTimeout(resolve, 200))
      const detail = notebookDetails[notebookId]
      if (detail) {
        setNotebookDetail(detail)
      }
    } catch (error) {
      console.error('Error fetching notebook detail:', error)
    } finally {
      setDetailLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="container mx-auto p-6 max-w-7xl">
        <Skeleton className="h-12 w-3/4 mb-6" />
        <div className="grid gap-6">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    )
  }

  const NotebookIcon = selectedNotebook ? notebookIcons[selectedNotebook as keyof typeof notebookIcons] : BookOpen

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      <header className="bg-gradient-to-r from-blue-600 to-indigo-700 dark:from-blue-900 dark:to-indigo-950 border-b border-blue-700 dark:border-blue-800 sticky top-0 z-10 shadow-lg">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-white/10 backdrop-blur-sm rounded-xl border border-white/20">
              <BookOpen className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">
                Dataset Preparation Notebooks
              </h1>
              <p className="text-blue-100 mt-1">
                NSL-KDD Network Security Dataset - Análisis y Preprocesamiento para Machine Learning
              </p>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto p-6 max-w-7xl">
        <div className="grid lg:grid-cols-4 gap-6">
          <aside className="lg:col-span-1">
            <NotebookMenu
              notebooks={notebooks}
              selectedNotebook={selectedNotebook}
              onSelectNotebook={setSelectedNotebook}
            />
          </aside>

          <main className="lg:col-span-3">
            {detailLoading ? (
              <div className="space-y-6">
                <Skeleton className="h-32 w-full" />
                <Skeleton className="h-64 w-full" />
                <Skeleton className="h-64 w-full" />
              </div>
            ) : notebookDetail ? (
              <div className="space-y-6">
                <Card className="border-2 border-blue-200 dark:border-blue-800 overflow-hidden">
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/50 dark:to-indigo-950/50 p-6 border-b">
                    <div className="flex items-start gap-4">
                      <div className="p-4 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl shadow-lg">
                        <NotebookIcon className="h-10 w-10 text-white" />
                      </div>
                      <div className="flex-1">
                        <CardTitle className="text-3xl mb-2 text-blue-900 dark:text-blue-100">
                          {notebookDetail.title}
                        </CardTitle>
                        <CardDescription className="text-base text-blue-700 dark:text-blue-300">
                          {notebookDetail.description}
                        </CardDescription>
                      </div>
                    </div>
                  </div>
                </Card>

                {(notebookDetail.dataset_stats || notebookDetail.technical_details) && (
                  <div className="grid lg:grid-cols-3 gap-6">
                    {notebookDetail.dataset_stats && (
                      <div className="lg:col-span-2">
                        <StatisticsTable stats={notebookDetail.dataset_stats} />
                      </div>
                    )}
                    {notebookDetail.technical_details && (
                      <div className="lg:col-span-1">
                        <TechnicalDetails details={notebookDetail.technical_details} />
                      </div>
                    )}
                  </div>
                )}

                <Tabs defaultValue="sections" className="w-full">
                  <TabsList className="grid w-full grid-cols-5 h-auto p-1">
                    <TabsTrigger value="sections" className="gap-2 py-3">
                      <BookOpen className="h-4 w-4" />
                      <span className="hidden sm:inline">Secciones</span>
                    </TabsTrigger>
                    <TabsTrigger value="code" className="gap-2 py-3">
                      <Code2 className="h-4 w-4" />
                      <span className="hidden sm:inline">Código</span>
                    </TabsTrigger>
                    {notebookDetail.method_comparisons && (
                      <TabsTrigger value="comparisons" className="gap-2 py-3">
                        <GitCompare className="h-4 w-4" />
                        <span className="hidden sm:inline">Comparaciones</span>
                      </TabsTrigger>
                    )}
                    <TabsTrigger value="keypoints" className="gap-2 py-3">
                      <Lightbulb className="h-4 w-4" />
                      <span className="hidden sm:inline">Puntos Clave</span>
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="sections" className="space-y-4 mt-6">
                    {notebookDetail.sections.map((section, index) => (
                      <Card key={index} className="border-2 border-slate-200 dark:border-slate-700 hover:shadow-lg transition-shadow">
                        <CardHeader className="bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-900 dark:to-gray-900">
                          <CardTitle className="text-xl flex items-center gap-3">
                            <Badge variant="secondary" className="text-sm px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-100">
                              Sección {index + 1}
                            </Badge>
                            {section.title}
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="pt-6">
                          <p className="text-slate-700 dark:text-slate-300 leading-relaxed text-justify">
                            {section.content}
                          </p>
                        </CardContent>
                      </Card>
                    ))}
                  </TabsContent>

                  <TabsContent value="code" className="space-y-6 mt-6">
                    {notebookDetail.code_examples.map((example, index) => (
                      <Card key={index} className="border-2 border-slate-200 dark:border-slate-700 overflow-hidden">
                        <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 border-b-2">
                          <CardTitle className="text-xl flex items-center gap-3">
                            <div className="p-2 bg-blue-600 rounded-lg">
                              <Code2 className="h-5 w-5 text-white" />
                            </div>
                            {example.title}
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4 pt-6">
                          <CodeBlock code={example.code} language="python" />
                          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/50 dark:to-indigo-950/50 border-2 border-blue-200 dark:border-blue-800 rounded-lg p-5">
                            <div className="flex items-start gap-3">
                              <div className="p-2 bg-blue-600 rounded-lg shrink-0">
                                <Lightbulb className="h-4 w-4 text-white" />
                              </div>
                              <div>
                                <h4 className="font-bold text-blue-900 dark:text-blue-100 mb-2">Explicación Técnica</h4>
                                <p className="text-sm text-blue-900 dark:text-blue-100 leading-relaxed text-justify">
                                  {example.explanation}
                                </p>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </TabsContent>

                  {notebookDetail.method_comparisons && (
                    <TabsContent value="comparisons" className="mt-6">
                      <MethodComparisonTable comparisons={notebookDetail.method_comparisons} />
                    </TabsContent>
                  )}

                  <TabsContent value="keypoints" className="mt-6">
                    <Card className="border-2 border-amber-200 dark:border-amber-800">
                      <CardHeader className="bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-950/30 dark:to-yellow-950/30">
                        <CardTitle className="text-2xl flex items-center gap-3">
                          <div className="p-2 bg-amber-600 rounded-lg">
                            <Lightbulb className="h-6 w-6 text-white" />
                          </div>
                          Puntos Clave del Notebook
                        </CardTitle>
                        <CardDescription>Resumen de conceptos y técnicas principales</CardDescription>
                      </CardHeader>
                      <CardContent className="pt-6">
                        <ul className="space-y-4">
                          {notebookDetail.key_points.map((point, index) => (
                            <li
                              key={index}
                              className="flex items-start gap-4 p-4 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 hover:shadow-md transition-shadow"
                            >
                              <Badge
                                variant="outline"
                                className="mt-1 shrink-0 h-8 w-8 p-0 flex items-center justify-center bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-100 border-blue-300 dark:border-blue-700 text-sm font-bold"
                              >
                                {index + 1}
                              </Badge>
                              <span className="leading-relaxed text-slate-700 dark:text-slate-300 text-justify">
                                {point}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  </TabsContent>
                </Tabs>
              </div>
            ) : (
              <Card className="border-slate-200 dark:border-slate-800">
                <CardContent className="py-12 text-center">
                  <BookOpen className="h-16 w-16 text-slate-400 mx-auto mb-4" />
                  <p className="text-slate-500 dark:text-slate-400 text-lg">
                    Selecciona un notebook del menú para comenzar
                  </p>
                </CardContent>
              </Card>
            )}
          </main>
        </div>
      </div>
    </div>
  )
}
