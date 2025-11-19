'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import type { DatasetStats } from '@/lib/notebook-data'

interface StatisticsTableProps {
  stats: DatasetStats
}

export function StatisticsTable({ stats }: StatisticsTableProps) {
  return (
    <Card className="border-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="text-2xl">📊</span>
          Estadísticas del Dataset
        </CardTitle>
        <CardDescription>Resumen cuantitativo de los datos procesados</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Overview Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800">
              <div className="text-sm font-medium text-blue-600 dark:text-blue-400">Total Registros</div>
              <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                {stats.totalRecords.toLocaleString()}
              </div>
            </div>
            
            <div className="p-4 rounded-lg bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-800">
              <div className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Total Features</div>
              <div className="text-2xl font-bold text-emerald-900 dark:text-emerald-100">{stats.totalFeatures}</div>
            </div>
            
            <div className="p-4 rounded-lg bg-violet-50 dark:bg-violet-950 border border-violet-200 dark:border-violet-800">
              <div className="text-sm font-medium text-violet-600 dark:text-violet-400">Features Numéricas</div>
              <div className="text-2xl font-bold text-violet-900 dark:text-violet-100">{stats.numericFeatures}</div>
            </div>
            
            <div className="p-4 rounded-lg bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800">
              <div className="text-sm font-medium text-amber-600 dark:text-amber-400">Features Categóricas</div>
              <div className="text-2xl font-bold text-amber-900 dark:text-amber-100">{stats.categoricalFeatures}</div>
            </div>
          </div>

          {/* Missing Values */}
          {stats.missingValues > 0 && (
            <div className="p-4 rounded-lg bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm font-medium text-red-600 dark:text-red-400">Valores Faltantes</div>
                  <div className="text-2xl font-bold text-red-900 dark:text-red-100">
                    {stats.missingValues.toLocaleString()}
                  </div>
                </div>
                <Badge variant="destructive">
                  {((stats.missingValues / stats.totalRecords) * 100).toFixed(2)}%
                </Badge>
              </div>
            </div>
          )}

          {/* Class Distribution */}
          {stats.classDistribution && stats.classDistribution.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold mb-3">Distribución de Clases</h4>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Clase</TableHead>
                    <TableHead className="text-right">Cantidad</TableHead>
                    <TableHead className="text-right">Porcentaje</TableHead>
                    <TableHead>Visualización</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {stats.classDistribution.map((item) => (
                    <TableRow key={item.label}>
                      <TableCell className="font-medium">{item.label}</TableCell>
                      <TableCell className="text-right font-mono">{item.count.toLocaleString()}</TableCell>
                      <TableCell className="text-right">
                        <Badge variant="secondary">{item.percentage.toFixed(1)}%</Badge>
                      </TableCell>
                      <TableCell>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-blue-600 dark:bg-blue-500 h-2 rounded-full"
                            style={{ width: `${item.percentage}%` }}
                          />
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
