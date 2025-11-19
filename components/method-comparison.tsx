'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { CheckCircle2, XCircle, AlertCircle } from 'lucide-react'
import type { MethodComparison } from '@/lib/notebook-data'

interface MethodComparisonProps {
  comparisons: MethodComparison[]
}

export function MethodComparisonTable({ comparisons }: MethodComparisonProps) {
  return (
    <Card className="border-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="text-2xl">⚖️</span>
          Comparación de Métodos
        </CardTitle>
        <CardDescription>Análisis comparativo de diferentes enfoques técnicos</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {comparisons.map((method, index) => (
            <div key={index} className="border rounded-lg overflow-hidden">
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 p-4 border-b">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-bold text-lg text-blue-900 dark:text-blue-100">{method.method}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{method.description}</p>
                  </div>
                  {method.method.includes('Utilizado') && (
                    <Badge className="bg-green-600 text-white">Implementado</Badge>
                  )}
                </div>
              </div>

              <div className="p-4 grid md:grid-cols-2 gap-4">
                {/* Advantages */}
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle2 className="h-5 w-5 text-green-600" />
                    <h4 className="font-semibold text-green-700 dark:text-green-400">Ventajas</h4>
                  </div>
                  <ul className="space-y-2">
                    {method.advantages.map((advantage, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm">
                        <span className="text-green-500 mt-0.5">•</span>
                        <span className="text-gray-700 dark:text-gray-300">{advantage}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Disadvantages */}
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <XCircle className="h-5 w-5 text-red-600" />
                    <h4 className="font-semibold text-red-700 dark:text-red-400">Desventajas</h4>
                  </div>
                  <ul className="space-y-2">
                    {method.disadvantages.map((disadvantage, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm">
                        <span className="text-red-500 mt-0.5">•</span>
                        <span className="text-gray-700 dark:text-gray-300">{disadvantage}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Use Case */}
              <div className="bg-gray-50 dark:bg-gray-900 p-4 border-t">
                <div className="flex items-start gap-2">
                  <AlertCircle className="h-5 w-5 text-amber-600 mt-0.5 flex-shrink-0" />
                  <div>
                    <h4 className="font-semibold text-sm text-amber-700 dark:text-amber-400 mb-1">Caso de Uso Ideal</h4>
                    <p className="text-sm text-gray-700 dark:text-gray-300">{method.useCase}</p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
