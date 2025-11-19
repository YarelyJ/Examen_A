'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Code2, Zap, Package } from 'lucide-react'

interface TechnicalDetailsProps {
  details: {
    algorithms: string[]
    complexity: string
    prerequisites: string[]
  }
}

export function TechnicalDetails({ details }: TechnicalDetailsProps) {
  return (
    <Card className="border-2 bg-gradient-to-br from-slate-50 to-gray-100 dark:from-slate-950 dark:to-gray-900">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Code2 className="h-5 w-5" />
          Detalles Técnicos
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Algorithms */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Zap className="h-4 w-4 text-blue-600" />
            <h4 className="font-semibold text-sm">Algoritmos Utilizados</h4>
          </div>
          <div className="flex flex-wrap gap-2">
            {details.algorithms.map((algo, index) => (
              <Badge key={index} variant="secondary" className="font-mono text-xs">
                {algo}
              </Badge>
            ))}
          </div>
        </div>

        {/* Complexity */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Zap className="h-4 w-4 text-amber-600" />
            <h4 className="font-semibold text-sm">Complejidad Computacional</h4>
          </div>
          <code className="block p-3 bg-white dark:bg-gray-950 rounded border font-mono text-sm">
            {details.complexity}
          </code>
        </div>

        {/* Prerequisites */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Package className="h-4 w-4 text-green-600" />
            <h4 className="font-semibold text-sm">Dependencias Requeridas</h4>
          </div>
          <div className="flex flex-wrap gap-2">
            {details.prerequisites.map((prereq, index) => (
              <Badge key={index} variant="outline" className="font-mono text-xs">
                {prereq}
              </Badge>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
