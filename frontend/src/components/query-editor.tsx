"use client"

import { useRef } from "react"
import { Play, Loader2, FileImage } from "lucide-react"

interface QueryEditorProps {
  query: string
  onQueryChange: (query: string) => void
  onExecute: () => void
  isLoading?: boolean
  selectedFile?: string | null
}

export function QueryEditor({ query, onQueryChange, onExecute, isLoading = false, selectedFile }: QueryEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleInsertFile = () => {
    if (!selectedFile) return

    const textarea = textareaRef.current
    if (textarea) {
      const start = textarea.selectionStart
      const end = textarea.selectionEnd
      const newQuery = query.substring(0, start) + `'${selectedFile}'` + query.substring(end)
      onQueryChange(newQuery)

      // Restore focus and set cursor position after insert
      setTimeout(() => {
        textarea.focus()
        const newPos = start + selectedFile.length + 2
        textarea.setSelectionRange(newPos, newPos)
      }, 0)
    } else {
      // Fallback: append to query
      onQueryChange(query + `'${selectedFile}'`)
    }
  }

  const handleKNNQuery = () => {
    if (!selectedFile) return
    
    // Detectar el tipo de archivo y la tabla
    const isImage = selectedFile.toLowerCase().match(/\.(jpg|jpeg|png|gif|bmp)$/)
    const isAudio = selectedFile.toLowerCase().match(/\.(mp3|wav|ogg|flac)$/)
    
    let tableName = "fashion" // default
    let columnName = "image_path" // default
    
    if (isAudio) {
      tableName = "fma"
      columnName = "audio"
    }
    
    const knnQuery = `SELECT * FROM ${tableName}\nWHERE ${columnName} <-> '${selectedFile}'\nLIMIT 5;`
    onQueryChange(knnQuery)
  }
  return (
    <div className="p-6 border-b" style={{ borderColor: "#2a3b43" }}>
      <div className="space-y-4">
        <div>
          <label htmlFor="sql-query" className="block text-sm font-medium mb-2" style={{ color: "#e5e7eb" }}>
            Consulta SQL
          </label>
          <textarea
            ref={textareaRef}
            id="sql-query"
            name="sql-query"
            rows={8}
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="SELECT * FROM usuarios;"
            className="block w-full rounded-lg p-3 shadow-sm focus:border-[#1193d4] focus:ring-[#1193d4] sm:text-sm font-mono"
            style={{
              backgroundColor: "#1a2b33",
              borderColor: "#2a3b43",
              color: "#e5e7eb",
              border: "1px solid #2a3b43",
            }}
          />
        </div>
        <div className="flex justify-end gap-2">
          {selectedFile && (
            <>
              <button
                onClick={handleKNNQuery}
                className="inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-[#2a3b43] focus:outline-none focus:ring-2 focus:ring-[#1193d4] focus:ring-offset-2"
                style={{ backgroundColor: "#1a2b33", border: "1px solid #22c55e" }}
                title="Generar consulta KNN de similitud"
              >
                <FileImage className="mr-2 -ml-1 h-4 w-4" />
                Buscar similares
              </button>
              <button
                onClick={handleInsertFile}
                className="inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-[#2a3b43] focus:outline-none focus:ring-2 focus:ring-[#1193d4] focus:ring-offset-2"
                style={{ backgroundColor: "#1a2b33", border: "1px solid #1193d4" }}
                title={`Insertar: ${selectedFile}`}
              >
                <FileImage className="mr-2 -ml-1 h-4 w-4" />
                Insertar ruta
              </button>
            </>
          )}
          <button
            onClick={onExecute}
            disabled={isLoading}
            className="inline-flex items-center justify-center rounded-lg bg-[#1193d4] px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-[#1193d4]/90 focus:outline-none focus:ring-2 focus:ring-[#1193d4] focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ backgroundColor: "#1193d4" }}
          >
            {isLoading ? (
              <Loader2 className="mr-2 -ml-1 h-4 w-4 animate-spin" />
            ) : (
              <Play className="mr-2 -ml-1 h-4 w-4" />
            )}
            {isLoading ? "Ejecutando..." : "Ejecutar"}
          </button>
        </div>
      </div>
    </div>
  )
}
