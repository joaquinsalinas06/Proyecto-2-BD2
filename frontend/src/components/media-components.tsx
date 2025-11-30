"use client"

import { useState, useRef } from "react"
import { X, Play, Pause, Volume2, VolumeX, Maximize2 } from "lucide-react"

const API_BASE_URL = "http://localhost:8000"

interface MediaCellProps {
  value: any
  columnType: string
}

// Extracts the file path from various data formats
function extractPath(value: any): string | null {
  if (!value) return null

  if (typeof value === "string") {
    return value
  }

  if (typeof value === "object" && value.path) {
    return value.path
  }

  return null
}

// Converts a file path to an API URL
function pathToUrl(path: string): string {
  const normalizedPath = path.replace(/\\/g, "/")
  return `${API_BASE_URL}/files?path=${encodeURIComponent(normalizedPath)}`
}

export function MediaCell({ value, columnType }: MediaCellProps) {
  const [showModal, setShowModal] = useState(false)
  const path = extractPath(value)

  if (!path) {
    return <span className="text-gray-400">-</span>
  }

  if (columnType === "IMAGE") {
    const url = pathToUrl(path)
    return (
      <>
        <div
          className="cursor-pointer group relative"
          onClick={() => setShowModal(true)}
        >
          <img
            src={url}
            alt="Preview"
            className="h-12 w-12 object-cover rounded border border-[#2a3b43] group-hover:border-[#1193d4] transition-colors"
            onError={(e) => {
              (e.target as HTMLImageElement).src = "/placeholder-image.png"
            }}
          />
          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded">
            <Maximize2 className="h-4 w-4 text-white" />
          </div>
        </div>
        {showModal && (
          <ImageModal
            src={url}
            alt={path}
            onClose={() => setShowModal(false)}
          />
        )}
      </>
    )
  }

  if (columnType === "AUDIO") {
    const url = pathToUrl(path)
    return <AudioPlayer src={url} filename={path.split("/").pop() || "audio"} />
  }

  // Default: show path as text
  return (
    <span className="text-sm text-gray-300 truncate max-w-[200px] block" title={path}>
      {path}
    </span>
  )
}

interface ImageModalProps {
  src: string
  alt: string
  onClose: () => void
}

export function ImageModal({ src, alt, onClose }: ImageModalProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
      onClick={onClose}
    >
      <div
        className="relative max-w-[90vw] max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute -top-10 right-0 p-2 text-white hover:text-[#1193d4] transition-colors"
        >
          <X className="h-6 w-6" />
        </button>
        <img
          src={src}
          alt={alt}
          className="max-w-full max-h-[85vh] object-contain rounded-lg shadow-2xl"
        />
        <div className="absolute -bottom-8 left-0 right-0 text-center text-sm text-gray-400 truncate">
          {alt}
        </div>
      </div>
    </div>
  )
}

interface AudioPlayerProps {
  src: string
  filename: string
}

export function AudioPlayer({ src, filename }: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [progress, setProgress] = useState(0)
  const [duration, setDuration] = useState(0)
  const audioRef = useRef<HTMLAudioElement>(null)

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const toggleMute = () => {
    if (audioRef.current) {
      audioRef.current.muted = !isMuted
      setIsMuted(!isMuted)
    }
  }

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      const current = audioRef.current.currentTime
      const total = audioRef.current.duration
      setProgress((current / total) * 100)
    }
  }

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration)
    }
  }

  const handleEnded = () => {
    setIsPlaying(false)
    setProgress(0)
  }

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (audioRef.current) {
      const rect = e.currentTarget.getBoundingClientRect()
      const clickX = e.clientX - rect.left
      const percent = clickX / rect.width
      audioRef.current.currentTime = percent * audioRef.current.duration
    }
  }

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  return (
    <div className="flex items-center gap-2 min-w-[180px] p-1 rounded bg-[#1a2b33] border border-[#2a3b43]">
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={handleEnded}
      />

      <button
        onClick={togglePlay}
        className="p-1 hover:text-[#1193d4] transition-colors"
      >
        {isPlaying ? (
          <Pause className="h-4 w-4" />
        ) : (
          <Play className="h-4 w-4" />
        )}
      </button>

      <div
        className="flex-1 h-1 bg-[#2a3b43] rounded cursor-pointer"
        onClick={handleProgressClick}
      >
        <div
          className="h-full bg-[#1193d4] rounded transition-all"
          style={{ width: `${progress}%` }}
        />
      </div>

      <span className="text-xs text-gray-400 min-w-[35px]">
        {formatTime(duration)}
      </span>

      <button
        onClick={toggleMute}
        className="p-1 hover:text-[#1193d4] transition-colors"
      >
        {isMuted ? (
          <VolumeX className="h-3 w-3" />
        ) : (
          <Volume2 className="h-3 w-3" />
        )}
      </button>
    </div>
  )
}

// Distance indicator for KNN results
interface DistanceIndicatorProps {
  distance: number
  maxDistance?: number
}

export function DistanceIndicator({ distance, maxDistance = 100 }: DistanceIndicatorProps) {
  // El valor 'distance' ya viene como porcentaje de SIMILITUD (0-100) desde el backend
  // 100% = idéntico, 0% = muy diferente
  const similarity = Math.max(0, Math.min(100, distance))

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-2 bg-[#2a3b43] rounded overflow-hidden">
        <div
          className="h-full rounded transition-all"
          style={{
            width: `${similarity}%`,
            backgroundColor: similarity >= 40 ? "#10b981" : similarity >= 20 ? "#f59e0b" : "#ef4444"
          }}
        />
      </div>
      <span className="text-xs text-gray-400">
        {distance.toFixed(4)}%
      </span>
    </div>
  )
}
