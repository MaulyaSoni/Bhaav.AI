"use client"

import { Bar, BarChart, XAxis, YAxis, ResponsiveContainer, Cell } from "recharts"
import type { EmotionDistribution } from "@/lib/emotion-types"

interface EmotionChartProps {
  distribution: EmotionDistribution
}

const emotionColors: Record<string, string> = {
  Angry: "hsl(0, 91%, 71%)",
  Disgust: "hsl(160, 84%, 39%)",
  Fear: "hsl(280, 80%, 60%)",
  Happy: "hsl(45, 93%, 53%)",
  Sad: "hsl(217, 91%, 68%)",
  Surprise: "hsl(185, 80%, 55%)",
  Neutral: "hsl(0, 0%, 60%)",
}

export function EmotionChart({ distribution }: EmotionChartProps) {
  const emotionOrder = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

  // Ensure distribution exists and default missing keys to 0
  const safeDistribution: Partial<EmotionDistribution> = distribution || ({} as Partial<EmotionDistribution>)

  const data = emotionOrder.map((emotion) => ({
    emotion,
    value: (safeDistribution[emotion as keyof EmotionDistribution] as number) ?? 0,
    fill: emotionColors[emotion] || "hsl(0, 0%, 50%)",
  }))

  return (
    <div className="h-[240px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 0, right: 20 }}>
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
            axisLine={{ stroke: "hsl(var(--border))" }}
          />
          <YAxis
            dataKey="emotion"
            type="category"
            width={70}
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
            axisLine={{ stroke: "hsl(var(--border))" }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} animationDuration={1000}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
