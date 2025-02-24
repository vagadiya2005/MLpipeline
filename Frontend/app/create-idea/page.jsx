"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function CreateIdea() {
  const [topic, setTopic] = useState("")
  const [description, setDescription] = useState("")

  const handleSubmit = (e) => {
    e.preventDefault()
    // Here you would typically handle the idea submission logic
    console.log("Idea submitted:", { topic, description })
    // For now, we'll just log the data
  }

  return (
    <div className="container mx-auto py-12">
      <Card className="w-full max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle>Create New Idea</CardTitle>
          <CardDescription>Let our AI help you generate and refine your ideas</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="topic">Topic</Label>
              <Input
                id="topic"
                placeholder="Enter the main topic or theme"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                placeholder="Describe your idea or concept..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                required
              />
            </div>
            <Button type="submit" className="w-full">
              Generate Ideas
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}

