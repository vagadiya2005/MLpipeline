"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function OrganizeIdeas() {
  const [ideas, setIdeas] = useState([
    { id: 1, content: "Implement AI-powered idea generation" },
    { id: 2, content: "Create a user-friendly interface for mind mapping" },
    { id: 3, content: "Develop collaboration features for team brainstorming" },
  ])
  const [newIdea, setNewIdea] = useState("")

  const handleAddIdea = (e) => {
    e.preventDefault()
    if (newIdea.trim()) {
      setIdeas([...ideas, { id: Date.now(), content: newIdea }])
      setNewIdea("")
    }
  }

  return (
    <div className="container mx-auto py-12">
      <Card className="w-full max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle>Organize Your Ideas</CardTitle>
          <CardDescription>Arrange and connect your thoughts visually</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleAddIdea} className="space-y-4 mb-6">
            <div className="space-y-2">
              <Label htmlFor="newIdea">Add New Idea</Label>
              <div className="flex space-x-2">
                <Input
                  id="newIdea"
                  placeholder="Enter a new idea"
                  value={newIdea}
                  onChange={(e) => setNewIdea(e.target.value)}
                />
                <Button type="submit">Add</Button>
              </div>
            </div>
          </form>
          <div className="space-y-2">
            {ideas.map((idea) => (
              <div key={idea.id} className="p-2 bg-gray-100 rounded-md">
                {idea.content}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

