"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function EnterSymptoms() {
  const [symptoms, setSymptoms] = useState("")
  const [age, setAge] = useState("")
  const [gender, setGender] = useState("")

  const handleSubmit = (e) => {
    e.preventDefault()
    // Here you would typically handle the symptom submission logic
    console.log("Symptoms submitted:", { symptoms, age, gender })
    // For now, we'll just log the data
  }

  return (
    <div className="container mx-auto py-12">
      <Card className="w-full max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle>Enter Your Symptoms</CardTitle>
          <CardDescription>Provide information about your symptoms for AI analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="symptoms">Symptoms</Label>
              <Textarea
                id="symptoms"
                placeholder="Describe your symptoms here..."
                value={symptoms}
                onChange={(e) => setSymptoms(e.target.value)}
                required
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="age">Age</Label>
                <Input
                  id="age"
                  type="number"
                  placeholder="Enter your age"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="gender">Gender</Label>
                <select
                  id="gender"
                  className="w-full p-2 border rounded-md"
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                  required
                >
                  <option value="">Select gender</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                </select>
              </div>
            </div>
            <Button type="submit" className="w-full">
              Submit Symptoms
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}

