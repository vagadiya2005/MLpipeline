"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function UploadImage() {
  const [file, setFile] = useState(null)

  const handleFileChange = (e) => {
    if (e.target.files) {
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (file) {
      // Here you would typically handle the image upload logic
      console.log("Image uploaded:", file.name)
      // For now, we'll just log the file name
    }
  }

  return (
    <div className="container mx-auto py-12">
      <Card className="w-full max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle>Upload Medical Image</CardTitle>
          <CardDescription>Upload an image for AI-powered diagnosis</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="image">Select Image</Label>
              <input
                id="image"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="w-full p-2 border rounded-md"
                required
              />
            </div>
            {file && (
              <div className="mt-4">
                <p>Selected file: {file.name}</p>
                <img
                  src={URL.createObjectURL(file) || "/placeholder.svg"}
                  alt="Preview"
                  className="mt-2 max-w-full h-auto rounded-md"
                />
              </div>
            )}
            <Button type="submit" className="w-full" disabled={!file}>
              Upload Image
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}

