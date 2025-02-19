const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");
require("dotenv").config(); // Load environment variables

const app = express();
app.use(bodyParser.json());

const FLASK_API_URL = process.env.FLASK_API_URL || "http://127.0.0.1:6000/predict";
const PORT = process.env.PORT || 7000;
const EXPECTED_FEATURES = ["area", "bedrooms", "age"]; // Define expected feature keys

// Prediction Endpoint
app.post("/predict", async (req, res) => {
    try {
        const { features } = req.body;

        // Validate request body
        if (!features || typeof features !== "object") {
            return res.status(400).json({ error: "Invalid request. Expected JSON with 'features' object." });
        }

        // Check for missing or unexpected features
        const missingFeatures = EXPECTED_FEATURES.filter(f => !(f in features));
        const extraFeatures = Object.keys(features).filter(f => !EXPECTED_FEATURES.includes(f));

        if (missingFeatures.length > 0) {
            return res.status(400).json({ error: `Missing features: ${missingFeatures.join(", ")}` });
        }
        if (extraFeatures.length > 0) {
            return res.status(400).json({ error: `Unexpected features: ${extraFeatures.join(", ")}` });
        }

        // Forward request to Flask API with a timeout (5s)
        const response = await axios.post(FLASK_API_URL, { features }, { timeout: 5000 });

        // Return prediction from Flask
        res.json(response.data);

    } catch (error) {
        console.error("Error calling Flask API:", error.message);

        // Handle Flask API errors
        if (error.response) {
            return res.status(error.response.status).json(error.response.data);
        } else if (error.code === "ECONNABORTED") {
            return res.status(504).json({ error: "Request to Flask API timed out." });
        } else if (error.code === "ECONNREFUSED") {
            return res.status(502).json({ error: "Flask API is unreachable. Please check if it's running." });
        }

        res.status(500).json({ error: "Internal server error." });
    }
});

// Health check endpoint
app.get("/health", (req, res) => {
    res.json({ status: "Node.js API is running" });
});

// Start server
app.listen(PORT, () => {
    console.log(`Node.js API is running on http://localhost:${PORT}`);
});
