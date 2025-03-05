const fs = require("fs");
const path = require("path");
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const dotenv = require("dotenv");
const { GoogleGenerativeAI } = require("@google/generative-ai");

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Google Gemini API Configuration
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.use(cors());
app.use(express.json());
app.use(express.static("uploads"));

// ✅ Route for "/" to check if server is running
app.get("/", (req, res) => {
    res.send("Server is running!");
});

// Multer setup for image uploads
const storage = multer.diskStorage({
  destination: "uploads/",
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  },
});
const upload = multer({ storage });

app.post("/upload", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image uploaded" });
    }

    const imagePath = path.join(__dirname, "uploads", req.file.filename);
    const imageBytes = fs.readFileSync(imagePath);

    //const model = genAI.getGenerativeModel({ model: "gemini-pro-vision" });
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });


    // ✅ Corrected API request format
    const result = await model.generateContent({
      contents: [
        {
          role: "user",
          parts: [
            { text: "Is this an image of a brain? Answer only 'yes' or 'no'." },
            { inlineData: { mimeType: "image/jpeg", data: imageBytes.toString("base64") } }
          ]
        }
      ]
    });

    const responseText = result.response.text().trim().toLowerCase();
    res.json({ result: responseText });

    fs.unlinkSync(imagePath);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

//???????????????????????????
// const fs = require("fs");
// const path = require("path");
// const express = require("express");
// const cors = require("cors");
// const multer = require("multer");
// const dotenv = require("dotenv");
// const { GoogleGenerativeAI } = require("@google/generative-ai");

// dotenv.config();

// const app = express();
// const PORT = process.env.PORT || 5000;

// // Google Gemini API Configuration
// const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// app.use(cors());
// app.use(express.json());
// app.use(express.static("uploads"));

// app.get("/", (req, res) => {
//     res.send("Server is running!");
// });

// // Multer setup for image uploads
// const storage = multer.diskStorage({
//   destination: "uploads/",
//   filename: (req, file, cb) => {
//     cb(null, file.originalname);
//   },
// });
// const upload = multer({ storage });

// app.post("/upload", upload.single("image"), async (req, res) => {
//   try {
//     if (!req.file) {
//       return res.status(400).json({ error: "No image uploaded" });
//     }

//     const imagePath = path.join(__dirname, "uploads", req.file.filename);
//     const imageBytes = fs.readFileSync(imagePath);

//     const model = genAI.getGenerativeModel({ model: "gemini-pro-vision" });

//     const result = await model.generateContent([
//       {
//         role: "user",
//         parts: [
//           { text: "Is this an image of a brain? Answer only 'yes' or 'no'." },
//           { inlineData: { mimeType: "image/jpeg", data: imageBytes.toString("base64") } }
//         ]
//       }
//     ]);

//     const responseText = result.response.text().trim().toLowerCase();
//     res.json({ result: responseText });

//     fs.unlinkSync(imagePath);
//   } catch (error) {
//     console.error("Error:", error);
//     res.status(500).json({ error: error.message });
//   }
// });

// app.listen(PORT, () => {
//   console.log(`Server running on http://localhost:${PORT}`);
// });
// //********************************************************************************** */
// // const express = require("express");
// // const cors = require("cors");
// // const multer = require("multer");
// // const dotenv = require("dotenv");
// // const fs = require("fs");
// // const path = require("path");
// // const { GoogleGenerativeAI } = require("@google/generative-ai");

// // dotenv.config();

// // const app = express();
// // const PORT = process.env.PORT || 5000;

// // // Google Gemini API Configuration
// // const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// // // Enable CORS and JSON parsing
// // app.use(cors());
// // app.use(express.json());
// // app.use(express.static("uploads"));

// // // Multer setup for image uploads
// // const storage = multer.diskStorage({
// //   destination: "uploads/",
// //   filename: (req, file, cb) => {
// //     cb(null, file.originalname);
// //   },
// // });
// // const upload = multer({ storage });

// // // Root endpoint
// // app.get("/", (req, res) => {
// //   res.send("Server is running!");
// // });

// // // API Endpoint for Image Upload
// // app.post("/upload", upload.single("image"), async (req, res) => {
// //   try {
// //     if (!req.file) {
// //       return res.status(400).json({ error: "No image uploaded" });
// //     }

// //     const imagePath = path.join(__dirname, "uploads", req.file.filename);
// //     const imageBytes = fs.readFileSync(imagePath);

// //     // Ask Gemini if the image is of a brain
// //     const model = genAI.getGenerativeModel({ model: "gemini-pro-vision" });
// //     const result = await model.generateContent([
// //       { type: "text", text: "Is this an image of a brain? Answer only 'yes' or 'no'." },
// //       { type: "image", data: imageBytes },
// //     ]);

// //     const responseText = await result.response.text(); // Ensure correct async handling
// //     res.json({ result: responseText.trim().toLowerCase() });

// //     // Delete uploaded image after processing
// //     fs.unlinkSync(imagePath);
// //   } catch (error) {
// //     res.status(500).json({ error: error.message });
// //   }
// // });

// // // Start the server
// // app.listen(PORT, () => {
// //   console.log(`Server running on http://localhost:${PORT}`);
// // });
// // //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// // // const express = require("express");
// // // const cors = require("cors");
// // // const multer = require("multer");
// // // const dotenv = require("dotenv");
// // // const { GoogleGenerativeAI } = require("@google/generative-ai");

// // // dotenv.config();

// // // app.get("/", (req, res) => {
// // //     res.send("Server is running!");
// // //   });
  
// // // //dotenv.config();
// // // const app = express();
// // // const PORT = process.env.PORT || 5000;

// // // // Google Gemini API Configuration
// // // const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// // // // Enable CORS
// // // app.use(cors());
// // // app.use(express.json());
// // // app.use(express.static("uploads"));

// // // // Multer setup for image uploads
// // // const storage = multer.diskStorage({
// // //   destination: "uploads/",
// // //   filename: (req, file, cb) => {
// // //     cb(null, file.originalname);
// // //   },
// // // });
// // // const upload = multer({ storage });

// // // // API Endpoint for Image Upload
// // // app.post("/upload", upload.single("image"), async (req, res) => {
// // //   try {
// // //     if (!req.file) {
// // //       return res.status(400).json({ error: "No image uploaded" });
// // //     }

// // //     const imagePath = path.join(__dirname, "uploads", req.file.filename);
// // //     const imageBytes = fs.readFileSync(imagePath);

// // //     // Ask Gemini if the image is a brain
// // //     const model = genAI.getGenerativeModel({ model: "gemini-pro-vision" });
// // //     const result = await model.generateContent([
// // //       { type: "text", text: "Is this an image of a brain? Answer only 'yes' or 'no'." },
// // //       { type: "image", data: imageBytes },
// // //     ]);

// // //     const responseText = result.response.text().trim().toLowerCase();
// // //     res.json({ result: responseText });

// // //     // Delete uploaded image after processing
// // //     fs.unlinkSync(imagePath);
// // //   } catch (error) {
// // //     res.status(500).json({ error: error.message });
// // //   }
// // // });

// // // app.listen(PORT, () => {
// // //   console.log(`Server running on http://localhost:${PORT}`);
// // // });
