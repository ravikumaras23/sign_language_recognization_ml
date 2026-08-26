const express = require("express");
const cors = require("cors");

const userRoutes = require("./routes/UserRoutes");
const videoRoutes = require("./routes/VideoRoutes");
const connectDB = require("./config/db");

const app = express();

const PORT = process.env.PORT || 10000;

app.use(
  cors({
    origin: true,
    credentials: false,
  })
);

app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// ============================================================
// BASIC ROUTES
// ============================================================

app.get("/", (req, res) => {
  res.json({
    success: true,
    service: "SingLangBackend API",
  });
});

app.get("/health", (req, res) => {
  res.json({
    success: true,
    status: "healthy",
  });
});

// ============================================================
// API ROUTES
// ============================================================

app.use("/user", userRoutes);
app.use("/sign-kit/videos", videoRoutes);

// ============================================================
// START SERVER
// ============================================================

const startServer = async () => {
    await connectDB();

  // Start Express immediately so Render can detect the service.
  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Backend running on port ${PORT}`);
  });

  // Connect to MongoDB after the HTTP server is available.
  const connected = await connectDB();

  if (!connected) {
    console.error(
      "WARNING: Backend is running, but MongoDB is not connected."
    );
  }
};

startServer();