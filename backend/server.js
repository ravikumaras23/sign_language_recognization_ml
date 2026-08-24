


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

app.get("/", (req, res) => {
  res.json({
    success: true,
    service: "SingLang Backend API",
  });
});

app.get("/health", (req, res) => {
  res.json({
    success: true,
    status: "healthy",
  });
});

app.use("/user", userRoutes);
app.use("/sign-kit/videos", videoRoutes);

const startServer = async () => {
  await connectDB();

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Backend running on port ${PORT}`);
  });
};

startServer();