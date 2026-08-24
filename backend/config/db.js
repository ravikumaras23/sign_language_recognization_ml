const mongoose = require("mongoose");

const connectDB = async () => {
  if (!process.env.MONGO_URI) {
    console.error("MONGO_URI environment variable is not set");
    return false;
  }

  try {
    await mongoose.connect(process.env.MONGO_URI, {
      serverSelectionTimeoutMS: 10000,
      connectTimeoutMS: 10000,
    });

    console.log("MongoDB connected successfully");
    return true;
  } catch (error) {
    console.error("MongoDB connection failed:", error.message);
    return false;
  }
};

module.exports = connectDB;