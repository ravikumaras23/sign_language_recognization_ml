const mongoose = require("mongoose");

const uri = process.env.MONGO_URI;

async function test() {
  try {
    console.log("Connecting to MongoDB...");
    await mongoose.connect(uri, {
      serverSelectionTimeoutMS: 10000,
    });

    console.log("MONGODB CONNECTION SUCCESS");
    await mongoose.disconnect();
  } catch (error) {
    console.error("MONGODB CONNECTION FAILED");
    console.error(error.message);
  }
}

test();