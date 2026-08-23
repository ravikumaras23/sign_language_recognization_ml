require("dotenv").config();
const mongoose = require("mongoose");
const User = require("./models/User");

async function deleteAllUsers() {
  try {
    await mongoose.connect(process.env.MONGO_URI);

    console.log("Connected to MongoDB");

    const result = await User.deleteMany({});

    console.log(`Deleted ${result.deletedCount} users.`);

    await mongoose.disconnect();
    process.exit(0);
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

deleteAllUsers();