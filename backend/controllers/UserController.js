const User = require("../models/User");

// Signup Controller
exports.signup = async (req, res) => {
  try {
    const { name, email, password } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: "Email already in use" });
    }

    // Create new user (password stored in plaintext)
    const user = await User.create({ name, email, password });
    res.status(201).json({ message: "User created", user ,  success:true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// Login Controller
exports.login = async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find user by email
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    // Check password (plaintext comparison - INSECURE!)
    if (user.password !== password) {
      return res.status(401).json({ error: "Invalid password" });
    }

    // Successful login
    res.json({ message: "Login successful", user, success:true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
