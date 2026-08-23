const mongoose = require("mongoose")

const connectDB = async()=>{
    try {
        await mongoose.connect(
          "mongodb+srv://somesh:somesh@cluster0.1yrr2nf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        );
        console.log("Connected to DB!!")
    } catch (error) {
        console.log(error)
    }
}

module.exports = connectDB