const crypto = require("crypto");
const Video = require("../models/Video");

const generateVideoId = () => {
  return crypto.randomBytes(6).toString("hex").toUpperCase();
};

exports.createVideo = async (req, res) => {
  try {
    const {
      title,
      desc,
      createdBy,
      type = "PUBLIC",
      content,
    } = req.body;

    if (!title || !desc || !createdBy || !content) {
      return res.status(400).json({
        success: false,
        message: "title, desc, createdBy and content are required",
      });
    }

    let videoId;
    let exists = true;

    while (exists) {
      videoId = generateVideoId();
      exists = await Video.exists({ videoId });
    }

    const video = await Video.create({
      videoId,
      title,
      desc,
      createdBy,
      type,
      content,
    });

    return res.status(201).json({
      success: true,
      videoId: video.videoId,
      video,
    });
  } catch (error) {
    console.error("Create video error:", error);

    return res.status(500).json({
      success: false,
      message: "Failed to create video",
      error: error.message,
    });
  }
};

exports.getAllVideos = async (req, res) => {
  try {
    const videos = await Video.find({
      type: "PUBLIC",
    }).sort({
      createdAt: -1,
    });

    return res.json(videos);
  } catch (error) {
    console.error("Get videos error:", error);

    return res.status(500).json({
      success: false,
      message: "Failed to retrieve videos",
    });
  }
};

exports.getVideo = async (req, res) => {
  try {
    const { videoId } = req.params;

    const video = await Video.findOne({
      videoId,
    });

    if (!video) {
      return res.status(404).json({
        success: false,
        message: "Video not found",
      });
    }

    return res.json(video);
  } catch (error) {
    console.error("Get video error:", error);

    return res.status(500).json({
      success: false,
      message: "Failed to retrieve video",
    });
  }
};