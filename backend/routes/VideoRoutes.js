const express = require("express");

const {
  createVideo,
  getAllVideos,
  getVideo,
} = require("../controllers/VideoController");

const router = express.Router();

router.post("/create-video", createVideo);

router.get("/all-videos", getAllVideos);

router.get("/:videoId", getVideo);

module.exports = router;