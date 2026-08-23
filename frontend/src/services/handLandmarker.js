import {
  FilesetResolver,
  HandLandmarker,
} from "@mediapipe/tasks-vision";

let handLandmarker = null;
let loadingPromise = null;

const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

const MODEL_PATH = "/models/hand_landmarker.task";

export async function createHandLandmarker() {
  if (handLandmarker) {
    return handLandmarker;
  }

  if (loadingPromise) {
    return loadingPromise;
  }

  loadingPromise = (async () => {
    console.log("Loading MediaPipe Hand Landmarker...");

    const vision = await FilesetResolver.forVisionTasks(
      WASM_PATH
    );

    const landmarker = await HandLandmarker.createFromOptions(
      vision,
      {
        baseOptions: {
          modelAssetPath: MODEL_PATH,
        },

        runningMode: "VIDEO",

        numHands: 1,

        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      }
    );

    handLandmarker = landmarker;

    console.log(
      "MediaPipe Hand Landmarker loaded successfully."
    );

    return landmarker;
  })();

  try {
    return await loadingPromise;
  } catch (error) {
    loadingPromise = null;
    throw error;
  }
}

export function getHandLandmarker() {
  return handLandmarker;
}

export async function detectHandLandmarks(
  videoElement,
  timestamp
) {
  const landmarker = await createHandLandmarker();

  const result = landmarker.detectForVideo(
    videoElement,
    timestamp
  );

  return result;
}

export function closeHandLandmarker() {
  if (handLandmarker) {
    handLandmarker.close();
    handLandmarker = null;
  }

  loadingPromise = null;
}