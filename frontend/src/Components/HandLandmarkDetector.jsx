import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  FilesetResolver,
  HandLandmarker,
} from "@mediapipe/tasks-vision";


const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm";

const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";


const FEATURE_COUNT = 84;

const LEFT_HAND = "Left";
const RIGHT_HAND = "Right";


function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;

  return Math.sqrt(
    dx * dx +
    dy * dy +
    dz * dz
  );
}


/**
 * Convert MediaPipe's hand landmarks into
 * the 84 features expected by the backend.
 *
 * 2 hands
 * × 21 landmarks
 * × 2 coordinates (x,y)
 * = 84
 */
function extract84Features(result) {

  const left = new Array(42).fill(0);
  const right = new Array(42).fill(0);

  if (
    !result ||
    !result.landmarks ||
    !result.handednesses
  ) {
    return [
      ...left,
      ...right,
    ];
  }


  for (
    let handIndex = 0;
    handIndex < result.landmarks.length;
    handIndex++
  ) {

    const landmarks =
      result.landmarks[handIndex];

    const handedness =
      result.handednesses[handIndex]?.[0];


    if (!landmarks || landmarks.length !== 21) {
      continue;
    }


    const category =
      handedness?.categoryName;


    const target =
      category === LEFT_HAND
        ? left
        : category === RIGHT_HAND
          ? right
          : null;


    if (!target) {
      continue;
    }


    for (
      let landmarkIndex = 0;
      landmarkIndex < 21;
      landmarkIndex++
    ) {

      const landmark =
        landmarks[landmarkIndex];


      target[
        landmarkIndex * 2
      ] = landmark.x;

      target[
        landmarkIndex * 2 + 1
      ] = landmark.y;
    }
  }


  return [
    ...left,
    ...right,
  ];
}


/**
 * Normalize coordinates relative to the wrist.
 *
 * This makes the model less sensitive to:
 * - hand position
 * - distance from camera
 * - movement around the screen
 */
function normalize84Features(features) {

  if (
    !features ||
    features.length !== FEATURE_COUNT
  ) {
    return features;
  }


  const normalized =
    [...features];


  // Left wrist = indices 0,1
  const leftWristX =
    normalized[0];

  const leftWristY =
    normalized[1];


  // Right wrist = indices 42,43
  const rightWristX =
    normalized[42];

  const rightWristY =
    normalized[43];


  // ----------------------------------------------------------
  // Left hand
  // ----------------------------------------------------------

  const leftPoints = [];

  for (let i = 0; i < 21; i++) {

    leftPoints.push({
      x: normalized[i * 2],
      y: normalized[i * 2 + 1],
    });
  }


  const leftScale =
    Math.max(
      distance(
        {
          x: leftPoints[0].x,
          y: leftPoints[0].y,
          z: 0,
        },
        {
          x: leftPoints[9].x,
          y: leftPoints[9].y,
          z: 0,
        }
      ),
      0.0001
    );


  for (let i = 0; i < 21; i++) {

    normalized[i * 2] =
      (
        normalized[i * 2] -
        leftWristX
      ) / leftScale;


    normalized[i * 2 + 1] =
      (
        normalized[i * 2 + 1] -
        leftWristY
      ) / leftScale;
  }


  // ----------------------------------------------------------
  // Right hand
  // ----------------------------------------------------------

  const rightPoints = [];

  for (let i = 0; i < 21; i++) {

    rightPoints.push({
      x: normalized[42 + i * 2],
      y: normalized[42 + i * 2 + 1],
    });
  }


  const rightScale =
    Math.max(
      distance(
        {
          x: rightPoints[0].x,
          y: rightPoints[0].y,
          z: 0,
        },
        {
          x: rightPoints[9].x,
          y: rightPoints[9].y,
          z: 0,
        }
      ),
      0.0001
    );


  for (let i = 0; i < 21; i++) {

    normalized[42 + i * 2] =
      (
        normalized[42 + i * 2] -
        rightWristX
      ) / rightScale;


    normalized[42 + i * 2 + 1] =
      (
        normalized[42 + i * 2 + 1] -
        rightWristY
      ) / rightScale;
  }


  return normalized;
}


export default function HandLandmarkDetector({
  active = false,
  onLandmarks,
  onDetection,
  onError,
}) {

  const videoRef =
    useRef(null);

  const canvasRef =
    useRef(null);

  const handLandmarkerRef =
    useRef(null);

  const streamRef =
    useRef(null);

  const animationFrameRef =
    useRef(null);

  const lastVideoTimeRef =
    useRef(-1);

  const lastSendTimeRef =
    useRef(0);


  const [ready, setReady] =
    useState(false);

  const [cameraReady, setCameraReady] =
    useState(false);

  const [handCount, setHandCount] =
    useState(0);

  const [error, setError] =
    useState(null);


  const initializeMediaPipe =
    useCallback(
      async () => {

        try {

          setError(null);


          console.log(
            "Loading MediaPipe..."
          );


          const vision =
            await FilesetResolver.forVisionTasks(
              WASM_PATH
            );


          const handLandmarker =
            await HandLandmarker.createFromOptions(
              vision,
              {
                baseOptions: {
                  modelAssetPath:
                    MODEL_PATH,

                  delegate: "GPU",
                },

                runningMode:
                  "VIDEO",

                numHands: 2,

                minHandDetectionConfidence:
                  0.55,

                minHandPresenceConfidence:
                  0.55,

                minTrackingConfidence:
                  0.55,
              }
            );


          handLandmarkerRef.current =
            handLandmarker;


          setReady(true);


          console.log(
            "MediaPipe Hands ready."
          );

        } catch (err) {

          console.error(
            "MediaPipe initialization failed:",
            err
          );


          setError(
            err.message ||
            "Could not initialize MediaPipe."
          );


          onError?.(err);
        }
      },
      [onError]
    );


  const startCamera =
    useCallback(
      async () => {

        try {

          setError(null);


          if (!navigator.mediaDevices) {

            throw new Error(
              "Camera API is not available."
            );
          }


          const stream =
            await navigator.mediaDevices.getUserMedia(
              {
                video: {
                  facingMode: "user",

                  width: {
                    ideal: 1280,
                  },

                  height: {
                    ideal: 720,
                  },

                  frameRate: {
                    ideal: 30,
                    max: 30,
                  },
                },

                audio: false,
              }
            );


          streamRef.current =
            stream;


          const video =
            videoRef.current;


          if (!video) {
            return;
          }


          video.srcObject =
            stream;


          video.playsInline =
            true;

          video.muted =
            true;


          await video.play();


          setCameraReady(true);


          console.log(
            "Camera started."
          );

        } catch (err) {

          console.error(
            "Camera error:",
            err
          );


          setError(
            err.message ||
            "Could not access camera."
          );


          onError?.(err);
        }
      },
      [onError]
    );


  const stopCamera =
    useCallback(() => {

      if (animationFrameRef.current) {

        cancelAnimationFrame(
          animationFrameRef.current
        );

        animationFrameRef.current =
          null;
      }


      if (streamRef.current) {

        streamRef.current
          .getTracks()
          .forEach(
            (track) =>
              track.stop()
          );

        streamRef.current =
          null;
      }


      if (videoRef.current) {

        videoRef.current.srcObject =
          null;
      }


      setCameraReady(false);
      setHandCount(0);

    }, []);


  const processFrame =
    useCallback(
      (timestamp) => {

        const video =
          videoRef.current;

        const canvas =
          canvasRef.current;

        const detector =
          handLandmarkerRef.current;


        if (
          !video ||
          !canvas ||
          !detector ||
          video.readyState < 2
        ) {

          animationFrameRef.current =
            requestAnimationFrame(
              processFrame
            );

          return;
        }


        // Don't process the exact same video frame twice.
        if (
          video.currentTime ===
          lastVideoTimeRef.current
        ) {

          animationFrameRef.current =
            requestAnimationFrame(
              processFrame
            );

          return;
        }


        lastVideoTimeRef.current =
          video.currentTime;


        try {

          const result =
            detector.detectForVideo(
              video,
              timestamp
            );


          const count =
            result.landmarks?.length || 0;


          setHandCount(count);


          // --------------------------------------------------
          // Draw landmarks
          // --------------------------------------------------

          const ctx =
            canvas.getContext("2d");


          canvas.width =
            video.videoWidth;

          canvas.height =
            video.videoHeight;


          ctx.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
          );


          if (count > 0) {

            drawHands(
              ctx,
              result,
              canvas.width,
              canvas.height
            );
          }


          // --------------------------------------------------
          // Extract 84 features
          // --------------------------------------------------

          const rawFeatures =
            extract84Features(
              result
            );


          const normalizedFeatures =
            normalize84Features(
              rawFeatures
            );


          onLandmarks?.(
            normalizedFeatures,
            result
          );


          onDetection?.({
            handCount: count,

            landmarks:
              normalizedFeatures,

            rawResult:
              result,
          });


        } catch (err) {

          console.error(
            "Hand detection error:",
            err
          );
        }


        animationFrameRef.current =
          requestAnimationFrame(
            processFrame
          );

      },
      [
        onLandmarks,
        onDetection,
      ]
    );


  // ----------------------------------------------------------
  // Initialize MediaPipe
  // ----------------------------------------------------------

  useEffect(() => {

    initializeMediaPipe();

    return () => {

      stopCamera();

      if (
        handLandmarkerRef.current
      ) {

        handLandmarkerRef.current.close();

        handLandmarkerRef.current =
          null;
      }
    };

  }, [
    initializeMediaPipe,
    stopCamera,
  ]);


  // ----------------------------------------------------------
  // Start / stop camera
  // ----------------------------------------------------------

  useEffect(() => {

    if (!active) {

      stopCamera();

      return;
    }


    if (!ready) {
      return;
    }


    startCamera();

  }, [
    active,
    ready,
    startCamera,
    stopCamera,
  ]);


  // ----------------------------------------------------------
  // Start detection loop
  // ----------------------------------------------------------

  useEffect(() => {

    if (
      !active ||
      !ready ||
      !cameraReady
    ) {
      return;
    }


    animationFrameRef.current =
      requestAnimationFrame(
        processFrame
      );


    return () => {

      if (
        animationFrameRef.current
      ) {

        cancelAnimationFrame(
          animationFrameRef.current
        );

        animationFrameRef.current =
          null;
      }
    };

  }, [
    active,
    ready,
    cameraReady,
    processFrame,
  ]);


  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        maxWidth: "720px",
        margin: "0 auto",
        background: "#000",
        borderRadius: "12px",
        overflow: "hidden",
      }}
    >

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: "100%",
          display: "block",
          transform: "scaleX(-1)",
        }}
      />


      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          transform: "scaleX(-1)",
          pointerEvents: "none",
        }}
      />


      <div
        style={{
          position: "absolute",
          top: 12,
          left: 12,
          padding: "8px 12px",
          borderRadius: "8px",
          background:
            "rgba(0,0,0,0.65)",
          color: "#fff",
          fontSize: "14px",
        }}
      >
        {!ready
          ? "Loading hand detector..."
          : !cameraReady
            ? "Starting camera..."
            : `Hands detected: ${handCount}`}
      </div>


      {error && (
        <div
          style={{
            position: "absolute",
            bottom: 12,
            left: 12,
            right: 12,
            padding: "10px",
            borderRadius: "8px",
            background:
              "rgba(180,0,0,0.85)",
            color: "#fff",
          }}
        >
          {error}
        </div>
      )}

    </div>
  );
}


/**
 * Draw 21-point MediaPipe hand landmarks.
 */
function drawHands(
  ctx,
  result,
  width,
  height
) {

  const connections = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],

    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],

    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],

    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],

    [13, 17],
    [17, 18],
    [18, 19],
    [19, 20],

    [0, 17],
  ];


  for (
    const landmarks of
    result.landmarks || []
  ) {

    // Lines
    ctx.beginPath();

    for (
      const [a, b] of connections
    ) {

      const p1 =
        landmarks[a];

      const p2 =
        landmarks[b];


      if (!p1 || !p2) {
        continue;
      }


      ctx.moveTo(
        p1.x * width,
        p1.y * height
      );


      ctx.lineTo(
        p2.x * width,
        p2.y * height
      );
    }


    ctx.strokeStyle =
      "#00ff88";

    ctx.lineWidth =
      3;

    ctx.stroke();


    // Points
    for (
      const point of landmarks
    ) {

      ctx.beginPath();

      ctx.arc(
        point.x * width,
        point.y * height,
        5,
        0,
        Math.PI * 2
      );


      ctx.fillStyle =
        "#00ff88";

      ctx.fill();
    }
  }
}