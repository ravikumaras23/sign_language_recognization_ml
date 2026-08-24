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


// ============================================================
// CONFIGURATION
// ============================================================

//const API_BASE_URL = "http://127.0.0.1:8000";

const MEDIAPIPE_WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm";

const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";


// ============================================================
// COMPONENT
// ============================================================

function SignLanguageCamera({
  language = "ISL",
  onPrediction,
}) {
  // ----------------------------------------------------------
  // DOM references
  // ----------------------------------------------------------

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const streamRef = useRef(null);
  const handLandmarkerRef = useRef(null);

  const animationFrameRef = useRef(null);

  const lastVideoTimeRef = useRef(-1);

  const mountedRef = useRef(true);

  const processingRef = useRef(false);

  const lastPredictionTimeRef = useRef(0);


  // ----------------------------------------------------------
  // State
  // ----------------------------------------------------------

  const [cameraActive, setCameraActive] =
    useState(false);

  const [detectionActive, setDetectionActive] =
    useState(false);

  const [loadingMediaPipe, setLoadingMediaPipe] =
    useState(true);

  const [cameraError, setCameraError] =
    useState("");

  const [landmarkCount, setLandmarkCount] =
    useState(0);

  const [currentPrediction, setCurrentPrediction] =
    useState(null);

  const [confidence, setConfidence] =
    useState(0);

  const [status, setStatus] =
    useState("Initializing...");


  // ==========================================================
  // CREATE MEDIAPIPE HAND LANDMARKER
  // ==========================================================

  const initializeMediaPipe =
    useCallback(async () => {

      try {
        setLoadingMediaPipe(true);
        setStatus("Loading hand detection...");

        console.log(
          "Loading MediaPipe Tasks Vision..."
        );

        const vision =
          await FilesetResolver.forVisionTasks(
            MEDIAPIPE_WASM_URL
          );

        const handLandmarker =
          await HandLandmarker.createFromOptions(
            vision,
            {
              baseOptions: {
                modelAssetPath:
                  HAND_MODEL_URL,

                delegate: "GPU",
              },

              runningMode: "VIDEO",

              numHands: 2,

              minHandDetectionConfidence:
                0.5,

              minHandPresenceConfidence:
                0.5,

              minTrackingConfidence:
                0.5,
            }
          );

        if (!mountedRef.current) {
          handLandmarker.close();
          return;
        }

        handLandmarkerRef.current =
          handLandmarker;

        setLoadingMediaPipe(false);

        setStatus(
          "Hand detector ready"
        );

        console.log(
          "MediaPipe HandLandmarker ready"
        );

      } catch (error) {

        console.error(
          "MediaPipe initialization failed:",
          error
        );

        setLoadingMediaPipe(false);

        setCameraError(
          "Could not initialize MediaPipe hand detection: " +
          error.message
        );

        setStatus(
          "MediaPipe initialization failed"
        );
      }
    }, []);


  // ==========================================================
  // START CAMERA
  // ==========================================================

  const startCamera =
    useCallback(async () => {

      try {

        setCameraError("");

        setStatus(
          "Requesting camera permission..."
        );

        const stream =
          await navigator.mediaDevices.getUserMedia(
            {
              video: {
                width: {
                  ideal: 1280,
                },

                height: {
                  ideal: 720,
                },

                facingMode: "user",
              },

              audio: false,
            }
          );

        streamRef.current = stream;

        const video =
          videoRef.current;

        if (!video) {
          throw new Error(
            "Video element is not available."
          );
        }

        video.srcObject = stream;

        await video.play();

        if (!mountedRef.current) {
          return;
        }

        setCameraActive(true);

        setStatus(
          "Camera running"
        );

        console.log(
          "Camera started"
        );

      } catch (error) {

        console.error(
          "Camera error:",
          error
        );

        setCameraError(
          "Could not access webcam: " +
          error.message
        );

        setCameraActive(false);

        setStatus(
          "Camera unavailable"
        );
      }
    }, []);


  // ==========================================================
  // STOP CAMERA
  // ==========================================================

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
          .forEach((track) => {
            track.stop();
          });

        streamRef.current = null;
      }

      if (videoRef.current) {
        videoRef.current.srcObject =
          null;
      }

      setCameraActive(false);

      setStatus(
        "Camera stopped"
      );

      console.log(
        "Camera stopped"
      );

    }, []);


  // ==========================================================
  // START BACKEND DETECTION
  // ==========================================================

  const startBackendDetection =
    useCallback(async () => {

      try {

        const response =
          await fetch(
            `${API_BASE_URL}/start_detection`,
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json",
              },

              body: JSON.stringify({
                language,
              }),
            }
          );

        const data =
          await response.json();

        if (!response.ok) {

          throw new Error(
            data.detail ||
            "Failed to start detection"
          );
        }

        console.log(
          "Backend detection started:",
          data
        );

        setDetectionActive(true);

        setStatus(
          "Detection is running"
        );

      } catch (error) {

        console.error(
          "Backend detection error:",
          error
        );

        setCameraError(
          "Backend detection failed: " +
          error.message
        );

        setDetectionActive(false);

        setStatus(
          "Backend unavailable"
        );
      }
    }, [language]);


  // ==========================================================
  // STOP BACKEND DETECTION
  // ==========================================================

  const stopBackendDetection =
    useCallback(async () => {

      try {

        await fetch(
          `${API_BASE_URL}/stop_detection`,
          {
            method: "POST",
          }
        );

      } catch (error) {

        console.error(
          "Stop detection error:",
          error
        );
      }

      setDetectionActive(false);

    }, []);


  // ==========================================================
  // CONVERT LANDMARKS TO FEATURES
  // ==========================================================

  const buildFeatures =
    useCallback((landmarks) => {

      if (
        !landmarks ||
        landmarks.length === 0
      ) {
        return null;
      }

      /*
       * MediaPipe Hands:
       *
       * 21 landmarks
       *
       * Each landmark:
       *   x
       *   y
       *   z
       *
       * Therefore:
       *
       * 21 × 3 = 63 features
       *
       *
       * BUT:
       *
       * Your LSTM model expects:
       *
       *     84 features
       *
       * So your original training pipeline must have
       * generated an additional 21 features.
       *
       * We MUST reproduce those exact features for
       * correct predictions.
       */


      // ------------------------------------------------------
      // First 63 features: x, y, z
      // ------------------------------------------------------

      const xyz = [];

      for (
        let i = 0;
        i < landmarks.length;
        i++
      ) {

        const point =
          landmarks[i];

        xyz.push(
          Number(point.x || 0),
          Number(point.y || 0),
          Number(point.z || 0)
        );
      }


      /*
       * TEMPORARY 84-FEATURE ADAPTER
       *
       * If your model absolutely requires 84 values,
       * we currently append 21 zeros.
       *
       * WARNING:
       *
       * This allows the API/model shape to be tested,
       * but it is NOT guaranteed to reproduce your
       * training preprocessing.
       *
       * Once you give me the original feature-extraction/
       * training code, replace this section with the
       * exact 84-feature calculation.
       */

      const additionalFeatures =
        new Array(21).fill(0);


      const features = [
        ...xyz,
        ...additionalFeatures,
      ];


      if (features.length !== 84) {

        console.error(
          "Invalid feature count:",
          features.length
        );

        return null;
      }


      return features;

    }, []);


  // ==========================================================
  // DRAW HAND LANDMARKS
  // ==========================================================

  const drawLandmarks =
    useCallback(
      (results) => {

        const video =
          videoRef.current;

        const canvas =
          canvasRef.current;

        if (!video || !canvas) {
          return;
        }

        const ctx =
          canvas.getContext("2d");

        if (!ctx) {
          return;
        }


        const width =
          video.videoWidth ||
          640;

        const height =
          video.videoHeight ||
          480;


        if (
          canvas.width !== width ||
          canvas.height !== height
        ) {

          canvas.width =
            width;

          canvas.height =
            height;
        }


        ctx.clearRect(
          0,
          0,
          width,
          height
        );


        if (
          !results ||
          !results.landmarks
        ) {
          return;
        }


        for (
          const hand of
          results.landmarks
        ) {

          // --------------------------------------------------
          // Connections
          // --------------------------------------------------

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


          // --------------------------------------------------
          // Draw connections
          // --------------------------------------------------

          ctx.lineWidth = 3;

          ctx.strokeStyle =
            "#00ff88";

          for (
            const [
              startIndex,
              endIndex,
            ] of connections
          ) {

            const start =
              hand[startIndex];

            const end =
              hand[endIndex];

            if (!start || !end) {
              continue;
            }

            ctx.beginPath();

            ctx.moveTo(
              start.x * width,
              start.y * height
            );

            ctx.lineTo(
              end.x * width,
              end.y * height
            );

            ctx.stroke();
          }


          // --------------------------------------------------
          // Draw points
          // --------------------------------------------------

          for (
            const point of hand
          ) {

            const x =
              point.x * width;

            const y =
              point.y * height;

            ctx.beginPath();

            ctx.arc(
              x,
              y,
              5,
              0,
              Math.PI * 2
            );

            ctx.fillStyle =
              "#00ffff";

            ctx.fill();

            ctx.strokeStyle =
              "#ffffff";

            ctx.lineWidth = 1;

            ctx.stroke();
          }
        }

      },
      []
    );


  // ==========================================================
  // SEND LANDMARKS TO FASTAPI
  // ==========================================================

  const sendLandmarksToAPI =
    useCallback(
      async (landmarks) => {

        if (
          !detectionActive ||
          processingRef.current
        ) {
          return;
        }


        // ----------------------------------------------------
        // Avoid sending too many requests
        // ----------------------------------------------------

        const now =
          Date.now();

        if (
          now -
          lastPredictionTimeRef.current <
          150
        ) {
          return;
        }


        lastPredictionTimeRef.current =
          now;


        // ----------------------------------------------------
        // Convert landmarks
        // ----------------------------------------------------

        const features =
          buildFeatures(
            landmarks
          );


        if (!features) {
          return;
        }


        processingRef.current =
          true;


        try {

          const response =
            await fetch(
              `${API_BASE_URL}/process_frame`,
              {
                method: "POST",

                headers: {
                  "Content-Type":
                    "application/json",
                },

                body: JSON.stringify({
                  language,

                  landmarks:
                    features,
                }),
              }
            );


          const data =
            await response.json();


          if (!response.ok) {

            console.error(
              "ML API error:",
              data
            );

            return;
          }


          if (!data.success) {

            console.warn(
              "Prediction unsuccessful:",
              data.message
            );

            return;
          }


          // --------------------------------------------------
          // Prediction
          // --------------------------------------------------

          if (
            data.prediction !== null &&
            data.prediction !== undefined
          ) {

            setCurrentPrediction(
              data.prediction
            );

            setConfidence(
              Number(
                data.confidence || 0
              )
            );


            if (
              typeof onPrediction ===
              "function"
            ) {

              onPrediction(data);
            }
          }


        } catch (error) {

          console.error(
            "Failed to send landmarks:",
            error
          );

        } finally {

          processingRef.current =
            false;
        }

      },
      [
        buildFeatures,
        detectionActive,
        language,
        onPrediction,
      ]
    );


  // ==========================================================
  // PROCESS CAMERA FRAME
  // ==========================================================

  const processFrame =
    useCallback(() => {

      const video =
        videoRef.current;

      const handLandmarker =
        handLandmarkerRef.current;


      if (
        !video ||
        !handLandmarker ||
        video.readyState <
          HTMLMediaElement
            .HAVE_CURRENT_DATA
      ) {

        animationFrameRef.current =
          requestAnimationFrame(
            processFrame
          );

        return;
      }


      const currentTime =
        video.currentTime;


      if (
        currentTime !==
        lastVideoTimeRef.current
      ) {

        lastVideoTimeRef.current =
          currentTime;


        try {

          const results =
            handLandmarker.detectForVideo(
              video,
              performance.now()
            );


          // --------------------------------------------------
          // Draw landmarks
          // --------------------------------------------------

          drawLandmarks(
            results
          );


          // --------------------------------------------------
          // Hand count
          // --------------------------------------------------

          const hands =
            results.landmarks || [];


          setLandmarkCount(
            hands.length
          );


          // --------------------------------------------------
          // Send first detected hand
          // --------------------------------------------------

          if (
            hands.length > 0 &&
            detectionActive
          ) {

            sendLandmarksToAPI(
              hands[0]
            );

          } else {

            setLandmarkCount(0);
          }


        } catch (error) {

          console.error(
            "Frame processing error:",
            error
          );
        }
      }


      animationFrameRef.current =
        requestAnimationFrame(
          processFrame
        );

    }, [
      detectionActive,
      drawLandmarks,
      sendLandmarksToAPI,
    ]);


  // ==========================================================
  // INITIALIZATION
  // ==========================================================

  useEffect(() => {

    mountedRef.current = true;

    initializeMediaPipe();

    return () => {

      mountedRef.current = false;

      if (
        animationFrameRef.current
      ) {

        cancelAnimationFrame(
          animationFrameRef.current
        );
      }

      if (streamRef.current) {

        streamRef.current
          .getTracks()
          .forEach(
            (track) =>
              track.stop()
          );
      }

      if (
        handLandmarkerRef.current
      ) {

        try {

          handLandmarkerRef.current.close();

        } catch (error) {

          console.warn(
            "MediaPipe cleanup:",
            error
          );
        }
      }

    };

  }, [
    initializeMediaPipe,
  ]);


  // ==========================================================
  // START EVERYTHING
  // ==========================================================

  useEffect(() => {

    if (
      !loadingMediaPipe &&
      handLandmarkerRef.current
    ) {

      startCamera();

    }

  }, [
    loadingMediaPipe,
    startCamera,
  ]);


  // ==========================================================
  // START BACKEND WHEN CAMERA IS READY
  // ==========================================================

  useEffect(() => {

    if (
      cameraActive &&
      !detectionActive
    ) {

      startBackendDetection();
    }

  }, [
    cameraActive,
    detectionActive,
    startBackendDetection,
  ]);


  // ==========================================================
  // START FRAME LOOP
  // ==========================================================

  useEffect(() => {

    if (
      cameraActive &&
      handLandmarkerRef.current
    ) {

      animationFrameRef.current =
        requestAnimationFrame(
          processFrame
        );

    }


    return () => {

      if (
        animationFrameRef.current
      ) {

        cancelAnimationFrame(
          animationFrameRef.current
        );
      }
    };

  }, [
    cameraActive,
    processFrame,
  ]);


  // ==========================================================
  // CAMERA TOGGLE
  // ==========================================================

  const handleCameraToggle =
    async () => {

      if (cameraActive) {

        await stopBackendDetection();

        stopCamera();

      } else {

        await startCamera();
      }
    };


  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div
      style={{
        width: "100%",
        maxWidth: "700px",
        margin: "0 auto",
      }}
    >

      {/* ====================================================
          CAMERA
      ==================================================== */}

      <div
        style={{
          position: "relative",
          width: "100%",
          background: "#111",
          borderRadius: "10px",
          overflow: "hidden",
          padding: "10px",
        }}
      >

        <div
          style={{
            color: "white",
            fontSize: "20px",
            marginBottom: "10px",
          }}
        >
          Webcam
        </div>


        <div
          style={{
            position: "relative",
            width: "100%",
            aspectRatio: "16 / 9",
            background: "black",
            overflow: "hidden",
            borderRadius: "6px",
          }}
        >

          {/* VIDEO */}

          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              position: "absolute",
              width: "100%",
              height: "100%",
              objectFit: "cover",

              /*
               * Mirror the user's webcam.
               */
              transform:
                "scaleX(-1)",
            }}
          />


          {/* LANDMARK CANVAS */}

          <canvas
            ref={canvasRef}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              pointerEvents: "none",

              /*
               * Mirror landmark drawing together with
               * webcam image.
               */
              transform:
                "scaleX(-1)",
            }}
          />


          {/* LOADING */}

          {loadingMediaPipe && (
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background:
                  "rgba(0,0,0,0.65)",
                color: "white",
                fontSize: "18px",
              }}
            >
              Loading hand detection...
            </div>
          )}

        </div>


        {/* ==================================================
            CAMERA BUTTON
        ================================================== */}

        <button
          onClick={
            handleCameraToggle
          }
          style={{
            marginTop: "10px",
            padding:
              "10px 20px",
            border: "none",
            borderRadius: "6px",
            background:
              cameraActive
                ? "#e63946"
                : "#28a745",
            color: "white",
            fontWeight: "bold",
            cursor: "pointer",
          }}
        >
          {cameraActive
            ? "Turn Off Camera"
            : "Turn On Camera"}
        </button>

      </div>


      {/* ====================================================
          STATUS
      ==================================================== */}

      <div
        style={{
          marginTop: "15px",
          padding: "15px",
          borderRadius: "8px",
          background: "#f5f7fa",
        }}
      >

        <div
          style={{
            fontWeight: "bold",
            marginBottom: "8px",
          }}
        >
          Detection Status
        </div>


        <div>
          Status:{" "}
          <strong>
            {status}
          </strong>
        </div>


        <div>
          Language:{" "}
          <strong>
            {language}
          </strong>
        </div>


        <div>
          Hands detected:{" "}
          <strong>
            {landmarkCount}
          </strong>
        </div>


        <div>
          Backend:{" "}
          <strong>
            {detectionActive
              ? "Connected"
              : "Inactive"}
          </strong>
        </div>

      </div>


      {/* ====================================================
          PREDICTION
      ==================================================== */}

      <div
        style={{
          marginTop: "15px",
          padding: "20px",
          borderRadius: "8px",
          background: "white",
          border:
            "1px solid #ddd",
        }}
      >

        <div
          style={{
            fontSize: "16px",
            color: "#555",
          }}
        >
          Current Character
        </div>


        <div
          style={{
            marginTop: "10px",
            fontSize: "36px",
            fontWeight: "bold",
            textAlign: "center",
            padding: "15px",
            background: "#f5f7fa",
            borderRadius: "8px",
          }}
        >
          {currentPrediction ||
            "Waiting for input..."}
        </div>


        <div
          style={{
            marginTop: "15px",
          }}
        >

          <div
            style={{
              fontWeight: "bold",
              marginBottom: "6px",
            }}
          >
            Confidence:{" "}
            {(
              confidence * 100
            ).toFixed(1)}
            %
          </div>


          <div
            style={{
              width: "100%",
              height: "10px",
              background: "#e9ecef",
              borderRadius: "10px",
              overflow: "hidden",
            }}
          >

            <div
              style={{
                width:
                  `${Math.min(
                    100,
                    Math.max(
                      0,
                      confidence * 100
                    )
                  )}%`,
                height: "100%",
                background:
                  "#28a745",
                transition:
                  "width 0.2s ease",
              }}
            />

          </div>

        </div>

      </div>


      {/* ====================================================
          ERROR
      ==================================================== */}

      {cameraError && (
        <div
          style={{
            marginTop: "15px",
            padding: "15px",
            background: "#ffe5e5",
            color: "#c62828",
            borderRadius: "8px",
            fontWeight: "500",
          }}
        >
          {cameraError}
        </div>
      )}

    </div>
  );
}


export default SignLanguageCamera;