import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
} from "react";

import Webcam from "react-webcam";

import {
  detectionBaseUrl,
} from "../Config/config";


const SignLanguageDetector = () => {

  const [language, setLanguage] =
    useState("ASL");

  const [isDetecting, setIsDetecting] =
    useState(false);

  const [status, setStatus] =
    useState(null);

  const [detectedText, setDetectedText] =
    useState("");

  const [languages, setLanguages] =
    useState([
      "ASL",
      "ISL",
      "TSL",
      "TAMIL",
    ]);

  const [modelInfo, setModelInfo] =
    useState(null);

  const [isCameraActive, setIsCameraActive] =
    useState(false);

  const [cameraError, setCameraError] =
    useState("");

  const [error, setError] =
    useState("");

  const webcamRef =
    useRef(null);

  const frameTimerRef =
    useRef(null);

  const statusTimerRef =
    useRef(null);

  const processingRef =
    useRef(false);


  // ========================================================
  // CLEAR TIMERS
  // ========================================================

  const clearIntervals = useCallback(() => {

    if (frameTimerRef.current) {

      clearInterval(
        frameTimerRef.current
      );

      frameTimerRef.current = null;

    }

    if (statusTimerRef.current) {

      clearInterval(
        statusTimerRef.current
      );

      statusTimerRef.current = null;

    }

  }, []);


  // ========================================================
  // LOAD SUPPORTED LANGUAGES
  // ========================================================

  useEffect(() => {

    const loadLanguages = async () => {

      try {

        const response =
          await fetch(
            `${detectionBaseUrl}/supported_languages`
          );

        const data =
          await response.json();

        if (
          data.supported_languages
        ) {

          setLanguages(
            data.supported_languages
          );

        }

      } catch (err) {

        console.error(
          "Could not load languages:",
          err
        );

      }

    };

    loadLanguages();

    return () => {
      clearIntervals();
    };

  }, [clearIntervals]);


  // ========================================================
  // MODEL INFO
  // ========================================================

  useEffect(() => {

    const fetchInfo = async () => {

      try {

        const response =
          await fetch(
            `${detectionBaseUrl}/model_info/${language}`
          );

        const data =
          await response.json();

        if (response.ok) {

          setModelInfo(
            data
          );

        }

      } catch (err) {

        console.error(
          "Model info error:",
          err
        );

      }

    };

    fetchInfo();

  }, [language]);


  // ========================================================
  // CAMERA
  // ========================================================

  const startCamera = async () => {

    setCameraError("");

    try {

      await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      setIsCameraActive(true);

    } catch (err) {

      console.error(
        "Camera error:",
        err
      );

      setCameraError(
        "Camera permission denied or camera unavailable."
      );

      setIsCameraActive(false);

    }

  };


  const stopCamera = async () => {

    if (isDetecting) {

      await stopDetection();

    }

    setIsCameraActive(false);

  };


  // ========================================================
  // CAPTURE FRAME
  // ========================================================

  const captureFrame = useCallback(
    async () => {

      if (!isDetecting) {
        return;
      }

      if (processingRef.current) {
        return;
      }

      if (!webcamRef.current) {
        return;
      }

      const imageSrc =
        webcamRef.current.getScreenshot();

      if (!imageSrc) {
        return;
      }

      processingRef.current = true;

      try {

        const response =
          await fetch(
            `${detectionBaseUrl}/process_frame`,
            {
              method: "POST",

              headers: {
                Accept:
                  "application/json",

                "Content-Type":
                  "application/json",
              },

              body: JSON.stringify({

                image:
                  imageSrc.split(",")[1],

                language,

              }),

            }
          );

        const data =
          await response.json();

        if (!response.ok) {

          throw new Error(
            data.message ||
            data.detail ||
            "Frame processing failed."
          );

        }

        setStatus(
          data
        );

        if (
          data.completed
        ) {

          setIsDetecting(
            false
          );

          await fetchDetectedText();

        }

      } catch (err) {

        console.error(
          "Frame error:",
          err
        );

        setError(
          err.message
        );

      } finally {

        processingRef.current =
          false;

      }

    },
    [
      isDetecting,
      language,
    ]
  );


  // ========================================================
  // FRAME TIMER
  // ========================================================

  useEffect(() => {

    if (!isDetecting) {

      if (frameTimerRef.current) {

        clearInterval(
          frameTimerRef.current
        );

        frameTimerRef.current = null;

      }

      return undefined;

    }

    frameTimerRef.current =
      setInterval(
        captureFrame,
        250
      );

    return () => {

      if (frameTimerRef.current) {

        clearInterval(
          frameTimerRef.current
        );

        frameTimerRef.current = null;

      }

    };

  }, [
    isDetecting,
    captureFrame,
  ]);


  // ========================================================
  // START DETECTION
  // ========================================================

  const startDetection = async () => {

    setError("");

    if (!isCameraActive) {

      await startCamera();

      return;

    }

    try {

      const response =
        await fetch(
          `${detectionBaseUrl}/start_detection`,
          {
            method: "POST",

            headers: {
              Accept:
                "application/json",

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
          data.message ||
          data.detail ||
          "Could not start detection."
        );

      }

      setStatus(
        data
      );

      setIsDetecting(
        true
      );

    } catch (err) {

      console.error(
        "Start detection error:",
        err
      );

      setError(
        err.message
      );

      setIsDetecting(
        false
      );

    }

  };


  // ========================================================
  // STATUS POLLING
  // ========================================================

  useEffect(() => {

    if (!isDetecting) {

      if (statusTimerRef.current) {

        clearInterval(
          statusTimerRef.current
        );

        statusTimerRef.current = null;

      }

      return undefined;

    }

    const pollStatus =
      async () => {

        try {

          const response =
            await fetch(
              `${detectionBaseUrl}/detection_status`
            );

          if (!response.ok) {
            return;
          }

          const data =
            await response.json();

          setStatus(
            data
          );

          if (
            data.completed
          ) {

            setIsDetecting(
              false
            );

            await fetchDetectedText();

          }

        } catch (err) {

          console.error(
            "Status error:",
            err
          );

        }

      };

    statusTimerRef.current =
      setInterval(
        pollStatus,
        1000
      );

    return () => {

      if (statusTimerRef.current) {

        clearInterval(
          statusTimerRef.current
        );

        statusTimerRef.current = null;

      }

    };

  }, [isDetecting]);


  // ========================================================
  // STOP DETECTION
  // ========================================================

  const stopDetection = async () => {

    clearIntervals();

    try {

      const response =
        await fetch(
          `${detectionBaseUrl}/stop_detection`,
          {
            method: "POST",

            headers: {
              Accept:
                "application/json",
            },
          }
        );

      const data =
        await response.json();

      setStatus(
        data
      );

      await fetchDetectedText();

    } catch (err) {

      console.error(
        "Stop detection error:",
        err
      );

    } finally {

      setIsDetecting(
        false
      );

    }

  };


  // ========================================================
  // FETCH TEXT
  // ========================================================

  const fetchDetectedText =
    async () => {

      try {

        const response =
          await fetch(
            `${detectionBaseUrl}/get_detected_text`
          );

        const data =
          await response.json();

        if (
          data.available
        ) {

          setDetectedText(
            data.text
          );

        }

      } catch (err) {

        console.error(
          "Detected text error:",
          err
        );

      }

    };


  // ========================================================
  // CLEAR SESSION
  // ========================================================

  const clearSession = async () => {

    clearIntervals();

    try {

      await fetch(
        `${detectionBaseUrl}/clear_session`,
        {
          method: "DELETE",
        }
      );

    } catch (err) {

      console.error(
        "Clear session error:",
        err
      );

    }

    setStatus(null);

    setDetectedText("");

    setError("");

    setIsDetecting(false);

  };


  // ========================================================
  // VIDEO
  // ========================================================

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
  };


  // ========================================================
  // RENDER
  // ========================================================

  return (

    <div
      style={{
        maxWidth: "900px",
        margin: "0 auto",
        padding: "30px",
      }}
    >

      <h1>
        Sign Language Detection
      </h1>


      {/* CAMERA */}

      <div
        style={{
          marginTop: "20px",
          background: "#111",
          padding: "15px",
          borderRadius: "12px",
          textAlign: "center",
        }}
      >

        {isCameraActive ? (

          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            screenshotQuality={0.75}
            videoConstraints={
              videoConstraints
            }
            mirrored
            style={{
              width: "100%",
              maxWidth: "640px",
              borderRadius: "8px",
            }}
          />

        ) : (

          <div
            style={{
              color: "#aaa",
              padding: "100px 20px",
            }}
          >
            Camera is off
          </div>

        )}

      </div>


      {cameraError && (

        <div
          style={{
            marginTop: "10px",
            color: "red",
          }}
        >
          {cameraError}
        </div>

      )}


      {error && (

        <div
          style={{
            marginTop: "10px",
            color: "red",
          }}
        >
          {error}
        </div>

      )}


      {/* CAMERA CONTROLS */}

      <div
        style={{
          display: "flex",
          gap: "10px",
          marginTop: "15px",
          flexWrap: "wrap",
        }}
      >

        {!isCameraActive ? (

          <button
            onClick={
              startCamera
            }
          >
            Start Camera
          </button>

        ) : (

          <button
            onClick={
              stopCamera
            }
            disabled={
              isDetecting
            }
          >
            Turn Off Camera
          </button>

        )}

      </div>


      {/* LANGUAGE */}

      <div
        style={{
          marginTop: "20px",
        }}
      >

        <label>
          Select Language:
        </label>

        <select
          value={language}
          onChange={
            (event) =>
              setLanguage(
                event.target.value
              )
          }
          disabled={
            isDetecting
          }
          style={{
            marginLeft: "10px",
            padding: "8px",
          }}
        >

          {languages.map(
            (lang) => (

              <option
                key={lang}
                value={lang}
              >
                {lang}
              </option>

            )
          )}

        </select>

      </div>


      {/* MODEL INFO */}

      {modelInfo && (

        <div
          style={{
            marginTop: "20px",
            padding: "15px",
            background:
              "#f5f5f5",
            borderRadius: "8px",
          }}
        >

          <h3>
            Model Information
          </h3>

          <p>
            Name:{" "}
            {modelInfo.name ||
              "N/A"}
          </p>

          <p>
            Type:{" "}
            {modelInfo.model_type ||
              "N/A"}
          </p>

          <p>
            Description:{" "}
            {modelInfo.description ||
              "N/A"}
          </p>

        </div>

      )}


      {/* BUTTONS */}

      <div
        style={{
          display: "flex",
          gap: "10px",
          marginTop: "20px",
          flexWrap: "wrap",
        }}
      >

        {!isDetecting ? (

          <button
            onClick={
              startDetection
            }
            disabled={
              !isCameraActive
            }
          >
            Start Detection
          </button>

        ) : (

          <button
            onClick={
              stopDetection
            }
          >
            Stop Detection
          </button>

        )}

        <button
          onClick={
            clearSession
          }
        >
          Clear Session
        </button>

      </div>


      {/* STATUS */}

      {status && (

        <div
          style={{
            marginTop: "25px",
            padding: "20px",
            background:
              "#e9f7ef",
            borderRadius: "8px",
          }}
        >

          <h3>
            Detection Status
          </h3>

          <p>
            Active:{" "}
            {status.active
              ? "Yes"
              : "No"}
          </p>

          <p>
            Last Detected:{" "}
            {status.detected_char ||
              status.last_detected_char ||
              "None"}
          </p>

          <p>
            Confidence:{" "}
            {Number(
              status.confidence ||
              0
            ).toFixed(1)}
            %
          </p>

          <p>
            Current Word:{" "}
            {status.word_buffer ||
              ""}
          </p>

          <p>
            Current Sentence:{" "}
            {status.sentence_buffer ||
              ""}
          </p>

        </div>

      )}


      {/* TEXT */}

      <div
        style={{
          marginTop: "25px",
          padding: "20px",
          background:
            "#f0f7ff",
          borderRadius: "8px",
        }}
      >

        <h3>
          Detected Text
        </h3>

        <div
          style={{
            minHeight: "80px",
            background:
              "white",
            padding: "15px",
            border:
              "1px solid #ddd",
            borderRadius: "5px",
          }}
        >
          {detectedText ||
            "No text detected yet"}
        </div>

      </div>

    </div>

  );
};


export default SignLanguageDetector;