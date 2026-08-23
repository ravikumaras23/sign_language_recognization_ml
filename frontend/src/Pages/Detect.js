
import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
} from "react";

import "../css/Detect.css";

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";

import Slider from "react-input-slider";

import "bootstrap/dist/css/bootstrap.min.css";
import "font-awesome/css/font-awesome.min.css";

import Webcam from "react-webcam";

import { detectionBaseUrl } from "../Config/config";

import xbot from "../Models/xbot/xbot.glb";
import ybot from "../Models/ybot/ybot.glb";

import xbotPic from "../Models/xbot/xbot.png";
import ybotPic from "../Models/ybot/ybot.png";

import * as words from "../Animations/words";
import * as alphabets from "../Animations/alphabets";
import { defaultPose } from "../Animations/defaultPose";


const Detect = () => {

  // =========================================================
  // DETECTION STATE
  // =========================================================

  const [isDetecting, setIsDetecting] =
    useState(false);

  const [isCameraActive, setIsCameraActive] =
    useState(false);

  const [detectedChars, setDetectedChars] =
    useState([]);

  const [currentChar, setCurrentChar] =
    useState("");

  const [language, setLanguage] =
    useState("ISL");

  const [confidence, setConfidence] =
    useState(0);

  const [sessionId, setSessionId] =
    useState("");

  const [isAutoDetection, setIsAutoDetection] =
    useState(true);

  /*
    IMPORTANT:

    inputText is now the continuous character buffer.

    Example:

    A
    AB
    ABC
    ABCD

    It will NOT be replaced every time a new
    frame is received.
  */
  const [inputText, setInputText] =
    useState("");

  const [isFinalized, setIsFinalized] =
    useState(false);

  const [animationText, setAnimationText] =
    useState("");

  const [errorMessage, setErrorMessage] =
    useState("");

  const [cameraError, setCameraError] =
    useState("");

  const [statusMessage, setStatusMessage] =
    useState("");

  const [detectionProgress, setDetectionProgress] =
    useState(0);


  // =========================================================
  // AVATAR
  // =========================================================

  const [bot, setBot] =
    useState(ybot);

  const [speed, setSpeed] =
    useState(0.1);

  const [pause, setPause] =
    useState(800);


  // =========================================================
  // REFS
  // =========================================================

  const webcamRef =
    useRef(null);

  const frameTimerRef =
    useRef(null);

  const statusTimerRef =
    useRef(null);

  const processingFrameRef =
    useRef(false);

  const mountedRef =
    useRef(true);

  const animationContainerRef =
    useRef(null);

  const animationRef =
    useRef({});


  // =========================================================
  // NEW CHARACTER STABILITY REFS
  // =========================================================

  /*
    Same character must be detected at 100%
    confidence for 3 frames within 3 seconds.
  */

  const stableCharRef =
    useRef("");

  const stableFrameCountRef =
    useRef(0);

  const stableStartTimeRef =
    useRef(0);

  /*
    Prevent the same held sign from being added
    repeatedly.

    Example:

    Hold A
    -> ABC

    It will NOT become:

    ABC
    ABCA
    ABCAA
    ...

    until the detected sign changes / confidence
    drops and another sign is detected.
  */

  const lastCommittedCharRef =
    useRef("");

  const waitingForNewSignRef =
    useRef(false);

  /*
    Prevent two fresh-frame requests at the
    same time.
  */

  const freshFrameProcessingRef =
    useRef(false);


  // =========================================================
  // DETECTION CONSTANTS
  // =========================================================

  /*
    Required confidence.

    The requirement is exactly 100%.
  */
  const REQUIRED_CONFIDENCE = 100;

  /*
    Same character must be detected for 3 frames.
  */
  const REQUIRED_STABLE_FRAMES = 3;

  /*
    Those 3 frames must occur within 3 seconds.
  */
  const STABILITY_WINDOW_MS = 3000;


  // =========================================================
  // CLEANUP
  // =========================================================

  const clearTimers = useCallback(() => {

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


  useEffect(() => {

    mountedRef.current = true;

    return () => {

      mountedRef.current = false;

      clearTimers();

      if (
        animationRef.current &&
        animationRef.current.renderer
      ) {

        animationRef.current.renderer.dispose();

      }

    };

  }, [clearTimers]);


  // =========================================================
  // RESET CHARACTER STABILITY
  // =========================================================

  const resetCharacterStability =
    useCallback(() => {

      stableCharRef.current = "";

      stableFrameCountRef.current = 0;

      stableStartTimeRef.current = 0;

    }, []);


  // =========================================================
  // CAMERA
  // =========================================================

  const handleCameraReady =
    useCallback(() => {

      console.log(
        "Browser camera ready."
      );

      setCameraError("");

      setIsCameraActive(true);

      setStatusMessage(
        "Camera ready."
      );

    }, []);


  const handleCameraError =
    useCallback(
      (error) => {

        console.error(
          "Webcam error:",
          error
        );

        setCameraError(
          "Unable to access webcam. " +
          "Please allow camera permission " +
          "and make sure no other application is using it."
        );

        setIsCameraActive(false);

      },
      []
    );


  // =========================================================
  // START CAMERA
  // =========================================================

  const startCamera = () => {

    setCameraError("");

    setErrorMessage("");

    setStatusMessage(
      "Opening camera..."
    );

    setIsCameraActive(true);
  };


  // =========================================================
  // STOP CAMERA
  // =========================================================

  const stopCamera = async () => {

    if (isDetecting) {

      await stopDetection();

    }

    setIsCameraActive(false);

    setCameraError("");

    setStatusMessage(
      "Camera stopped."
    );
  };


  // =========================================================
  // START DETECTION
  // =========================================================

  const startDetection = async () => {

    setErrorMessage("");

    setStatusMessage("");

    if (!isCameraActive) {

      setErrorMessage(
        "Please start the camera first."
      );

      return;
    }

    try {

      setStatusMessage(
        "Loading sign language model..."
      );

      setDetectedChars([]);

      setCurrentChar("");

      setInputText("");

      setIsFinalized(false);

      setConfidence(0);

      setDetectionProgress(0);

      /*
        Reset all character tracking.
      */

      resetCharacterStability();

      lastCommittedCharRef.current = "";

      waitingForNewSignRef.current = false;

      freshFrameProcessingRef.current = false;

      const response = await fetch(
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

      console.log(
        "Detection started:",
        data
      );

      if (!mountedRef.current) {
        return;
      }

      setSessionId(
        data.session_id || ""
      );

      setIsDetecting(true);

      setStatusMessage(
        "Detection running. Hold your sign steadily."
      );

    } catch (error) {

      console.error(
        "Start detection error:",
        error
      );

      setIsDetecting(false);

      setErrorMessage(
        error.message ||
        "Could not start detection."
      );

    }
  };


  // =========================================================
  // APPEND CONFIRMED CHARACTER
  // =========================================================

  const appendConfirmedCharacter =
    useCallback(
      (character) => {

        if (!mountedRef.current) {
          return;
        }

        const normalizedChar =
          String(character)
            .trim()
            .toUpperCase();

        if (!normalizedChar) {
          return;
        }

        /*
          Do not append the same character
          repeatedly while the user is holding
          the same sign.
        */

        if (
          waitingForNewSignRef.current &&
          lastCommittedCharRef.current ===
            normalizedChar
        ) {

          return;
        }

        /*
          Add the new character to the existing
          input text.

          Example:

          A -> AB -> ABC -> ABCD
        */

        setInputText(
          (previousText) => {

            const newText =
              `${previousText}${normalizedChar}`;

            return newText;
          }
        );

        /*
          Keep the latest confirmed character
          visible in Predicted Sign.
        */

        setCurrentChar(
          normalizedChar
        );

        /*
          Store the committed character.
        */

        lastCommittedCharRef.current =
          normalizedChar;

        /*
          Require a new sign before this
          character can be committed again.
        */

        waitingForNewSignRef.current =
          true;

        /*
          Reset stability so the next character
          needs another 3 frames.
        */

        resetCharacterStability();

        setDetectionProgress(0);

        setStatusMessage(
          `Character "${normalizedChar}" confirmed. Show the next sign.`
        );

        console.log(
          "Confirmed character:",
          normalizedChar
        );

      },
      [
        resetCharacterStability,
      ]
    );


  // =========================================================
  // PROCESS FRESH FRAME AFTER 3 STABLE FRAMES
  // =========================================================

  const processFreshFrame =
    useCallback(
      async () => {

        if (!mountedRef.current) {
          return;
        }

        if (!isDetecting) {
          return;
        }

        if (
          freshFrameProcessingRef.current
        ) {
          return;
        }

        if (!webcamRef.current) {
          return;
        }

        const video =
          webcamRef.current.video;

        if (!video) {
          return;
        }

        if (video.readyState < 2) {
          return;
        }

        const freshImage =
          webcamRef.current.getScreenshot();

        if (!freshImage) {
          return;
        }

        freshFrameProcessingRef.current =
          true;

        try {

          setStatusMessage(
            "3 frames confirmed. Capturing fresh frame..."
          );

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
                    freshImage,

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
              "Fresh frame processing failed."
            );

          }

          if (!mountedRef.current) {
            return;
          }

          const freshChar =
            data.detected_char
              ? String(
                  data.detected_char
                )
              : "";

          const freshConfidence =
            Number(
              data.confidence || 0
            );

          setConfidence(
            freshConfidence
          );

          /*
            The fresh frame must also identify
            the character correctly.

            We accept the fresh frame when it
            matches the stable character and has
            100% confidence.
          */

          if (
            freshChar &&
            freshChar !== "?" &&
            freshConfidence >=
              REQUIRED_CONFIDENCE &&
            freshChar.toUpperCase() ===
              stableCharRef.current.toUpperCase()
          ) {

            const confirmedCharacter =
              freshChar.toUpperCase();

            appendConfirmedCharacter(
              confirmedCharacter
            );

          } else {

            /*
              Fresh frame changed or confidence
              dropped.

              Start detection again.
            */

            resetCharacterStability();

            waitingForNewSignRef.current =
              false;

            setDetectionProgress(0);

            setStatusMessage(
              "Fresh frame changed. Please hold the next sign steadily."
            );

          }

        } catch (error) {

          console.error(
            "Fresh frame error:",
            error
          );

          /*
            Do not stop the entire detection
            session because the fresh frame failed.
          */

          resetCharacterStability();

          setDetectionProgress(0);

          setStatusMessage(
            "Fresh frame failed. Continuing detection..."
          );

        } finally {

          freshFrameProcessingRef.current =
            false;

        }

      },
      [
        isDetecting,
        language,
        appendConfirmedCharacter,
        resetCharacterStability,
      ]
    );


  // =========================================================
  // PROCESS CHARACTER STABILITY
  // =========================================================

  const processCharacterStability =
    useCallback(
      (detectedChar, currentConfidence) => {

        if (!detectedChar) {
          return;
        }

        const character =
          String(
            detectedChar
          )
            .trim()
            .toUpperCase();

        if (
          !character ||
          character === "?"
        ) {
          return;
        }

        /*
          If confidence is below 100%,
          this is not a valid confirmation frame.
        */

        if (
          currentConfidence <
          REQUIRED_CONFIDENCE
        ) {

          /*
            Confidence dropped.

            The user can now show a new sign.
          */

          resetCharacterStability();

          waitingForNewSignRef.current =
            false;

          setDetectionProgress(0);

          setStatusMessage(
            "Hold the sign until confidence reaches 100%."
          );

          return;
        }

        /*
          If we are waiting for a new sign
          and the current character is still
          the previous committed character,
          don't count it again.
        */

        if (
          waitingForNewSignRef.current &&
          lastCommittedCharRef.current ===
            character
        ) {

          setStatusMessage(
            `Sign "${character}" is still being held. Show a new sign.`
          );

          return;
        }

        const now =
          Date.now();

        /*
          FIRST 100% FRAME
        */

        if (
          stableCharRef.current !==
          character
        ) {

          stableCharRef.current =
            character;

          stableFrameCountRef.current =
            1;

          stableStartTimeRef.current =
            now;

          setDetectionProgress(
            1 /
              REQUIRED_STABLE_FRAMES
          );

          setStatusMessage(
            `Detected "${character}" — frame 1/${REQUIRED_STABLE_FRAMES}.`
          );

          return;
        }

        /*
          Check whether the 3-second window
          has expired.
        */

        const elapsed =
          now -
          stableStartTimeRef.current;

        if (
          elapsed >
          STABILITY_WINDOW_MS
        ) {

          /*
            Restart the 3-frame sequence.
          */

          stableFrameCountRef.current =
            1;

          stableStartTimeRef.current =
            now;

          setDetectionProgress(
            1 /
              REQUIRED_STABLE_FRAMES
          );

          setStatusMessage(
            `Detection window expired. "${character}" frame 1/${REQUIRED_STABLE_FRAMES}.`
          );

          return;
        }

        /*
          SAME CHARACTER + 100% CONFIDENCE
        */

        stableFrameCountRef.current +=
          1;

        const progress =
          Math.min(
            stableFrameCountRef.current /
              REQUIRED_STABLE_FRAMES,
            1
          );

        setDetectionProgress(
          progress
        );

        setStatusMessage(
          `"${character}" confirmed at 100% — frame ${Math.min(
            stableFrameCountRef.current,
            REQUIRED_STABLE_FRAMES
          )}/${REQUIRED_STABLE_FRAMES}.`
        );

        /*
          3 successful frames reached.
        */

        if (
          stableFrameCountRef.current >=
          REQUIRED_STABLE_FRAMES
        ) {

          /*
            IMPORTANT:

            We do NOT immediately append the
            character.

            First take one completely fresh frame.
          */

          processFreshFrame();

        }

      },
      [
        resetCharacterStability,
        processFreshFrame,
      ]
    );


  // =========================================================
  // CAPTURE FRAME
  // =========================================================

  const captureFrame =
    useCallback(
      async () => {

        if (!mountedRef.current) {
          return;
        }

        if (!isDetecting) {
          return;
        }

        if (processingFrameRef.current) {
          return;
        }

        if (
          freshFrameProcessingRef.current
        ) {
          return;
        }

        if (!webcamRef.current) {
          return;
        }

        const video =
          webcamRef.current.video;

        if (!video) {
          return;
        }

        if (
          video.readyState < 2
        ) {
          return;
        }

        const imageSrc =
          webcamRef.current.getScreenshot();

        if (!imageSrc) {
          return;
        }

        processingFrameRef.current =
          true;

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
                    imageSrc,

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

          if (!mountedRef.current) {
            return;
          }


          // =================================================
          // PREDICTED CHARACTER
          // =================================================

          const detectedChar =
            data.detected_char
              ? String(
                  data.detected_char
                )
              : "";

          const frameConfidence =
            Number(
              data.confidence || 0
            );

          /*
            Current predicted sign is shown
            immediately in the Predicted Sign
            display.

            BUT it is only added to Current
            Prediction after 3 stable 100%
            frames + fresh frame.
          */

          if (
            detectedChar &&
            detectedChar !== "?"
          ) {

            setCurrentChar(
              detectedChar
            );

          }


          // =================================================
          // CONFIDENCE
          // =================================================

          setConfidence(
            frameConfidence
          );


          // =================================================
          // PROCESS 3-FRAME STABILITY
          // =================================================

          if (
            detectedChar &&
            detectedChar !== "?"
          ) {

            processCharacterStability(
              detectedChar,
              frameConfidence
            );

          } else {

            /*
              No valid character detected.
            */

            resetCharacterStability();

            waitingForNewSignRef.current =
              false;

            setDetectionProgress(0);

            setStatusMessage(
              "Analyzing hand position..."
            );

          }


          // =================================================
          // AUTO DETECTION
          // =================================================

          if (
            typeof data.should_auto_detect !==
            "undefined"
          ) {

            setIsAutoDetection(
              data.should_auto_detect !==
                false
            );

          }


          // =================================================
          // DO NOT OVERWRITE inputText
          // =================================================

          /*
            IMPORTANT:

            We intentionally do NOT do this:

              setInputText(data.word_buffer)

            because that would overwrite our
            continuous frontend character buffer.

            The Current Prediction box is now
            controlled by confirmed characters.
          */


          // =================================================
          // SENTENCE
          // =================================================

          if (
            typeof data.sentence_buffer ===
            "string"
          ) {

            const sentence =
              data.sentence_buffer.trim();

            if (sentence) {

              setDetectedChars(
                sentence
                  .split(/\s+/)
                  .filter(Boolean)
              );

            }

          }


        } catch (error) {

          console.error(
            "Frame processing error:",
            error
          );

          /*
            Do not stop detection because
            of one failed frame.
          */

        } finally {

          processingFrameRef.current =
            false;
        }

      },
      [
        isDetecting,
        language,
        processCharacterStability,
        resetCharacterStability,
      ]
    );


  // =========================================================
  // FRAME LOOP
  // =========================================================

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

    /*
      4 frames per second.

      The 3 qualifying frames therefore normally
      occur within approximately 750ms, while the
      3-second window protects against slow frames.
    */

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


  // =========================================================
  // STATUS POLLING
  // =========================================================

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

    const fetchStatus =
      async () => {

        try {

          const response =
            await fetch(
              `${detectionBaseUrl}/detection_status`
            );

          if (!response.ok) {

            console.warn(
              "Status endpoint:",
              response.status
            );

            return;
          }

          const data =
            await response.json();

          if (!mountedRef.current) {
            return;
          }


          // ===============================================
          // STATUS INFORMATION ONLY
          // ===============================================

          setConfidence(
            Number(
              data.confidence || 0
            )
          );

          setSessionId(
            data.session_id || ""
          );

          setIsAutoDetection(
            data.auto_detection_enabled !==
              false
          );


          /*
            IMPORTANT:

            Do not update inputText from
            word_buffer here.

            Otherwise the backend status poll
            could overwrite the continuous
            frontend character buffer.
          */


          // ===============================================
          // SENTENCE
          // ===============================================

          if (
            typeof data.sentence_buffer ===
            "string"
          ) {

            const sentence =
              data.sentence_buffer.trim();

            if (sentence) {

              setDetectedChars(
                sentence
                  .split(/\s+/)
                  .filter(Boolean)
              );

            }

          }


          // ===============================================
          // FINAL SENTENCE
          // ===============================================

          if (
            data.final_sentence
          ) {

            const finalSentence =
              String(
                data.final_sentence
              ).trim();

            if (finalSentence) {

              setInputText(
                finalSentence
              );

              setDetectedChars(
                finalSentence
                  .split(/\s+/)
                  .filter(Boolean)
              );

              setIsFinalized(true);

            }

          }

        } catch (error) {

          console.error(
            "Status polling error:",
            error
          );

        }

      };


    fetchStatus();

    statusTimerRef.current =
      setInterval(
        fetchStatus,
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


  // =========================================================
  // STOP DETECTION
  // =========================================================

  const stopDetection =
    async () => {

      clearTimers();

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

        console.log(
          "Detection stopped:",
          data
        );


        if (
          data.final_sentence
        ) {

          const finalSentence =
            String(
              data.final_sentence
            ).trim();

          if (finalSentence) {

            setInputText(
              finalSentence
            );

            setDetectedChars(
              finalSentence
                .split(/\s+/)
                .filter(Boolean)
            );

            setIsFinalized(true);

          }

        }

      } catch (error) {

        console.error(
          "Stop detection error:",
          error
        );

      } finally {

        setIsDetecting(false);

        setCurrentChar("");

        setDetectionProgress(0);

        resetCharacterStability();

        lastCommittedCharRef.current =
          "";

        waitingForNewSignRef.current =
          false;

        freshFrameProcessingRef.current =
          false;

        setStatusMessage(
          "Detection stopped."
        );

      }

    };


  // =========================================================
  // ENTER
  // =========================================================

  const handleEnter =
    async () => {

      if (!inputText.trim()) {
        return;
      }

      try {

        const response =
          await fetch(
            `${detectionBaseUrl}/enter`,
            {
              method: "POST",

              headers: {
                Accept:
                  "application/json",

                "Content-Type":
                  "application/json",
              },
            }
          );

        const data =
          await response.json();

        if (!response.ok) {

          throw new Error(
            data.message ||
            "Could not add word."
          );

        }

        if (
          data.sentence
        ) {

          setDetectedChars(
            data.sentence
              .split(/\s+/)
              .filter(Boolean)
          );

        }

        /*
          Clear only the current word.

          The finalized sentence stays in
          Detected Text.
        */

        setInputText("");

        setCurrentChar("");

        resetCharacterStability();

        lastCommittedCharRef.current =
          "";

        waitingForNewSignRef.current =
          false;

        setDetectionProgress(0);

      } catch (error) {

        console.error(
          "Enter error:",
          error
        );

      }

    };


  // =========================================================
  // BACKSPACE
  // =========================================================

  const handleBackspace =
    async () => {

      try {

        /*
          First remove the last character
          locally so the Current Prediction
          box behaves immediately.
        */

        setInputText(
          (previousText) =>
            previousText.slice(
              0,
              -1
            )
        );

        setCurrentChar("");

        resetCharacterStability();

        lastCommittedCharRef.current =
          "";

        waitingForNewSignRef.current =
          false;

        /*
          Synchronize with backend.
        */

        const response =
          await fetch(
            `${detectionBaseUrl}/backspace`,
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

        /*
          Only use backend word_buffer if
          it exists and is not empty.

          This avoids unexpectedly replacing
          our local continuous buffer.
        */

        if (
          typeof data.word_buffer ===
            "string" &&
          data.word_buffer.length > 0
        ) {

          setInputText(
            data.word_buffer
          );

        }

      } catch (error) {

        console.error(
          "Backspace error:",
          error
        );

      }

    };


  // =========================================================
  // CLEAR
  // =========================================================

  const clearSession =
    async () => {

      try {

        clearTimers();

        await fetch(
          `${detectionBaseUrl}/clear_session`,
          {
            method: "DELETE",

            headers: {
              Accept:
                "application/json",
            },
          }
        );

      } catch (error) {

        console.error(
          "Clear session error:",
          error
        );

      } finally {

        setIsDetecting(false);

        setDetectedChars([]);

        setCurrentChar("");

        setInputText("");

        setConfidence(0);

        setSessionId("");

        setDetectionProgress(0);

        setIsFinalized(false);

        resetCharacterStability();

        lastCommittedCharRef.current =
          "";

        waitingForNewSignRef.current =
          false;

        freshFrameProcessingRef.current =
          false;

        setStatusMessage(
          "Session cleared."
        );

      }

    };


  // =========================================================
  // INPUT
  // =========================================================

  const handleInputChange =
    (event) => {

      setInputText(
        event.target.value
      );

      /*
        Manual input should reset automatic
        duplicate protection.
      */

      resetCharacterStability();

      lastCommittedCharRef.current =
        "";

      waitingForNewSignRef.current =
        false;

    };


  // =========================================================
  // AVATAR INITIALIZATION
  // =========================================================

  useEffect(() => {

    const container =
      animationContainerRef.current;

    if (!container) {
      return undefined;
    }

    const animation =
      animationRef.current;

    animation.flag = false;

    animation.pending = false;

    animation.animations = [];

    animation.characters = [];

    animation.scene =
      new THREE.Scene();

    animation.scene.background =
      new THREE.Color(
        0xdddddd
      );

    const light =
      new THREE.SpotLight(
        0xffffff,
        2
      );

    light.position.set(
      0,
      5,
      5
    );

    animation.scene.add(
      light
    );

    animation.renderer =
      new THREE.WebGLRenderer({
        antialias: true,
      });

    animation.renderer.setSize(
      400,
      300
    );

    container.innerHTML = "";

    container.appendChild(
      animation.renderer.domElement
    );

    animation.camera =
      new THREE.PerspectiveCamera(
        40,
        400 / 300,
        0.1,
        1000
      );

    animation.camera.position.z =
      1.6;

    animation.camera.position.y =
      1.4;

    const loader =
      new GLTFLoader();

    loader.load(

      bot,

      (gltf) => {

        if (!mountedRef.current) {
          return;
        }

        gltf.scene.traverse(
          (child) => {

            if (
              child.type ===
              "SkinnedMesh"
            ) {

              child.frustumCulled =
                false;

            }

          }
        );

        animation.avatar =
          gltf.scene;

        animation.avatar.scale.set(
          0.8,
          0.8,
          0.8
        );

        animation.scene.add(
          animation.avatar
        );

        defaultPose(
          animation
        );

        animation.renderer.render(
          animation.scene,
          animation.camera
        );

      },

      undefined,

      (error) => {

        console.error(
          "Avatar loading error:",
          error
        );

      }
    );

    return () => {

      if (
        animation.renderer
      ) {

        animation.renderer.dispose();

      }

      container.innerHTML = "";

    };

  }, [bot]);


  // =========================================================
  // AVATAR ANIMATION
  // =========================================================

  const animateAvatar =
    useCallback(() => {

      const animation =
        animationRef.current;

      if (!animation.animations) {
        return;
      }

      if (
        !animation.avatar ||
        !animation.renderer
      ) {
        return;
      }

      if (
        animation.animations.length === 0
      ) {

        animation.pending = false;

        animation.renderer.render(
          animation.scene,
          animation.camera
        );

        return;
      }

      requestAnimationFrame(
        animateAvatar
      );

      if (
        animation.animations[0].length
      ) {

        if (!animation.flag) {

          if (
            animation.animations[0][0] ===
            "add-text"
          ) {

            animation.animations.shift();

          } else {

            for (
              let i = 0;
              i <
              animation.animations[0].length;
            ) {

              const [
                boneName,
                action,
                axis,
                limit,
                signValue,
              ] =
                animation.animations[0][i];

              const bone =
                animation.avatar
                  .getObjectByName(
                    boneName
                  );

              if (!bone) {

                animation.animations[0]
                  .splice(i, 1);

                continue;
              }

              if (
                signValue === "+" &&
                bone[action][axis] <
                  limit
              ) {

                bone[action][axis] +=
                  speed;

                bone[action][axis] =
                  Math.min(
                    bone[action][axis],
                    limit
                  );

                i++;

              } else if (
                signValue === "-" &&
                bone[action][axis] >
                  limit
              ) {

                bone[action][axis] -=
                  speed;

                bone[action][axis] =
                  Math.max(
                    bone[action][axis],
                    limit
                  );

                i++;

              } else {

                animation.animations[0]
                  .splice(i, 1);

              }

            }

          }

        }

      } else {

        animation.flag = true;

        setTimeout(() => {

          animation.flag = false;

        }, pause);

        animation.animations.shift();

      }

      animation.renderer.render(
        animation.scene,
        animation.camera
      );

    }, [
      pause,
      speed,
    ]);


  // =========================================================
  // SIGN ANIMATION
  // =========================================================

  const sign =
    useCallback(
      (text) => {

        const animation =
          animationRef.current;

        if (
          !animation ||
          !animation.animations
        ) {
          return;
        }

        if (!animation.avatar) {
          return;
        }

        const str =
          text.toUpperCase();

        const strWords =
          str.split(" ");

        for (
          const word of strWords
        ) {

          if (!word) {
            continue;
          }

          if (words[word]) {

            animation.animations.push([
              "add-text",
              word + " ",
            ]);

            words[word](
              animation
            );

          } else {

            for (
              const [
                index,
                character
              ] of word
                .split("")
                .entries()
            ) {

              if (
                !alphabets[character]
              ) {
                continue;
              }

              animation.animations.push([
                "add-text",
                index ===
                  word.length - 1
                  ? character + " "
                  : character,
              ]);

              alphabets[character](
                animation
              );

            }

          }

        }

        if (!animation.pending) {

          animation.pending = true;

          animateAvatar();

        }

      },
      [animateAvatar]
    );


  // =========================================================
  // ANIMATION BUTTONS
  // =========================================================

  const startAnimation =
    () => {

      const text =
        detectedChars.join(" ") ||
        inputText;

      if (!text.trim()) {
        return;
      }

      setAnimationText(
        text
      );

      sign(text);

    };


  const startAnimationFromInput =
    () => {

      if (!inputText.trim()) {
        return;
      }

      setAnimationText(
        inputText
      );

      sign(
        inputText
      );

    };


  // =========================================================
  // VIDEO
  // =========================================================

  const videoConstraints = {

    width: {
      ideal: 640,
    },

    height: {
      ideal: 480,
    },

    facingMode: "user",

  };


  // =========================================================
  // RENDER
  // =========================================================

  return (

    <div className="detect-container">

      <div className="detect-header">

        <h1>
          Sign Language Detection System
        </h1>

        <p className="subtitle">
          Real-time character recognition
        </p>

      </div>


      {/* =====================================================
          CAMERA
      ===================================================== */}

      <div
        style={{
          width: "100%",
          maxWidth: "700px",
          margin: "0 auto 25px",
          padding: "15px",
          background: "#111",
          borderRadius: "12px",
        }}
      >

        <h3
          style={{
            color: "white",
            marginBottom: "15px",
          }}
        >
          Webcam
        </h3>


        <div
          style={{
            width: "100%",
            minHeight: "360px",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            background: "#222",
            borderRadius: "8px",
            overflow: "hidden",
            position: "relative",
          }}
        >

          {isCameraActive ? (

            <Webcam

              ref={webcamRef}

              audio={false}

              screenshotFormat="image/jpeg"

              screenshotQuality={0.8}

              videoConstraints={
                videoConstraints
              }

              onUserMedia={
                handleCameraReady
              }

              onUserMediaError={
                handleCameraError
              }

              mirrored={true}

              forceScreenshotSourceSize={true}

              style={{
                width: "100%",
                maxWidth: "640px",
                display: "block",
              }}

            />

          ) : (

            <div
              style={{
                color: "#aaa",
                padding: "60px 20px",
                textAlign: "center",
              }}
            >

              <div
                style={{
                  fontSize: "50px",
                  marginBottom: "15px",
                }}
              >
                📷
              </div>

              <div>
                Camera is off
              </div>

              <div
                style={{
                  fontSize: "13px",
                  marginTop: "8px",
                }}
              >
                Click "Start Camera"
              </div>

            </div>

          )}


          {/* DETECTION OVERLAY */}

          {isDetecting && (

            <div
              style={{
                position: "absolute",
                top: "15px",
                left: "15px",
                right: "15px",
                display: "flex",
                justifyContent:
                  "space-between",
                pointerEvents: "none",
              }}
            >

              <span
                style={{
                  background:
                    "rgba(198,40,40,.9)",
                  color: "white",
                  padding:
                    "6px 12px",
                  borderRadius:
                    "20px",
                  fontSize:
                    "13px",
                  fontWeight:
                    "bold",
                }}
              >
                ● DETECTING
              </span>

              <span
                style={{
                  background:
                    "rgba(0,0,0,.7)",
                  color: "white",
                  padding:
                    "6px 12px",
                  borderRadius:
                    "20px",
                  fontSize:
                    "13px",
                }}
              >

                {Number(
                  confidence
                ).toFixed(1)}
                %

              </span>

            </div>

          )}

        </div>


        {cameraError && (

          <div
            style={{
              marginTop: "12px",
              padding: "12px",
              background: "#ffebee",
              color: "#c62828",
              borderRadius: "6px",
            }}
          >
            {cameraError}
          </div>

        )}


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
              className="action-button start-button"
              onClick={
                startCamera
              }
            >
              <i className="fas fa-camera"></i>{" "}
              Start Camera
            </button>

          ) : (

            <button
              className="action-button stop-button"
              onClick={
                stopCamera
              }
              disabled={
                isDetecting
              }
            >
              <i className="fas fa-camera"></i>{" "}
              Turn Off Camera
            </button>

          )}

        </div>

      </div>


      {/* =====================================================
          MAIN
      ===================================================== */}

      <div className="detect-content">

        <div className="left-column">


          {/* STATUS */}

          <div className="status-panel">

            <div className="status-card">

              <h3>
                Detection Status
              </h3>

              <div
                className="status-indicator"
              >

                <span
                  className={
                    `status-dot ${
                      isDetecting
                        ? "active"
                        : ""
                    }`
                  }
                />

                <span>
                  {isDetecting
                    ? "ACTIVE"
                    : "INACTIVE"}
                </span>

              </div>

              <div className="status-details">

                <p>
                  <strong>
                    Language:
                  </strong>{" "}
                  {language}
                </p>

                <p>
                  <strong>
                    Session ID:
                  </strong>{" "}
                  {sessionId ||
                    "Not started"}
                </p>

                <p>
                  <strong>
                    Auto Detection:
                  </strong>{" "}
                  {isAutoDetection
                    ? "ON"
                    : "OFF"}
                </p>

                {statusMessage && (

                  <p>
                    <strong>
                      Status:
                    </strong>{" "}
                    {statusMessage}
                  </p>

                )}

              </div>

            </div>


            {/* CURRENT CHARACTER */}

            <div className="current-character">

              <h3>
                Predicted Sign
              </h3>

              <div className="character-display">

                {currentChar ? (

                  <span
                    style={{
                      fontSize:
                        "72px",
                      fontWeight:
                        "bold",
                    }}
                  >
                    {currentChar}
                  </span>

                ) : (

                  <span
                    className="placeholder-char"
                  >
                    Waiting for input...
                  </span>

                )}

              </div>


              <div className="confidence-meter">

                <div className="confidence-label">

                  Confidence:{" "}

                  {Number(
                    confidence
                  ).toFixed(1)}

                  %

                </div>

                <div className="confidence-bar">

                  <div
                    className="confidence-fill"
                    style={{
                      width:
                        `${Math.min(
                          100,
                          Math.max(
                            0,
                            confidence
                          )
                        )}%`,
                    }}
                  />

                </div>

              </div>


              <div
                style={{
                  marginTop: "15px",
                }}
              >

                <small>
                  Detection progress
                </small>

                <div
                  style={{
                    height: "8px",
                    background: "#ddd",
                    borderRadius: "5px",
                    overflow: "hidden",
                    marginTop: "5px",
                  }}
                >

                  <div
                    style={{
                      height: "100%",
                      width:
                        `${Math.min(
                          1,
                          Math.max(
                            0,
                            detectionProgress
                          )
                        ) * 100}%`,
                      background:
                        "#2196f3",
                      transition:
                        "width 0.2s",
                    }}
                  />

                </div>

              </div>

            </div>

          </div>


          {/* ERROR */}

          {errorMessage && (

            <div
              style={{
                background:
                  "#ffebee",
                color:
                  "#c62828",
                padding:
                  "15px",
                borderRadius:
                  "8px",
                marginBottom:
                  "15px",
              }}
            >

              <strong>
                Error:
              </strong>{" "}

              {errorMessage}

            </div>

          )}


          {/* INPUT */}

          <div className="input-section">

            <h3>
              Current Prediction
            </h3>

            <div
              className="input-control-group"
            >

              <div
                className="input-container"
              >

                <input
                  type="text"
                  value={
                    inputText
                  }
                  onChange={
                    handleInputChange
                  }
                  placeholder={
                    "Confirmed characters will appear here..."
                  }
                  className="text-input"
                />

              </div>


              {isDetecting && (

                <div
                  className="input-action-buttons"
                >

                  <button
                    className="input-action-button backspace-button"
                    onClick={
                      handleBackspace
                    }
                  >
                    <i className="fas fa-backspace"></i>{" "}
                    Backspace
                  </button>

                  <button
                    className="input-action-button enter-button"
                    onClick={
                      handleEnter
                    }
                  >
                    <i className="fas fa-arrow-right"></i>{" "}
                    Enter
                  </button>

                </div>

              )}

            </div>


            <div
              className="input-instructions"
            >

              <p>
                <i className="fas fa-info-circle"></i>{" "}
                Hold the same sign until 100%
                confidence is reached for 3
                frames within 3 seconds.
              </p>

            </div>


            <button
              onClick={
                startAnimationFromInput
              }
              className="btn btn-primary w-100 btn-style btn-start"
              disabled={
                !inputText.trim()
              }
            >
              Animate Current Input
            </button>

          </div>


          {/* DETECTED TEXT */}

          <div className="text-output">

            <h3>
              Detected Text
            </h3>

            <div className="text-display">

              {detectedChars.length > 0 ? (

                <div
                  className="detected-words"
                >

                  {detectedChars.map(
                    (
                      word,
                      index
                    ) => (

                      <span
                        key={index}
                        className="detected-word"
                      >
                        {word}
                      </span>

                    )
                  )}

                </div>

              ) : (

                <p
                  className="placeholder-text"
                >
                  Your finalized text
                  will appear here
                </p>

              )}

            </div>


            <button
              onClick={
                startAnimation
              }
              className="btn btn-primary w-100 btn-style btn-start"
              disabled={
                detectedChars.length ===
                0
              }
            >
              Animate Detected Text
            </button>

          </div>

        </div>


        {/* ===================================================
            AVATAR
        =================================================== */}

        <div className="right-column">

          <div className="animation-section">

            <h3>
              Avatar Animation
            </h3>

            <div className="animation-info">

              Currently animating:{" "}

              {animationText ||
                "Nothing yet"}

            </div>

            <div
              ref={
                animationContainerRef
              }
              id="animation-canvas"
              className="animation-canvas"
            />


            <div
              className="animation-controls"
            >

              <div
                className="avatar-selection"
              >

                <h4>
                  Select Avatar:
                </h4>

                <div
                  className="avatar-images"
                >

                  <img
                    src={xbotPic}
                    className={
                      `bot-image ${
                        bot === xbot
                          ? "selected"
                          : ""
                      }`
                    }
                    onClick={() =>
                      setBot(xbot)
                    }
                    alt="XBOT"
                  />

                  <img
                    src={ybotPic}
                    className={
                      `bot-image ${
                        bot === ybot
                          ? "selected"
                          : ""
                      }`
                    }
                    onClick={() =>
                      setBot(ybot)
                    }
                    alt="YBOT"
                  />

                </div>

              </div>


              <div
                className="slider-controls"
              >

                <div
                  className="slider-group"
                >

                  <label>
                    Animation Speed:{" "}
                    {Math.round(
                      speed * 100
                    ) / 100}
                  </label>

                  <Slider
                    axis="x"
                    xmin={0.05}
                    xmax={0.5}
                    xstep={0.01}
                    x={speed}
                    onChange={
                      ({ x }) =>
                        setSpeed(x)
                    }
                    className="w-100"
                  />

                </div>


                <div
                  className="slider-group"
                >

                  <label>
                    Pause time:{" "}
                    {pause} ms
                  </label>

                  <Slider
                    axis="x"
                    xmin={0}
                    xmax={2000}
                    xstep={100}
                    x={pause}
                    onChange={
                      ({ x }) =>
                        setPause(x)
                    }
                    className="w-100"
                  />

                </div>

              </div>

            </div>

          </div>

        </div>


        {/* ===================================================
            CONTROLS
        =================================================== */}

        <div
          className="control-panel"
        >

          <div
            className="language-selector"
          >

            <label
              htmlFor="language"
            >
              Detection Language:
            </label>

            <select
              id="language"
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
            >

              <option value="ISL">
                ISL
              </option>

              <option value="ASL">
                ASL
              </option>

              <option value="TSL">
                TSL
              </option>

              <option value="TAMIL">
                Tamil
              </option>

            </select>

          </div>


          <div
            className="action-buttons"
          >

            {!isDetecting ? (

              <button
                className="action-button start-button"
                onClick={
                  startDetection
                }
                disabled={
                  !isCameraActive
                }
              >

                <i className="fas fa-play"></i>{" "}
                Start Detection

              </button>

            ) : (

              <button
                className="action-button stop-button"
                onClick={
                  stopDetection
                }
              >

                <i className="fas fa-stop"></i>{" "}
                Stop Detection

              </button>

            )}


            <button
              className="action-button"
              onClick={
                clearSession
              }
            >

              <i className="fas fa-trash"></i>{" "}
              Clear Session

            </button>

          </div>

        </div>

      </div>

    </div>

  );

};


export default Detect;
