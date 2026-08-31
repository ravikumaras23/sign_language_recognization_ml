# ============================================================
# api.py
# Unified Browser-Based Sign Language Detection API
#
# Browser webcam
#       |
#       v
# React Detect.js
#       |
#       | POST /process_frame
#       v
# FastAPI / Render
#       |
#       v
# MediaPipe Hands
#       |
#       +--> ASL Random Forest
#       +--> ISL LSTM
#       +--> TSL LSTM
#
# IMPORTANT:
# This backend NEVER opens a local webcam.
# ============================================================

import os
import base64
import binascii
import threading
from datetime import datetime
from typing import Optional, Any

import cv2
import numpy as np

from fastapi import (
    FastAPI,
    HTTPException,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models import (
    UnifiedSignLanguageDetector,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Unified Sign Language Detection API",
    version="4.0.0",
)


# ============================================================
# CORS
# ============================================================

FRONTEND_ORIGIN = (
    "https://sign-language-frontend-onnp.onrender.com"
)

LOCAL_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]

ALLOWED_ORIGINS = [
    FRONTEND_ORIGIN,
    *LOCAL_ORIGINS,
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DETECTED_TEXT_FILE = os.path.join(
    BASE_DIR,
    "detected_text.txt",
)

DETECTION_LOG_FILE = os.path.join(
    BASE_DIR,
    "sign_language_detections.txt",
)


# ============================================================
# GLOBAL DETECTOR
# ============================================================

detector = UnifiedSignLanguageDetector()


# Prevent two browser requests from processing the same
# MediaPipe/model object simultaneously.
detector_lock = threading.RLock()


# ============================================================
# SESSION STATE
# ============================================================

detection_active = False

detection_status = {
    "active": False,
    "language": None,
    "word_buffer": "",
    "sentence_buffer": "",
    "last_detected_char": "?",
    "confidence": 0.0,
    "session_id": None,
    "completed": False,
    "final_sentence": "",
    "detection_progress": 0.0,
    "auto_detection_enabled": True,
}


# ============================================================
# REQUEST MODELS
# ============================================================

class DetectionRequest(BaseModel):

    language: str = "ASL"


class FrameRequest(BaseModel):

    image: Optional[str] = None

    landmarks: Optional[Any] = None

    language: str = "ASL"


# ============================================================
# LANGUAGE
# ============================================================

def normalize_language(
    language: str,
) -> str:

    value = (
        str(language)
        .upper()
        .strip()
    )

    if value in (
        "TAMIL",
        "TAMIL_SIGN_LANGUAGE",
    ):

        return "TSL"

    if value not in (
        "ASL",
        "ISL",
        "TSL",
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Language must be "
                "ASL, ISL, TSL, or TAMIL"
            ),
        )

    return value


# ============================================================
# LOGGING
# ============================================================

def log_detection(
    message: str,
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:

        with open(
            DETECTION_LOG_FILE,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                f"[{timestamp}] {message}\n"
            )

    except Exception as exc:

        print(
            "Logging error:",
            repr(exc),
        )


# ============================================================
# SAVE FINAL TEXT
# ============================================================

def save_detected_text(
    text: str,
):

    try:

        with open(
            DETECTED_TEXT_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                text
            )

    except Exception as exc:

        print(
            "Could not save detected text:",
            repr(exc),
        )


# ============================================================
# RESET SESSION
# ============================================================

def reset_session_state():

    global detection_status

    detection_status = {
        "active": False,
        "language": None,
        "word_buffer": "",
        "sentence_buffer": "",
        "last_detected_char": "?",
        "confidence": 0.0,
        "session_id": None,
        "completed": False,
        "final_sentence": "",
        "detection_progress": 0.0,
        "auto_detection_enabled": True,
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": (
            "Unified Sign Language Detection API"
        ),
        "version": "4.0.0",

        "camera_mode": "browser",

        "backend_camera": False,

        "supported_languages": [
            "ASL",
            "ISL",
            "TSL",
            "TAMIL",
        ],

        "models": {
            "ASL":
                "random_forest_isl_model.pkl",

            "ISL":
                "final_lstm_hand_model11.keras",

            "TSL":
                "best_lstm_model.keras",
        },

        "endpoints": {
            "start_detection":
                "POST /start_detection",

            "process_frame":
                "POST /process_frame",

            "stop_detection":
                "POST /stop_detection",

            "finish_detection":
                "POST /finish_detection",

            "detection_status":
                "GET /detection_status",

            "enter":
                "POST /enter",

            "backspace":
                "POST /backspace",

            "get_detected_text":
                "GET /get_detected_text",

            "model_info":
                "GET /model_info/{language}",

            "supported_languages":
                "GET /supported_languages",
        },
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "sign-language-recognization-ml",
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# START DETECTION
# ============================================================

@app.post("/start_detection")
def start_detection(
    request: DetectionRequest,
):

    global detection_active
    global detection_status

    language = normalize_language(
        request.language
    )

    with detector_lock:

        # --------------------------------------------
        # Existing session
        # --------------------------------------------

        if detection_active:

            current_language = (
                detection_status.get(
                    "language"
                )
            )

            if (
                current_language
                == language
            ):

                return {
                    "success": True,
                    "message":
                        f"{language} detection "
                        "is already running",

                    "language":
                        language,

                    "session_id":
                        detection_status.get(
                            "session_id"
                        ),

                    "instructions":
                        detector.get_instructions(),

                    "detector_info":
                        detector.get_detector_info(),

                    "camera_mode":
                        "browser",

                    "frame_endpoint":
                        "/process_frame",
                }

            return JSONResponse(
                status_code=409,
                content={
                    "success": False,
                    "message":
                        "Another detection session "
                        "is already running",

                    "language":
                        current_language,
                },
            )

        # --------------------------------------------
        # Load model
        # --------------------------------------------

        print(
            f"[API] Starting {language} detection..."
        )

        loaded = detector.initialize(
            language
        )

        if not loaded:

            info = (
                detector.get_detector_info()
            )

            print(
                f"[API] {language} model failed "
                "to load."
            )

            return JSONResponse(
                status_code=503,
                content={
                    "success": False,

                    "message":
                        f"{language} model "
                        "could not be loaded",

                    "language":
                        language,

                    "detector_info":
                        info,
                },
            )

        # --------------------------------------------
        # New session
        # --------------------------------------------

        session_id = (
            datetime.now()
            .isoformat()
        )

        detection_active = True

        detection_status = {
            "active": True,

            "language":
                language,

            "word_buffer":
                "",

            "sentence_buffer":
                "",

            "last_detected_char":
                "?",

            "confidence":
                0.0,

            "session_id":
                session_id,

            "completed":
                False,

            "final_sentence":
                "",

            "detection_progress":
                0.0,

            "auto_detection_enabled":
                True,
        }

        detector.session_start_time = (
            session_id
        )

        detector.reset()

        # --------------------------------------------
        # Start log
        # --------------------------------------------

        try:

            with open(
                DETECTION_LOG_FILE,
                "w",
                encoding="utf-8",
            ) as file:

                file.write(
                    f"=== Browser Detection "
                    f"Session Started ===\n"
                )

                file.write(
                    f"Language: {language}\n"
                )

                file.write(
                    f"Session: {session_id}\n\n"
                )

        except Exception as exc:

            print(
                "Could not initialize log:",
                repr(exc),
            )

        print(
            f"[API] {language} detection started "
            f"session={session_id}"
        )

        return {
            "success": True,

            "message":
                f"{language} detection "
                "started successfully.",

            "language":
                language,

            "session_id":
                session_id,

            "instructions":
                detector.get_instructions(),

            "detector_info":
                detector.get_detector_info(),

            "camera_mode":
                "browser",

            "frame_endpoint":
                "/process_frame",
        }


# ============================================================
# DECODE BROWSER IMAGE
# ============================================================

def decode_browser_image(
    image: str,
):

    if not image:

        raise ValueError(
            "Empty image payload"
        )

    image = str(image).strip()

    # --------------------------------------------------------
    # Browser getScreenshot() normally returns:
    #
    # data:image/jpeg;base64,/9j/4AAQ...
    #
    # Remove the data URL prefix.
    # --------------------------------------------------------

    if "," in image:

        prefix, encoded = (
            image.split(
                ",",
                1,
            )
        )

        if (
            "base64"
            not in prefix.lower()
        ):

            raise ValueError(
                "Unsupported image data URL"
            )

        image = encoded

    # Remove accidental whitespace/newlines.
    image = "".join(
        image.split()
    )

    try:

        image_bytes = base64.b64decode(
            image,
            validate=False,
        )

    except (
        ValueError,
        binascii.Error,
    ) as exc:

        raise ValueError(
            f"Invalid base64 image: {exc}"
        )

    if not image_bytes:

        raise ValueError(
            "Decoded image is empty"
        )

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    if frame is None:

        raise ValueError(
            "OpenCV could not decode browser image"
        )

    return frame


# ============================================================
# PROCESS FRAME
# ============================================================

@app.post("/process_frame")
def process_frame_endpoint(
    request: FrameRequest,
):

    global detection_status
    global detection_active

    language = normalize_language(
        request.language
    )

    # --------------------------------------------------------
    # Session validation
    # --------------------------------------------------------

    if not detection_active:

        raise HTTPException(
            status_code=409,
            detail=(
                "Detection session is not active. "
                "Call /start_detection first."
            ),
        )

    current_language = (
        detection_status.get(
            "language"
        )
    )

    if current_language != language:

        raise HTTPException(
            status_code=409,
            detail=(
                f"Active session is "
                f"{current_language}, "
                f"but frame language is "
                f"{language}."
            ),
        )

    # --------------------------------------------------------
    # Decode frame
    # --------------------------------------------------------

    try:

        frame = decode_browser_image(
            request.image
        )

    except Exception as exc:

        print(
            "[API] Frame decode error:",
            repr(exc),
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid image: {exc}"
            ),
        )

    # --------------------------------------------------------
    # Process ML
    # --------------------------------------------------------

    with detector_lock:

        # Make sure the selected detector remains loaded.
        if (
            detector.language
            != language
            or detector.current_detector
            is None
            or not detector.current_detector.model_loaded
        ):

            if not detector.initialize(
                language
            ):

                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,

                        "message":
                            f"{language} model "
                            "is unavailable",

                        "language":
                            language,

                        "detector_info":
                            detector.get_detector_info(),
                    },
                )

        try:

            (
                detected_char,
                confidence,
                extra_info,
            ) = detector.process_frame(
                frame
            )

        except Exception as exc:

            print(
                "[API] ML processing error:",
                repr(exc),
            )

            return JSONResponse(
                status_code=500,
                content={
                    "success": False,

                    "message":
                        "Frame processing failed",

                    "error":
                        str(exc),
                },
            )

        if detected_char is None:

            detected_char = "?"

        confidence = float(
            confidence or 0.0
        )

        extra_info = (
            extra_info or {}
        )

        should_auto_detect = bool(
            extra_info.get(
                "should_auto_detect",
                False,
            )
        )

        progress = float(
            extra_info.get(
                "detection_progress",
                0.0,
            )
        )

        # ----------------------------------------------------
        # Auto append character
        # ----------------------------------------------------

        if (
            should_auto_detect
            and detected_char != "?"
        ):

            current_word = (
                detection_status.get(
                    "word_buffer",
                    "",
                )
            )

            current_word += (
                detected_char
            )

            detection_status[
                "word_buffer"
            ] = current_word

            detection_status[
                "last_detected_char"
            ] = detected_char

            log_detection(
                "Auto-detected "
                f"'{detected_char}' "
                f"confidence="
                f"{confidence:.2f}% "
                f"word='{current_word}'"
            )

        # ----------------------------------------------------
        # Update status
        # ----------------------------------------------------

        detection_status.update({

            "active":
                detection_active,

            "language":
                language,

            "word_buffer":
                detection_status.get(
                    "word_buffer",
                    "",
                ),

            "sentence_buffer":
                detection_status.get(
                    "sentence_buffer",
                    "",
                ),

            "last_detected_char":
                detected_char,

            "confidence":
                confidence,

            "session_id":
                detection_status.get(
                    "session_id"
                ),

            "completed":
                False,

            "final_sentence":
                "",

            "detection_progress":
                progress,

            "auto_detection_enabled":
                True,
        })

        # ----------------------------------------------------
        # Response expected by Detect.js
        # ----------------------------------------------------

        return {
            "success": True,

            "detected_char":
                detected_char,

            "confidence":
                confidence,

            "should_auto_detect":
                should_auto_detect,

            "detection_progress":
                progress,

            "language":
                language,

            "completed":
                False,

            "word_buffer":
                detection_status[
                    "word_buffer"
                ],

            "sentence_buffer":
                detection_status[
                    "sentence_buffer"
                ],

            "hand_detected":
                extra_info.get(
                    "hand_detected",
                    False,
                ),

            "hand_count":
                extra_info.get(
                    "hand_count",
                    0,
                ),
        }


# ============================================================
# DETECTION STATUS
# ============================================================

@app.get("/detection_status")
def get_detection_status():

    status = dict(
        detection_status
    )

    status["success"] = True

    status["detection_active"] = bool(
        detection_status.get(
            "active",
            False,
        )
    )

    status["status"] = (
        "Detection is running"
        if detection_status.get(
            "active",
            False,
        )
        else
        "Detection is stopped"
    )

    status[
        "detector_info"
    ] = detector.get_detector_info()

    return status


# ============================================================
# ENTER
# ============================================================

@app.post("/enter")
def enter_word():

    global detection_status

    word = (
        detection_status
        .get(
            "word_buffer",
            "",
        )
        .strip()
    )

    if not word:

        return {
            "success": True,
            "message":
                "No word to add",
            "added": False,
            "sentence":
                detection_status.get(
                    "sentence_buffer",
                    "",
                ),
        }

    sentence = (
        detection_status
        .get(
            "sentence_buffer",
            "",
        )
        .strip()
    )

    if sentence:

        sentence = (
            sentence
            + " "
            + word
        )

    else:

        sentence = word

    detection_status[
        "sentence_buffer"
    ] = sentence

    detection_status[
        "word_buffer"
    ] = ""

    log_detection(
        f"Word added to sentence: "
        f"'{word}' -> '{sentence}'"
    )

    return {
        "success": True,
        "message":
            "Word added",
        "added":
            True,
        "word":
            word,
        "sentence":
            sentence,
    }


# ============================================================
# BACKSPACE
# ============================================================

@app.post("/backspace")
def backspace_word():

    global detection_status

    word = (
        detection_status
        .get(
            "word_buffer",
            "",
        )
    )

    if not word:

        return {
            "success": True,
            "message":
                "No characters to remove",
            "word":
                "",
        }

    word = word[:-1]

    detection_status[
        "word_buffer"
    ] = word

    return {
        "success": True,
        "message":
            "Last character removed",
        "word":
            word,
    }


# ============================================================
# STOP DETECTION
# ============================================================

@app.post("/stop_detection")
def stop_detection():

    global detection_active
    global detection_status

    with detector_lock:

        if not detection_active:

            return {
                "success": True,
                "message":
                    "Detection is already stopped",
                "active":
                    False,
                "final_sentence":
                    detection_status.get(
                        "final_sentence",
                        "",
                    ),
            }

        detection_active = False

        word = (
            detection_status
            .get(
                "word_buffer",
                "",
            )
            .strip()
        )

        sentence = (
            detection_status
            .get(
                "sentence_buffer",
                "",
            )
            .strip()
        )

        # Add unfinished word.
        if word:

            if sentence:

                sentence += (
                    " "
                    + word
                )

            else:

                sentence = word

        detection_status.update({

            "active":
                False,

            "word_buffer":
                "",

            "sentence_buffer":
                sentence,

            "final_sentence":
                sentence,

            "completed":
                True,

            "detection_progress":
                0.0,
        })

        save_detected_text(
            sentence
        )

        log_detection(
            "Browser detection stopped. "
            f"Final sentence: '{sentence}'"
        )

        # Do NOT destroy TensorFlow models.
        # They remain cached for the next session.

        detector.reset()

        print(
            "[API] Detection stopped."
        )

        return {
            "success": True,

            "message":
                "Detection stopped successfully",

            "active":
                False,

            "final_sentence":
                sentence,
        }


# ============================================================
# FINISH DETECTION
# ============================================================

@app.post("/finish_detection")
def finish_detection():

    global detection_active
    global detection_status

    word = (
        detection_status
        .get(
            "word_buffer",
            "",
        )
        .strip()
    )

    sentence = (
        detection_status
        .get(
            "sentence_buffer",
            "",
        )
        .strip()
    )

    if word:

        if sentence:

            sentence += (
                " "
                + word
            )

        else:

            sentence = word

    detection_active = False

    detection_status.update({

        "active":
            False,

        "word_buffer":
            "",

        "sentence_buffer":
            sentence,

        "final_sentence":
            sentence,

        "completed":
            True,

        "detection_progress":
            0.0,
    })

    save_detected_text(
        sentence
    )

    log_detection(
        "Detection completed. "
        f"Final sentence: '{sentence}'"
    )

    detector.reset()

    return {
        "success": True,

        "message":
            "Detection completed",

        "sentence":
            sentence,
    }


# ============================================================
# GET DETECTED TEXT
# ============================================================

@app.get("/get_detected_text")
def get_detected_text():

    try:

        if os.path.isfile(
            DETECTED_TEXT_FILE
        ):

            with open(
                DETECTED_TEXT_FILE,
                "r",
                encoding="utf-8",
            ) as file:

                text = (
                    file.read()
                    .strip()
                )

            return {
                "success": True,
                "text": text,
                "available": True,
                "language":
                    detection_status.get(
                        "language"
                    ),
                "timestamp":
                    datetime.now().isoformat(),
            }

        return {
            "success": True,
            "text": "",
            "available": False,
            "language":
                detection_status.get(
                    "language"
                ),
            "timestamp":
                datetime.now().isoformat(),
        }

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message":
                    f"Could not read detected text: {exc}",
            },
        )


# ============================================================
# DETECTION LOG
# ============================================================

@app.get("/detection_log")
def get_detection_log():

    try:

        if not os.path.isfile(
            DETECTION_LOG_FILE
        ):

            return {
                "success": True,
                "log": "",
            }

        with open(
            DETECTION_LOG_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            content = file.read()

        return {
            "success": True,
            "log": content,
        }

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message":
                    str(exc),
            },
        )


# ============================================================
# CLEAR SESSION
# ============================================================

@app.delete("/clear_session")
def clear_session():

    global detection_active

    detection_active = False

    detector.reset()

    reset_session_state()

    try:

        if os.path.isfile(
            DETECTED_TEXT_FILE
        ):

            os.remove(
                DETECTED_TEXT_FILE
            )

    except Exception as exc:

        print(
            "Could not remove detected text:",
            repr(exc),
        )

    return {
        "success": True,
        "message":
            "Session cleared successfully",
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model_info/{language}")
def model_info(
    language: str,
):

    language = normalize_language(
        language
    )

    with detector_lock:

        loaded = detector.initialize(
            language
        )

        info = (
            detector.get_detector_info()
        )

        return {
            "success":
                bool(loaded),

            "language":
                language,

            "model_available":
                bool(loaded),

            "detector_info":
                info,
        }


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

@app.get("/supported_languages")
def supported_languages():

    return {
        "success": True,

        "supported_languages": [
            "ASL",
            "ISL",
            "TSL",
            "TAMIL",
        ],

        "ASL": {
            "name":
                "American Sign Language",

            "model":
                "Random Forest",

            "model_file":
                "random_forest_isl_model.pkl",

            "input_features":
                84,

            "hands":
                2,

            "classes":
                35,
        },

        "ISL": {
            "name":
                "Indian Sign Language",

            "model":
                "LSTM",

            "model_file":
                "final_lstm_hand_model11.keras",

            "input_shape":
                "(1, 1, 84)",

            "hands":
                2,

            "classes":
                35,
        },

        "TSL": {
            "name":
                "Tamil Sign Language",

            "model":
                "LSTM",

            "model_file":
                "best_lstm_model.keras",

            "labels_file":
                "tamil_labels.json",

            "input_shape":
                "(1, 1, 84)",

            "hands":
                2,
        },

        "TAMIL": {
            "alias_for":
                "TSL",
        },
    }


# ============================================================
# WORD FORMATION
# ============================================================

@app.get("/word_formation/{language}")
def word_formation(
    language: str,
):

    language = normalize_language(
        language
    )

    common_words = {

        "ASL": {
            "HELLO":
                ["H", "E", "L", "L", "O"],

            "WORLD":
                ["W", "O", "R", "L", "D"],

            "LOVE":
                ["L", "O", "V", "E"],

            "PEACE":
                ["P", "E", "A", "C", "E"],
        },

        "ISL": {
            "HELLO":
                ["H", "E", "L", "L", "O"],

            "NAMASTE":
                ["N", "A", "M", "A", "S", "T", "E"],

            "INDIA":
                ["I", "N", "D", "I", "A"],
        },

        "TSL": {
            "VANAKKAM":
                ["வ", "ண", "க", "க", "ம"],

            "NANDRI":
                ["ந", "ன", "ற", "ி"],

            "TAMIL":
                ["த", "மி", "ழ"],
        },
    }

    return {
        "success": True,

        "language":
            language,

        "available_words":
            common_words.get(
                language,
                {},
            ),

        "instructions":
            (
                "Hold each character sign "
                "steadily until it is detected."
            ),
    }


# ============================================================
# STATISTICS
# ============================================================

@app.get("/statistics")
def statistics():

    word = (
        detection_status
        .get(
            "word_buffer",
            "",
        )
    )

    sentence = (
        detection_status
        .get(
            "sentence_buffer",
            "",
        )
    )

    characters_detected = (
        len(word)
        + len(sentence)
    )

    words_formed = (
        len(
            sentence.split()
        )
        if sentence
        else 0
    )

    session_duration = None

    session_id = (
        detection_status.get(
            "session_id"
        )
    )

    if (
        detection_status.get(
            "active"
        )
        and session_id
    ):

        try:

            started = (
                datetime.fromisoformat(
                    session_id
                )
            )

            duration = (
                datetime.now()
                - started
            )

            session_duration = (
                str(duration)
                .split(".")[0]
            )

        except Exception:
            session_duration = None

    return {
        "success": True,

        "current_session": {

            "active":
                detection_status.get(
                    "active",
                    False,
                ),

            "language":
                detection_status.get(
                    "language"
                ),

            "characters_detected":
                characters_detected,

            "words_formed":
                words_formed,

            "session_duration":
                session_duration,
        },

        "system_info": {

            "supported_languages": [
                "ASL",
                "ISL",
                "TSL",
                "TAMIL",
            ],

            "camera":
                "Browser",

            "detection_method":
                "Browser webcam + MediaPipe + ML",

            "api_version":
                "4.0.0",
        },
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    os.makedirs(
        BASE_DIR,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "Unified Sign Language API starting"
    )
    print("=" * 70)

    print(
        "BASE_DIR:",
        BASE_DIR,
    )

    print(
        "Frontend:",
        FRONTEND_ORIGIN,
    )

    print(
        "ASL model:",
        os.path.join(
            BASE_DIR,
            "random_forest_isl_model.pkl",
        ),
    )

    print(
        "ISL model:",
        os.path.join(
            BASE_DIR,
            "final_lstm_hand_model11.keras",
        ),
    )

    print(
        "TSL model:",
        os.path.join(
            BASE_DIR,
            "best_lstm_model.keras",
        ),
    )

    print(
        "TSL labels:",
        os.path.join(
            BASE_DIR,
            "tamil_labels.json",
        ),
    )

    # Do NOT load all TensorFlow models at startup.
    #
    # Render can take a long time to boot TensorFlow/MediaPipe.
    # Models are loaded lazily on /start_detection.
    #
    # This also prevents ASL from failing because an unrelated
    # TSL/ISL model has a problem.

    print(
        "Models will be loaded lazily "
        "when detection starts."
    )

    print("=" * 70)


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
def shutdown_event():

    print(
        "Shutting down sign language API..."
    )

    try:

        detector.cleanup()

    except Exception as exc:

        print(
            "Detector cleanup error:",
            repr(exc),
        )


# ============================================================
# LOCAL EXECUTION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            "10000",
        )
    )

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )