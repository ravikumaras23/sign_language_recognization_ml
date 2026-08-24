from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import base64
import cv2
import numpy as np
import os
from datetime import datetime

from models import UnifiedSignLanguageDetector


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SingLang Sign Language Detection API",
    version="5.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FILES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DETECTED_TEXT_FILE = os.path.join(
    BASE_DIR,
    "detected_text.txt"
)

DETECTION_LOG_FILE = os.path.join(
    BASE_DIR,
    "sign_language_detections.txt"
)


# ============================================================
# GLOBAL DETECTOR
# ============================================================

detector = UnifiedSignLanguageDetector()


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
    language: str = "ISL"


class FrameRequest(BaseModel):
    image: str
    language: str = "ISL"


# ============================================================
# LANGUAGES
# ============================================================

SUPPORTED_LANGUAGES = [
    "ASL",
    "ISL",
    "TSL",
    "TAMIL",
]


def normalize_language(language: str) -> str:

    language = (
        language or "ISL"
    ).strip().upper()

    if language not in SUPPORTED_LANGUAGES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported language. "
                "Use ASL, ISL, TSL or TAMIL."
            ),
        )

    return language


# ============================================================
# LOGGING
# ============================================================

def log_detection(message: str):

    try:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(
            DETECTION_LOG_FILE,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                f"[{timestamp}] {message}\n"
            )

    except Exception as exc:

        print(
            "Log error:",
            exc
        )


# ============================================================
# RESET STATUS
# ============================================================

def reset_status(language=None):

    global detection_status

    detection_status = {

        "active": False,

        "language": language,

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
# START DETECTION
# ============================================================

@app.post("/start_detection")
async def start_detection(
    request: DetectionRequest
):

    global detection_active
    global detection_status
    global detector

    language = normalize_language(
        request.language
    )

    if detection_active:

        return JSONResponse(

            status_code=409,

            content={
                "success": False,
                "message":
                    "Detection is already running.",
                "language":
                    detection_status.get(
                        "language"
                    ),
            }
        )

    print(
        f"Starting browser detection: {language}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        initialized = detector.initialize(
            language
        )

    except Exception as exc:

        print(
            "Detector initialization error:",
            exc
        )

        return JSONResponse(

            status_code=503,

            content={
                "success": False,
                "message":
                    f"Could not initialize {language} detector.",
                "error": str(exc),
            }
        )

    if not initialized:

        try:

            info = detector.get_detector_info()

        except Exception:

            info = {}

        return JSONResponse(

            status_code=503,

            content={
                "success": False,
                "message":
                    f"{language} model could not be loaded.",
                "detector_info": info,
            }
        )

    # --------------------------------------------------------
    # New session
    # --------------------------------------------------------

    session_id = datetime.now().isoformat()

    detection_active = True

    detection_status = {

        "active": True,

        "language": language,

        "word_buffer": "",

        "sentence_buffer": "",

        "last_detected_char": "?",

        "confidence": 0.0,

        "session_id": session_id,

        "completed": False,

        "final_sentence": "",

        "detection_progress": 0.0,

        "auto_detection_enabled": True,
    }

    try:

        detector.session_start_time = session_id

    except Exception:

        pass

    # --------------------------------------------------------
    # Clear old output
    # --------------------------------------------------------

    try:

        with open(
            DETECTED_TEXT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write("")

    except Exception as exc:

        print(
            "Could not clear detected text:",
            exc
        )

    try:

        with open(
            DETECTION_LOG_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                f"=== {language} Detection Session "
                f"Started: {session_id} ===\n"
            )

    except Exception:

        pass

    log_detection(
        f"Browser detection started: {language}"
    )

    try:

        instructions = detector.get_instructions()

    except Exception:

        instructions = []

    try:

        detector_info = (
            detector.get_detector_info()
        )

    except Exception:

        detector_info = {}

    print(
        f"Detection started: "
        f"{language} | "
        f"session={session_id}"
    )

    return {

        "success": True,

        "message":
            f"{language} detection started successfully.",

        "language":
            language,

        "session_id":
            session_id,

        "instructions":
            instructions,

        "detector_info":
            detector_info,

        "camera_mode":
            "browser",

        "frame_endpoint":
            "/process_frame",
    }


# ============================================================
# PROCESS FRAME
# ============================================================

@app.post("/process_frame")
async def process_frame(
    request: FrameRequest
):

    global detection_active
    global detection_status
    global detector

    language = normalize_language(
        request.language
    )

    if not detection_active:

        raise HTTPException(

            status_code=400,

            detail=(
                "Detection session is not active. "
                "Click Start Detection first."
            )
        )

    if not request.image:

        raise HTTPException(
            status_code=400,
            detail="No image was received."
        )

    # --------------------------------------------------------
    # Decode browser JPEG
    # --------------------------------------------------------

    try:

        image_string = request.image

        if "," in image_string:

            image_string = image_string.split(
                ",",
                1
            )[1]

        image_bytes = base64.b64decode(
            image_string
        )

        np_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            raise ValueError(
                "OpenCV could not decode frame."
            )

    except Exception as exc:

        print(
            "Frame decode error:",
            exc
        )

        raise HTTPException(

            status_code=400,

            detail=(
                f"Invalid webcam frame: {exc}"
            )
        )

    # --------------------------------------------------------
    # Make sure correct model is loaded
    # --------------------------------------------------------

    try:

        current_language = getattr(
            detector,
            "language",
            None
        )

        if current_language != language:

            initialized = detector.initialize(
                language
            )

            if not initialized:

                return JSONResponse(

                    status_code=503,

                    content={
                        "success": False,
                        "message":
                            f"{language} detector unavailable."
                    }
                )

    except Exception as exc:

        print(
            "Detector initialization error:",
            exc
        )

        return JSONResponse(

            status_code=503,

            content={
                "success": False,
                "message":
                    "Detector initialization failed.",
                "error": str(exc),
            }
        )

    # --------------------------------------------------------
    # OpenCV + MediaPipe + ML
    # --------------------------------------------------------

    try:

        result = detector.process_frame(
            frame
        )

        if not result:

            raise RuntimeError(
                "Detector returned no result."
            )

        detected_char = result[0]

        confidence = result[1]

        extra_info = (
            result[2]
            if len(result) > 2
            else {}
        )

        if detected_char is None:

            detected_char = "?"

        confidence = float(
            confidence or 0
        )

        if not isinstance(
            extra_info,
            dict
        ):

            extra_info = {}

        progress = float(
            extra_info.get(
                "detection_progress",
                0
            ) or 0
        )

        should_auto_detect = bool(
            extra_info.get(
                "should_auto_detect",
                False
            )
        )

        # ----------------------------------------------------
        # Current word
        # ----------------------------------------------------

        current_word = detection_status.get(
            "word_buffer",
            ""
        )

        last_char = detection_status.get(
            "last_detected_char",
            "?"
        )

        # ----------------------------------------------------
        # Add prediction
        # ----------------------------------------------------

        if (
            should_auto_detect
            and detected_char != "?"
        ):

            # Prevent adding the same held sign
            # continuously every 250 ms.

            if detected_char != last_char:

                current_word += str(
                    detected_char
                )

                detection_status[
                    "word_buffer"
                ] = current_word

                log_detection(
                    f"Detected "
                    f"'{detected_char}' "
                    f"confidence="
                    f"{confidence:.2f}% "
                    f"word="
                    f"'{current_word}'"
                )

        # ----------------------------------------------------
        # Update status
        # ----------------------------------------------------

        detection_status.update({

            "active":
                True,

            "language":
                language,

            "word_buffer":
                detection_status.get(
                    "word_buffer",
                    ""
                ),

            "sentence_buffer":
                detection_status.get(
                    "sentence_buffer",
                    ""
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

        return {

            "success":
                True,

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

            "active":
                True,

            "word_buffer":
                detection_status[
                    "word_buffer"
                ],

            "sentence_buffer":
                detection_status[
                    "sentence_buffer"
                ],
        }

    except Exception as exc:

        print(
            "Frame processing error:",
            repr(exc)
        )

        return JSONResponse(

            status_code=500,

            content={

                "success":
                    False,

                "message":
                    "Frame processing failed.",

                "error":
                    str(exc),

                "detected_char":
                    "?",

                "confidence":
                    0,

                "detection_progress":
                    0,
            }
        )


# ============================================================
# ENTER WORD
# ============================================================

@app.post("/enter")
async def enter_word():

    global detection_status

    word = detection_status.get(
        "word_buffer",
        ""
    ).strip()

    sentence = detection_status.get(
        "sentence_buffer",
        ""
    ).strip()

    if word:

        if sentence:

            sentence += " " + word

        else:

            sentence = word

    detection_status[
        "word_buffer"
    ] = ""

    detection_status[
        "sentence_buffer"
    ] = sentence

    detection_status[
        "last_detected_char"
    ] = "?"

    log_detection(
        f"Word added: '{word}'"
    )

    return {

        "success":
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
async def backspace():

    global detection_status

    word = detection_status.get(
        "word_buffer",
        ""
    )

    if word:

        word = word[:-1]

    detection_status[
        "word_buffer"
    ] = word

    detection_status[
        "last_detected_char"
    ] = "?"

    return {

        "success":
            True,

        "word_buffer":
            word,
    }


# ============================================================
# STOP DETECTION
# ============================================================

@app.post("/stop_detection")
async def stop_detection():

    global detection_active
    global detection_status

    detection_active = False

    sentence = detection_status.get(
        "sentence_buffer",
        ""
    ).strip()

    word = detection_status.get(
        "word_buffer",
        ""
    ).strip()

    if word:

        if sentence:

            sentence += " " + word

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
            0,
    })

    try:

        with open(
            DETECTED_TEXT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(sentence)

    except Exception as exc:

        print(
            "Could not save text:",
            exc
        )

    log_detection(
        f"Detection stopped. "
        f"Final sentence: '{sentence}'"
    )

    return {

        "success":
            True,

        "message":
            "Detection stopped successfully.",

        "active":
            False,

        "final_sentence":
            sentence,
    }


# ============================================================
# FINISH
# ============================================================

@app.post("/finish_detection")
async def finish_detection():

    return await stop_detection()


# ============================================================
# STATUS
# ============================================================

@app.get("/detection_status")
async def get_detection_status():

    return detection_status


# ============================================================
# DETECTED TEXT
# ============================================================

@app.get("/get_detected_text")
async def get_detected_text():

    try:

        if not os.path.exists(
            DETECTED_TEXT_FILE
        ):

            return {

                "available":
                    False,

                "text":
                    "",

                "language":
                    detection_status.get(
                        "language"
                    ),
            }

        with open(
            DETECTED_TEXT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read().strip()

        return {

            "available":
                bool(text),

            "text":
                text,

            "language":
                detection_status.get(
                    "language"
                ),

            "timestamp":
                datetime.now().isoformat(),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# DETECTION LOG
# ============================================================

@app.get("/detection_log")
async def get_detection_log():

    try:

        if not os.path.exists(
            DETECTION_LOG_FILE
        ):

            return {
                "log": ""
            }

        with open(
            DETECTION_LOG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            content = file.read()

        return {
            "log":
                content
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# CLEAR SESSION
# ============================================================

@app.delete("/clear_session")
async def clear_session():

    global detection_active

    detection_active = False

    try:

        if os.path.exists(
            DETECTED_TEXT_FILE
        ):

            os.remove(
                DETECTED_TEXT_FILE
            )

    except Exception as exc:

        print(
            "Could not delete detected text:",
            exc
        )

    reset_status()

    return {

        "success":
            True,

        "message":
            "Session cleared successfully.",
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model_info/{language}")
async def model_info(
    language: str
):

    language = normalize_language(
        language
    )

    try:

        if getattr(
            detector,
            "language",
            None
        ) != language:

            detector.initialize(
                language
            )

        info = (
            detector.get_detector_info()
        )

        return {

            "success":
                True,

            "language":
                language,

            **(
                info
                if isinstance(info, dict)
                else {}
            )
        }

    except Exception as exc:

        return JSONResponse(

            status_code=500,

            content={

                "success":
                    False,

                "language":
                    language,

                "message":
                    str(exc),
            }
        )


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

@app.get("/supported_languages")
async def supported_languages():

    return {

        "supported_languages":
            SUPPORTED_LANGUAGES,

        "ASL": {
            "name":
                "American Sign Language",
            "capture_method":
                "Browser webcam",
            "hands_required":
                "Single hand",
        },

        "ISL": {
            "name":
                "Indian Sign Language",
            "capture_method":
                "Browser webcam",
            "model_type":
                "LSTM Neural Network",
            "hands_required":
                "Both hands supported",
        },

        "TSL": {
            "name":
                "Tamil Sign Language",
            "capture_method":
                "Browser webcam",
            "model_type":
                "LSTM Neural Network",
            "hands_required":
                "Both hands supported",
        },

        "TAMIL": {
            "name":
                "Tamil Sign Language",
            "capture_method":
                "Browser webcam",
            "model_type":
                "LSTM Neural Network",
            "hands_required":
                "Both hands supported",
        },
    }


# ============================================================
# STATISTICS
# ============================================================

@app.get("/statistics")
async def statistics():

    sentence = detection_status.get(
        "sentence_buffer",
        ""
    )

    word = detection_status.get(
        "word_buffer",
        ""
    )

    return {

        "current_session": {

            "active":
                detection_status.get(
                    "active",
                    False
                ),

            "language":
                detection_status.get(
                    "language"
                ),

            "characters_detected":
                len(sentence + word),

            "words_formed":
                len(sentence.split())
                if sentence
                else 0,

            "session_id":
                detection_status.get(
                    "session_id"
                ),
        },

        "system_info": {

            "supported_languages":
                SUPPORTED_LANGUAGES,

            "detection_method":
                "Browser webcam + OpenCV + MediaPipe + ML",

            "api_version":
                "5.0.0",
        }
    }
# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "SingLang ML API",
        "version": "5.0.0",
        "ml": True,
        "supported_languages": SUPPORTED_LANGUAGES,
    }


    

# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {

        "message":
            "SingLang Sign Language Detection API",

        "version":
            "5.0.0",

        "camera":
            "Browser",

        "processing":
            "OpenCV + MediaPipe + ML",

        "supported_languages":
            SUPPORTED_LANGUAGES,

        "endpoints": {

            "start":
                "POST /start_detection",

            "frame":
                "POST /process_frame",

            "status":
                "GET /detection_status",

            "stop":
                "POST /stop_detection",

            "enter":
                "POST /enter",

            "backspace":
                "POST /backspace",

            "text":
                "GET /get_detected_text",

            "clear":
                "DELETE /clear_session",
        }
    }









# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    print(
        "=========================================="
    )

    print(
        " SingLang ML API"
    )

    print(
        " Browser Camera + OpenCV + ML"
    )

    print(
        " http://127.0.0.1:8000"
    )

    print(
        "=========================================="
    )

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )