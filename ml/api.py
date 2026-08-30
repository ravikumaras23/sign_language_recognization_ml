import base64
import os
import threading
from datetime import datetime
from typing import Optional, Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models import UnifiedSignLanguageDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "https://sign-language-frontend-onnp.onrender.com")

app = FastAPI(title="Unified Sign Language Detection API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DETECTED_TEXT_FILE = os.path.join(BASE_DIR, "detected_text.txt")
DETECTION_LOG_FILE = os.path.join(BASE_DIR, "sign_language_detections.txt")

VALID_LANGUAGES = {"ASL", "ISL", "TSL", "TAMIL"}

detector = UnifiedSignLanguageDetector()
state_lock = threading.Lock()
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


class DetectionRequest(BaseModel):
    language: str = "ISL"


class FrameRequest(BaseModel):
    image: Optional[str] = None
    landmarks: Optional[Any] = None
    language: str = "ISL"


def log_detection(message: str):
    try:
        with open(DETECTION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")
    except Exception as exc:
        print("Log error:", exc)


def normalize_language(value: str) -> str:
    language = str(value or "").upper().strip()
    if language == "TAMIL":
        return "TSL"
    return language


def reset_status():
    detection_status.update({
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
    })


@app.get("/")
async def root():
    return {
        "success": True,
        "message": "Unified Sign Language Detection API",
        "version": "4.0.0",
        "camera_mode": "browser",
        "supported_languages": ["ASL", "ISL", "TSL", "TAMIL"],
        "frame_endpoint": "/process_frame",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "sign-language-ml"}


@app.post("/start_detection")
async def start_detection(request: DetectionRequest):
    global detection_active
    language = normalize_language(request.language)

    if language not in {"ASL", "ISL", "TSL"}:
        raise HTTPException(status_code=400, detail="Language must be ASL, ISL, TSL, or TAMIL")

    with state_lock:
        if detection_active:
            return JSONResponse(status_code=409, content={
                "success": False,
                "message": "Detection is already running",
                "language": detection_status.get("language"),
            })

        print(f"Starting browser detection: {language}")
        ok = detector.initialize(language)
        if not ok:
            info = detector.get_detector_info()
            print(f"MODEL LOAD FAILED: {language} | {info}")
            return JSONResponse(status_code=503, content={
                "success": False,
                "message": f"{language} model could not be loaded",
                "language": language,
                "detector_info": info,
            })

        detection_active = True
        session_id = datetime.now().isoformat()
        detector.session_start_time = session_id
        detector.reset()
        reset_status()
        detection_status.update({
            "active": True,
            "language": language,
            "session_id": session_id,
        })
        log_detection(f"Session started: {language} {session_id}")

        return {
            "success": True,
            "message": f"{language} detection started successfully.",
            "language": language,
            "session_id": session_id,
            "instructions": detector.get_instructions(),
            "detector_info": detector.get_detector_info(),
            "camera_mode": "browser",
            "frame_endpoint": "/process_frame",
        }


@app.post("/process_frame")
async def process_frame_endpoint(request: FrameRequest):
    global detection_active
    language = normalize_language(request.language)

    if language not in {"ASL", "ISL", "TSL"}:
        raise HTTPException(status_code=400, detail="Unsupported language")
    if not request.image:
        raise HTTPException(status_code=400, detail="No image provided")

    with state_lock:
        if not detection_active:
            raise HTTPException(status_code=400, detail="Detection session is not active. Start detection first.")

        # react-webcam returns data:image/jpeg;base64,...
        raw = request.image
        if "," in raw and raw.startswith("data:"):
            raw = raw.split(",", 1)[1]

        try:
            image_data = base64.b64decode(raw, validate=False)
            frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("OpenCV could not decode image")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

        # Switch model only if the browser changes language during a session.
        if detector.language != language or detector.current_detector is None or not detector.current_detector.model_loaded:
            if not detector.initialize(language):
                return JSONResponse(status_code=503, content={
                    "success": False,
                    "message": f"{language} model could not be loaded",
                    "detector_info": detector.get_detector_info(),
                })

        try:
            detected_char, confidence, extra = detector.process_frame(frame)
            confidence = float(confidence or 0.0)
            extra = extra or {}
            auto_detect = bool(extra.get("should_auto_detect", False))
            progress = float(extra.get("detection_progress", 0.0))

            if auto_detect and detected_char != "?":
                detection_status["word_buffer"] += detected_char
                log_detection(f"Auto-added {detected_char} to {language}")

            detection_status.update({
                "active": True,
                "language": language,
                "last_detected_char": detected_char,
                "confidence": confidence,
                "detection_progress": progress,
                "auto_detection_enabled": True,
            })

            return {
                "success": True,
                "detected_char": detected_char,
                "confidence": confidence,
                "should_auto_detect": auto_detect,
                "detection_progress": progress,
                "language": language,
                "word_buffer": detection_status["word_buffer"],
                "sentence_buffer": detection_status["sentence_buffer"],
                "hand_detected": bool(extra.get("hand_detected", False)),
                "hand_count": int(extra.get("hand_count", 0)),
            }
        except Exception as exc:
            print("Frame processing error:", repr(exc))
            return JSONResponse(status_code=500, content={
                "success": False,
                "message": f"Frame processing failed: {exc}",
            })


@app.get("/detection_status")
async def get_detection_status():
    with state_lock:
        return {"success": True, **detection_status}


@app.post("/enter")
async def enter_word():
    with state_lock:
        word = detection_status["word_buffer"].strip()
        sentence = detection_status["sentence_buffer"].strip()
        if word:
            sentence = f"{sentence} {word}".strip()
        detection_status["sentence_buffer"] = sentence
        detection_status["word_buffer"] = ""
        detector.reset()
        return {"success": True, "sentence": sentence, "word_buffer": ""}


@app.post("/backspace")
async def backspace():
    with state_lock:
        word = detection_status["word_buffer"]
        if word:
            detection_status["word_buffer"] = word[:-1]
        else:
            sentence = detection_status["sentence_buffer"]
            detection_status["sentence_buffer"] = sentence[:-1] if sentence else ""
        detector.reset()
        return {
            "success": True,
            "word_buffer": detection_status["word_buffer"],
            "sentence_buffer": detection_status["sentence_buffer"],
        }


@app.post("/stop_detection")
async def stop_detection():
    global detection_active
    with state_lock:
        if not detection_active:
            return {"success": True, "message": "Detection is not running", "active": False, "final_sentence": detection_status.get("final_sentence", "")}

        final_sentence = detection_status["sentence_buffer"].strip()
        word = detection_status["word_buffer"].strip()
        if word:
            final_sentence = f"{final_sentence} {word}".strip()

        detection_active = False
        detection_status.update({
            "active": False,
            "sentence_buffer": final_sentence,
            "word_buffer": "",
            "final_sentence": final_sentence,
            "completed": True,
            "detection_progress": 0.0,
        })
        detector.reset()
        log_detection(f"Session stopped. Final sentence: {final_sentence}")

        try:
            with open(DETECTED_TEXT_FILE, "w", encoding="utf-8") as f:
                f.write(final_sentence)
        except Exception as exc:
            print("Could not save detected text:", exc)

        return {"success": True, "message": "Detection stopped successfully", "active": False, "final_sentence": final_sentence}


@app.post("/finish_detection")
async def finish_detection():
    return await stop_detection()


@app.delete("/clear_session")
async def clear_session():
    global detection_active
    with state_lock:
        detection_active = False
        detector.reset()
        reset_status()
        try:
            if os.path.exists(DETECTED_TEXT_FILE):
                os.remove(DETECTED_TEXT_FILE)
        except Exception:
            pass
        return {"success": True, "message": "Session cleared successfully"}


@app.get("/get_detected_text")
async def get_detected_text():
    text = detection_status.get("final_sentence", "")
    if not text and os.path.exists(DETECTED_TEXT_FILE):
        try:
            with open(DETECTED_TEXT_FILE, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception:
            pass
    return {"text": text, "available": bool(text), "language": detection_status.get("language")}


@app.get("/detection_log")
async def get_detection_log():
    try:
        with open(DETECTION_LOG_FILE, "r", encoding="utf-8") as f:
            return {"log": f.read()}
    except FileNotFoundError:
        return {"log": ""}


@app.get("/supported_languages")
async def supported_languages():
    return {
        "supported_languages": ["ASL", "ISL", "TSL", "TAMIL"],
        "ASL": {"name": "American Sign Language", "model_type": "Random Forest", "input_features": 84, "hands": 1},
        "ISL": {"name": "Indian Sign Language", "model_type": "LSTM", "input_shape": "(1, 1, 84)", "hands": 2},
        "TSL": {"name": "Tamil Sign Language", "model_type": "LSTM", "input_shape": "(1, 1, 84)", "hands": 2},
    }


@app.get("/model_info/{language}")
async def model_info(language: str):
    language = normalize_language(language)
    if language not in {"ASL", "ISL", "TSL"}:
        raise HTTPException(status_code=400, detail="Unsupported language")
    with state_lock:
        if detector.language != language or detector.current_detector is None or not detector.current_detector.model_loaded:
            ok = detector.initialize(language)
        else:
            ok = True
        info = detector.get_detector_info()
        info["success"] = bool(ok)
        return info


@app.get("/statistics")
async def statistics():
    with state_lock:
        return {
            "current_session": {
                "active": detection_status["active"],
                "language": detection_status.get("language"),
                "characters_detected": len(detection_status.get("word_buffer", "")),
                "sentence": detection_status.get("sentence_buffer", ""),
                "session_id": detection_status.get("session_id"),
            },
            "system_info": {
                "camera": "browser",
                "supported_languages": ["ASL", "ISL", "TSL", "TAMIL"],
            },
        }
