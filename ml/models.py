import os
import json
import pickle
import traceback
from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def model_path(filename: str) -> str:
    return os.path.join(BASE_DIR, filename)


ASL_LABELS = {
    0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8", 8: "9",
    9: "A", 10: "B", 11: "C", 12: "D", 13: "E", 14: "F", 15: "G",
    16: "H", 17: "I", 18: "J", 19: "K", 20: "L", 21: "M", 22: "N",
    23: "O", 24: "P", 25: "Q", 26: "R", 27: "S", 28: "T", 29: "U",
    30: "V", 31: "W", 32: "X", 33: "Y", 34: "Z",
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def _make_hands(max_num_hands: int = 2):
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _hand_xy(hand) -> np.ndarray:
    values = []
    for lm in hand.landmark[:21]:
        values.extend([float(lm.x), float(lm.y)])
    values = np.asarray(values, dtype=np.float32)
    if values.size < 42:
        values = np.pad(values, (0, 42 - values.size))
    return values[:42]


def _two_hand_xy(hands) -> np.ndarray:
    first = np.zeros(42, dtype=np.float32)
    second = np.zeros(42, dtype=np.float32)
    if hands:
        first = _hand_xy(hands[0])
        if len(hands) > 1:
            second = _hand_xy(hands[1])
    features = np.concatenate([first, second]).astype(np.float32)
    if features.size != 84:
        raise ValueError(f"Expected 84 features, got {features.size}")
    return features.reshape(1, 84)


def _asl_84_features(hand) -> np.ndarray:
    """ASL RF preprocessing: normalized first hand (42) + 42 zero features."""
    if hand is None:
        raise ValueError("No ASL hand landmarks")

    xs = [float(lm.x) for lm in hand.landmark[:21]]
    ys = [float(lm.y) for lm in hand.landmark[:21]]
    min_x = min(xs)
    min_y = min(ys)

    values = []
    for lm in hand.landmark[:21]:
        values.extend([float(lm.x) - min_x, float(lm.y) - min_y])

    values = values[:42]
    values += [0.0] * (42 - len(values))
    features = np.asarray(values + [0.0] * 42, dtype=np.float32)
    if features.size != 84:
        raise ValueError(f"Expected 84 ASL features, got {features.size}")
    return features.reshape(1, 84)


def _extract_model(obj):
    if isinstance(obj, dict):
        for key in ("model", "classifier", "estimator"):
            if key in obj:
                return obj[key]
        raise ValueError("Pickle dictionary does not contain model/classifier/estimator")
    return obj


class StableDetectorMixin:
    confidence_threshold = 60.0
    required_consistent_frames = 3
    cooldown_frames = 4

    def _init_state(self):
        self.last_prediction = None
        self.prediction_counter = 0
        self.cooldown_counter = 0
        self.last_added_char = None
        self.last_results = None

    def reset_detection_state(self):
        self._init_state()

    def should_auto_detect(self, detected_char: str, confidence: float):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, 0.0

        if detected_char == "?" or confidence < self.confidence_threshold:
            self.last_prediction = None
            self.prediction_counter = 0
            return False, 0.0

        if self.last_prediction == detected_char:
            self.prediction_counter += 1
        else:
            self.last_prediction = detected_char
            self.prediction_counter = 1

        progress = min(self.prediction_counter / self.required_consistent_frames, 1.0)
        if self.prediction_counter >= self.required_consistent_frames:
            self.prediction_counter = 0
            self.cooldown_counter = self.cooldown_frames
            self.last_added_char = detected_char
            return True, 1.0
        return False, progress


class ASLDetector(StableDetectorMixin):
    def __init__(self):
        self.model = None
        self.hands = None
        self.model_loaded = False
        self.model_path = model_path("random_forest_isl_model.pkl")
        self.labels_dict = ASL_LABELS.copy()
        self.expected_features = 84
        self.confidence_threshold = 60.0
        self.required_consistent_frames = 3
        self.cooldown_frames = 4
        self._init_state()

    def load_model(self):
        try:
            print("[ASL] Loading Random Forest:", self.model_path)
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(self.model_path)
            with open(self.model_path, "rb") as f:
                self.model = _extract_model(pickle.load(f))

            actual = getattr(self.model, "n_features_in_", None)
            classes = getattr(self.model, "classes_", None)
            print("[ASL] model type:", type(self.model))
            print("[ASL] features:", actual)
            print("[ASL] classes:", classes)
            if actual != 84:
                raise ValueError(f"ASL Random Forest expects 84 features, got {actual}")

            self.hands = _make_hands(1)
            self.reset_detection_state()
            self.model_loaded = True
            return True
        except Exception as exc:
            print("[ASL] load error:", repr(exc))
            traceback.print_exc()
            self.model = None
            self.model_loaded = False
            return False

    def predict(self, hands):
        try:
            if not self.model_loaded or self.model is None or not hands:
                return "?", 0.0
            features = _asl_84_features(hands[0])
            pred = self.model.predict(features)[0]
            idx = int(pred)
            confidence = 0.0
            if hasattr(self.model, "predict_proba"):
                p = np.asarray(self.model.predict_proba(features)[0])
                if p.size:
                    confidence = float(np.max(p)) * 100.0
            char = self.labels_dict.get(idx, "?")
            return (char, confidence) if confidence >= self.confidence_threshold else ("?", confidence)
        except Exception as exc:
            print("[ASL] prediction error:", repr(exc))
            return "?", 0.0

    def get_instructions(self):
        return [
            "ASL - Place one hand inside the camera area",
            "Hold the sign steadily",
            "The same character is accepted after 3 stable frames",
            "Show a different sign for the next character",
            "Use Enter to add the current word to the sentence",
        ]

    def cleanup(self):
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
        self.hands = None
        self.last_results = None


class ISLDetector(StableDetectorMixin):
    def __init__(self):
        self.model = None
        self.hands = None
        self.model_loaded = False
        self.model_path = model_path("final_lstm_hand_model11.keras")
        self.labels_dict = ASL_LABELS.copy()
        self.confidence_threshold = 60.0
        self.required_consistent_frames = 3
        self.cooldown_frames = 4
        self._init_state()

    def load_model(self):
        try:
            print("[ISL] Loading LSTM:", self.model_path)
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(self.model_path)
            self.model = load_model(self.model_path, compile=False)
            print("[ISL] input:", self.model.input_shape, "output:", self.model.output_shape)
            self.hands = _make_hands(2)
            self.reset_detection_state()
            self.model_loaded = True
            return True
        except Exception as exc:
            print("[ISL] load error:", repr(exc))
            traceback.print_exc()
            self.model = None
            self.model_loaded = False
            return False

    def predict(self, hands):
        try:
            if not self.model_loaded or self.model is None or not hands:
                return "?", 0.0
            features = _two_hand_xy(hands).reshape(1, 1, 84)
            prediction = np.asarray(self.model.predict(features, verbose=0)).flatten()
            if prediction.size == 0:
                return "?", 0.0
            idx = int(np.argmax(prediction))
            confidence = float(prediction[idx]) * 100.0
            char = self.labels_dict.get(idx, "?")
            return (char, confidence) if confidence >= self.confidence_threshold else ("?", confidence)
        except Exception as exc:
            print("[ISL] prediction error:", repr(exc))
            return "?", 0.0

    def get_instructions(self):
        return [
            "ISL - Place hands inside the camera area",
            "Hold the same sign steadily",
            "The same character is accepted after 3 stable frames",
            "Show a different sign for the next character",
            "Use Enter to add the current word to the sentence",
        ]

    def cleanup(self):
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
        self.hands = None
        self.last_results = None


class TSLDetector(StableDetectorMixin):
    def __init__(self):
        self.model = None
        self.hands = None
        self.model_loaded = False
        self.model_path = model_path("best_lstm_model.keras")
        self.labels_path = model_path("tamil_labels.json")
        self.labels_dict: Dict[int, str] = {}
        self.confidence_threshold = 60.0
        self.required_consistent_frames = 3
        self.cooldown_frames = 4
        self._init_state()

    def load_model(self):
        try:
            print("[TSL] Loading Tamil LSTM:", self.model_path)
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(self.model_path)
            if not os.path.isfile(self.labels_path):
                raise FileNotFoundError(self.labels_path)

            self.model = load_model(self.model_path, compile=False)
            with open(self.labels_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.labels_dict = {int(k): str(v) for k, v in raw.items()}
            print("[TSL] labels:", len(self.labels_dict))
            print("[TSL] input:", self.model.input_shape, "output:", self.model.output_shape)
            self.hands = _make_hands(2)
            self.reset_detection_state()
            self.model_loaded = True
            return True
        except Exception as exc:
            print("[TSL] load error:", repr(exc))
            traceback.print_exc()
            self.model = None
            self.model_loaded = False
            return False

    def predict(self, hands):
        try:
            if not self.model_loaded or self.model is None or not hands:
                return "?", 0.0
            features = _two_hand_xy(hands).reshape(1, 1, 84)
            prediction = np.asarray(self.model.predict(features, verbose=0)).flatten()
            if prediction.size == 0:
                return "?", 0.0
            idx = int(np.argmax(prediction))
            confidence = float(prediction[idx]) * 100.0
            char = self.labels_dict.get(idx, "?")
            return (char, confidence) if confidence >= self.confidence_threshold else ("?", confidence)
        except Exception as exc:
            print("[TSL] prediction error:", repr(exc))
            return "?", 0.0

    def get_instructions(self):
        return [
            "TSL / Tamil - Place hands inside the camera area",
            "Hold the same sign steadily",
            "The same character is accepted after 3 stable frames",
            "Show a different sign for the next character",
            "Use Enter to add the current word to the sentence",
        ]

    def cleanup(self):
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
        self.hands = None
        self.last_results = None


class UnifiedSignLanguageDetector:
    def __init__(self):
        self.asl_detector = ASLDetector()
        self.isl_detector = ISLDetector()
        self.tsl_detector = TSLDetector()
        self.current_detector = None
        self.language = None
        self.session_start_time = None

    def initialize(self, language: str = "ISL") -> bool:
        language = str(language).upper().strip()
        if language == "TAMIL":
            language = "TSL"
        self.language = language

        if language == "ASL":
            self.current_detector = self.asl_detector
        elif language == "ISL":
            self.current_detector = self.isl_detector
        elif language == "TSL":
            self.current_detector = self.tsl_detector
        else:
            self.current_detector = None
            return False

        return self.current_detector.load_model()

    def get_detector_info(self) -> Dict[str, Any]:
        d = self.current_detector
        if d is None:
            return {}
        return {
            "language": self.language,
            "model_type": type(d.model).__name__ if d.model is not None else None,
            "model_path": d.model_path,
            "model_exists": os.path.isfile(d.model_path),
            "model_loaded": bool(d.model_loaded),
            "input_features": 84,
            "hands_supported": 1 if self.language == "ASL" else 2,
            "classes": len(getattr(d, "labels_dict", {})),
            "supported_characters": list(getattr(d, "labels_dict", {}).values()),
            "confidence_threshold": d.confidence_threshold,
            "consistent_frames_required": d.required_consistent_frames,
        }

    def process_frame(self, frame) -> Tuple[str, float, Dict[str, Any]]:
        if self.current_detector is None or not self.current_detector.model_loaded:
            return "?", 0.0, {"hand_detected": False, "detection_progress": 0.0, "should_auto_detect": False}

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.current_detector.hands.process(frame_rgb)
        self.current_detector.last_results = results
        hands = list(results.multi_hand_landmarks or [])

        if not hands:
            self.current_detector.last_prediction = None
            self.current_detector.prediction_counter = 0
            return "?", 0.0, {"hand_detected": False, "hand_count": 0, "detection_progress": 0.0, "should_auto_detect": False}

        detected_char, confidence = self.current_detector.predict(hands)
        should_detect, progress = self.current_detector.should_auto_detect(detected_char, confidence)
        return detected_char, float(confidence), {
            "hand_detected": True,
            "hand_count": len(hands),
            "detection_progress": float(progress),
            "should_auto_detect": bool(should_detect),
            "cooldown_active": self.current_detector.cooldown_counter > 0,
        }

    def draw_landmarks(self, frame):
        d = self.current_detector
        if d is None or d.last_results is None:
            return frame
        for hand in d.last_results.multi_hand_landmarks or []:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        return frame

    def get_instructions(self):
        return self.current_detector.get_instructions() if self.current_detector else []

    def reset(self):
        if self.current_detector:
            self.current_detector.reset_detection_state()

    def cleanup(self):
        self.asl_detector.cleanup()
        self.isl_detector.cleanup()
        self.tsl_detector.cleanup()
