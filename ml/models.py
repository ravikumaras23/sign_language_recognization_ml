# ============================================================
# models.py
# Unified ASL + ISL + TSL Sign Language Detection
#
# Browser camera -> FastAPI -> MediaPipe -> ML model
#
# Models:
#   ASL : random_forest_isl_model.pkl
#   ISL : final_lstm_hand_model11.keras
#   TSL : best_lstm_model.keras
#
# Important:
#   - Browser owns webcam
#   - Backend NEVER opens cv2.VideoCapture(0)
#   - All model paths are relative to this file
#   - Random Forest expects exactly 84 features
#   - ISL/TSL LSTM expects (1, 1, 84)
# ============================================================

import os
import json
import pickle
import traceback
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import mediapipe as mp

from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def model_path(filename: str) -> str:
    """
    Render-safe absolute path.

    models.py is inside:
        /opt/render/project/src/ml/

    Therefore models are expected inside the same ml directory.
    """
    return os.path.join(BASE_DIR, filename)


# ============================================================
# COMMON 35-CLASS LABELS
# Verified Random Forest:
#
# classes_ = 0 ... 34
# n_features_in_ = 84
#
# 0-8  -> 1-9
# 9-34 -> A-Z
# ============================================================

RF_LABELS = {
    0: "1",
    1: "2",
    2: "3",
    3: "4",
    4: "5",
    5: "6",
    6: "7",
    7: "8",
    8: "9",

    9: "A",
    10: "B",
    11: "C",
    12: "D",
    13: "E",
    14: "F",
    15: "G",
    16: "H",
    17: "I",
    18: "J",
    19: "K",
    20: "L",
    21: "M",
    22: "N",
    23: "O",
    24: "P",
    25: "Q",
    26: "R",
    27: "S",
    28: "T",
    29: "U",
    30: "V",
    31: "W",
    32: "X",
    33: "Y",
    34: "Z",
}


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def create_hands(max_num_hands: int):
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# ============================================================
# LANDMARK FEATURE EXTRACTION
# ============================================================

def hand_xy(hand_landmarks) -> np.ndarray:
    """
    Extract:
        21 landmarks * x,y = 42 values
    """

    values = []

    for landmark in hand_landmarks.landmark[:21]:
        values.append(float(landmark.x))
        values.append(float(landmark.y))

    values = np.asarray(values, dtype=np.float32)

    if values.size < 42:
        values = np.pad(
            values,
            (0, 42 - values.size),
            mode="constant",
        )

    return values[:42]


def two_hand_features(multi_hand_landmarks) -> np.ndarray:
    """
    Create exactly 84 features:

        first hand  = 42
        second hand = 42

        total = 84

    This preserves the feature format used by the existing
    ISL/TSL code.
    """

    first_hand = np.zeros(42, dtype=np.float32)
    second_hand = np.zeros(42, dtype=np.float32)

    if multi_hand_landmarks:

        for index, hand in enumerate(
            multi_hand_landmarks[:2]
        ):

            values = hand_xy(hand)

            if index == 0:
                first_hand = values

            elif index == 1:
                second_hand = values

    features = np.concatenate(
        [
            first_hand,
            second_hand,
        ]
    ).astype(np.float32)

    if features.size != 84:
        raise ValueError(
            f"Expected 84 features, got {features.size}"
        )

    return features


def two_hand_lstm_features(multi_hand_landmarks):
    """
    Convert 84 features to:

        (batch=1, time_steps=1, features=84)
    """

    features = two_hand_features(
        multi_hand_landmarks
    )

    return features.reshape(
        1,
        1,
        84,
    )


def extract_pickle_model(obj):
    """
    Supports both:

        pickle.dump(model)

    and:

        pickle.dump({"model": model})
    """

    if isinstance(obj, dict):

        for key in (
            "model",
            "classifier",
            "estimator",
        ):

            if key in obj:
                return obj[key]

        raise ValueError(
            "Pickle dictionary does not contain "
            "'model', 'classifier', or 'estimator'."
        )

    return obj


# ============================================================
# STABLE DETECTION
# ============================================================

class StableDetection:

    def __init__(
        self,
        confidence_threshold: float = 60.0,
        required_frames: int = 3,
        cooldown_frames: int = 4,
    ):

        self.confidence_threshold = (
            confidence_threshold
        )

        self.required_consistent_frames = (
            required_frames
        )

        self.cooldown_frames = (
            cooldown_frames
        )

        self.reset_detection_state()

    def reset_detection_state(self):

        self.last_prediction = None
        self.prediction_counter = 0
        self.cooldown_counter = 0
        self.last_added_char = None

    def should_auto_detect(
        self,
        detected_char: str,
        confidence: float,
    ):

        if self.cooldown_counter > 0:

            self.cooldown_counter -= 1

            return False, 0.0

        if (
            detected_char == "?"
            or confidence < self.confidence_threshold
        ):

            self.last_prediction = None
            self.prediction_counter = 0

            return False, 0.0

        if (
            self.last_prediction
            == detected_char
        ):

            self.prediction_counter += 1

        else:

            self.last_prediction = (
                detected_char
            )

            self.prediction_counter = 1

        progress = min(
            self.prediction_counter
            / self.required_consistent_frames,
            1.0,
        )

        if (
            self.prediction_counter
            >= self.required_consistent_frames
        ):

            self.prediction_counter = 0

            self.cooldown_counter = (
                self.cooldown_frames
            )

            self.last_added_char = (
                detected_char
            )

            return True, 1.0

        return False, progress


# ============================================================
# ASL
# ============================================================

class ASLDetector(StableDetection):

    def __init__(self):

        super().__init__(
            confidence_threshold=60.0,
            required_frames=3,
            cooldown_frames=4,
        )

        self.model = None
        self.hands = None
        self.last_results = None
        self.model_loaded = False

        # Existing verified 84-feature RF.
        # Filename is retained exactly as it exists in repo.
        self.model_path = model_path(
            "random_forest_isl_model.pkl"
        )

        self.labels_dict = RF_LABELS.copy()

        self.expected_features = 84

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    def load_model(self):

        # Already loaded.
        if (
            self.model_loaded
            and self.model is not None
            and self.hands is not None
        ):
            return True

        try:

            print("=" * 60)
            print("[ASL] Loading Random Forest")
            print("[ASL] Path:", self.model_path)
            print("=" * 60)

            if not os.path.isfile(
                self.model_path
            ):

                raise FileNotFoundError(
                    self.model_path
                )

            with open(
                self.model_path,
                "rb",
            ) as file:

                loaded = pickle.load(file)

            self.model = extract_pickle_model(
                loaded
            )

            print(
                "[ASL] Model type:",
                type(self.model),
            )

            actual_features = getattr(
                self.model,
                "n_features_in_",
                None,
            )

            print(
                "[ASL] n_features_in_:",
                actual_features,
            )

            if actual_features != 84:

                raise ValueError(
                    "ASL Random Forest must have "
                    f"84 features, but has "
                    f"{actual_features}"
                )

            classes = getattr(
                self.model,
                "classes_",
                None,
            )

            print(
                "[ASL] classes_:",
                classes,
            )

            if classes is not None:

                print(
                    "[ASL] class count:",
                    len(classes),
                )

            self.hands = create_hands(
                max_num_hands=2
            )

            self.reset_detection_state()

            self.model_loaded = True

            print(
                "[ASL] Model loaded successfully."
            )

            return True

        except Exception as exc:

            print(
                "[ASL] MODEL LOAD ERROR:",
                repr(exc),
            )

            traceback.print_exc()

            self.model = None
            self.model_loaded = False

            return False

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(
        self,
        multi_hand_landmarks,
    ):

        try:

            if (
                not self.model_loaded
                or self.model is None
                or not multi_hand_landmarks
            ):

                return "?", 0.0

            features = two_hand_features(
                multi_hand_landmarks
            )

            prediction = self.model.predict(
                features.reshape(1, 84)
            )

            predicted_index = int(
                prediction[0]
            )

            confidence = 0.0

            if hasattr(
                self.model,
                "predict_proba",
            ):

                probabilities = np.asarray(
                    self.model.predict_proba(
                        features.reshape(1, 84)
                    )[0]
                )

                if probabilities.size:

                    confidence = (
                        float(
                            np.max(
                                probabilities
                            )
                        )
                        * 100.0
                    )

            character = (
                self.labels_dict.get(
                    predicted_index,
                    "?",
                )
            )

            if (
                confidence
                >= self.confidence_threshold
            ):

                return (
                    character,
                    confidence,
                )

            return (
                "?",
                confidence,
            )

        except Exception as exc:

            print(
                "[ASL] prediction error:",
                repr(exc),
            )

            return "?", 0.0

    def get_instructions(self):

        return [
            "ASL - Place one hand inside the detection area",
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


# ============================================================
# ISL
# ============================================================

class ISLDetector(StableDetection):

    def __init__(self):

        super().__init__(
            confidence_threshold=60.0,
            required_frames=3,
            cooldown_frames=4,
        )

        self.model = None
        self.hands = None
        self.last_results = None
        self.model_loaded = False

        self.model_path = model_path(
            "final_lstm_hand_model11.keras"
        )

        self.labels_dict = RF_LABELS.copy()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    def load_model(self):

        if (
            self.model_loaded
            and self.model is not None
            and self.hands is not None
        ):
            return True

        try:

            print("=" * 60)
            print("[ISL] Loading LSTM")
            print("[ISL] Path:", self.model_path)
            print("=" * 60)

            if not os.path.isfile(
                self.model_path
            ):

                raise FileNotFoundError(
                    self.model_path
                )

            self.model = load_model(
                self.model_path,
                compile=False,
            )

            print(
                "[ISL] Input:",
                self.model.input_shape,
            )

            print(
                "[ISL] Output:",
                self.model.output_shape,
            )

            self.hands = create_hands(
                max_num_hands=2
            )

            self.reset_detection_state()

            self.model_loaded = True

            print(
                "[ISL] Model loaded successfully."
            )

            return True

        except Exception as exc:

            print(
                "[ISL] MODEL LOAD ERROR:",
                repr(exc),
            )

            traceback.print_exc()

            self.model = None
            self.model_loaded = False

            return False

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(
        self,
        multi_hand_landmarks,
    ):

        try:

            if (
                not self.model_loaded
                or self.model is None
                or not multi_hand_landmarks
            ):

                return "?", 0.0

            features = two_hand_lstm_features(
                multi_hand_landmarks
            )

            prediction = np.asarray(
                self.model.predict(
                    features,
                    verbose=0,
                )
            ).flatten()

            if prediction.size == 0:

                return "?", 0.0

            predicted_index = int(
                np.argmax(prediction)
            )

            confidence = (
                float(
                    prediction[
                        predicted_index
                    ]
                )
                * 100.0
            )

            character = (
                self.labels_dict.get(
                    predicted_index,
                    "?",
                )
            )

            if (
                confidence
                >= self.confidence_threshold
            ):

                return (
                    character,
                    confidence,
                )

            return (
                "?",
                confidence,
            )

        except Exception as exc:

            print(
                "[ISL] prediction error:",
                repr(exc),
            )

            return "?", 0.0

    def get_instructions(self):

        return [
            "ISL - Place your hand(s) inside the detection area",
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


# ============================================================
# TSL / TAMIL
# ============================================================

class TSLDetector(StableDetection):

    def __init__(self):

        super().__init__(
            confidence_threshold=60.0,
            required_frames=3,
            cooldown_frames=4,
        )

        self.model = None
        self.hands = None
        self.last_results = None
        self.model_loaded = False

        self.model_path = model_path(
            "best_lstm_model.keras"
        )

        self.labels_path = model_path(
            "tamil_labels.json"
        )

        self.labels_dict: Dict[int, str] = {}

        self.label_details: Dict[
            int,
            Dict[str, Any]
        ] = {}

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    def load_model(self):

        if (
            self.model_loaded
            and self.model is not None
            and self.hands is not None
            and self.labels_dict
        ):
            return True

        try:

            print("=" * 60)
            print("[TSL] Loading Tamil LSTM")
            print("[TSL] Model:", self.model_path)
            print("[TSL] Labels:", self.labels_path)
            print("=" * 60)

            if not os.path.isfile(
                self.model_path
            ):

                raise FileNotFoundError(
                    self.model_path
                )

            if not os.path.isfile(
                self.labels_path
            ):

                raise FileNotFoundError(
                    self.labels_path
                )

            self.model = load_model(
                self.model_path,
                compile=False,
            )

            with open(
                self.labels_path,
                "r",
                encoding="utf-8",
            ) as file:

                raw_labels = json.load(file)

            self.label_details = {}

            for key, value in raw_labels.items():

                index = int(key)

                if isinstance(
                    value,
                    dict,
                ):

                    tamil_character = str(
                        value.get(
                            "tamil",
                            "?",
                        )
                    )

                    self.label_details[
                        index
                    ] = value

                else:

                    tamil_character = str(
                        value
                    )

                    self.label_details[
                        index
                    ] = {
                        "tamil":
                            tamil_character
                    }

                self.labels_dict[
                    index
                ] = tamil_character

            print(
                "[TSL] Label count:",
                len(self.labels_dict),
            )

            print(
                "[TSL] Input:",
                self.model.input_shape,
            )

            print(
                "[TSL] Output:",
                self.model.output_shape,
            )

            self.hands = create_hands(
                max_num_hands=2
            )

            self.reset_detection_state()

            self.model_loaded = True

            print(
                "[TSL] Model loaded successfully."
            )

            return True

        except Exception as exc:

            print(
                "[TSL] MODEL LOAD ERROR:",
                repr(exc),
            )

            traceback.print_exc()

            self.model = None
            self.model_loaded = False

            return False

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(
        self,
        multi_hand_landmarks,
    ):

        try:

            if (
                not self.model_loaded
                or self.model is None
                or not multi_hand_landmarks
            ):

                return "?", 0.0

            features = two_hand_lstm_features(
                multi_hand_landmarks
            )

            prediction = np.asarray(
                self.model.predict(
                    features,
                    verbose=0,
                )
            ).flatten()

            if prediction.size == 0:

                return "?", 0.0

            predicted_index = int(
                np.argmax(prediction)
            )

            confidence = (
                float(
                    prediction[
                        predicted_index
                    ]
                )
                * 100.0
            )

            character = (
                self.labels_dict.get(
                    predicted_index,
                    "?",
                )
            )

            # Ignore the Background class.
            if (
                character == "Background"
            ):

                return "?", confidence

            if (
                confidence
                >= self.confidence_threshold
            ):

                return (
                    character,
                    confidence,
                )

            return (
                "?",
                confidence,
            )

        except Exception as exc:

            print(
                "[TSL] prediction error:",
                repr(exc),
            )

            return "?", 0.0

    def get_instructions(self):

        return [
            "TSL / Tamil - Place your hand(s) inside the detection area",
            "Hold the same Tamil sign steadily",
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


# ============================================================
# UNIFIED DETECTOR
# ============================================================

class UnifiedSignLanguageDetector:

    def __init__(self):

        self.asl_detector = ASLDetector()
        self.isl_detector = ISLDetector()
        self.tsl_detector = TSLDetector()

        self.current_detector = None
        self.language = None
        self.session_start_time = None

    # --------------------------------------------------------
    # NORMALIZE LANGUAGE
    # --------------------------------------------------------

    @staticmethod
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

        return value

    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    def initialize(
        self,
        language: str = "ASL",
    ) -> bool:

        language = (
            self.normalize_language(
                language
            )
        )

        if language == "ASL":

            detector = (
                self.asl_detector
            )

        elif language == "ISL":

            detector = (
                self.isl_detector
            )

        elif language == "TSL":

            detector = (
                self.tsl_detector
            )

        else:

            print(
                "[UNIFIED] Unsupported language:",
                language,
            )

            self.current_detector = None
            self.language = None

            return False

        self.language = language
        self.current_detector = detector

        # IMPORTANT:
        # load_model() is cached by every detector.
        # Therefore repeated /start_detection requests
        # do NOT reload TensorFlow models.
        return detector.load_model()

    # --------------------------------------------------------
    # PROCESS FRAME
    # --------------------------------------------------------

    def process_frame(
        self,
        frame,
    ) -> Tuple[
        str,
        float,
        Dict[str, Any],
    ]:

        detector = (
            self.current_detector
        )

        if (
            detector is None
            or not detector.model_loaded
            or detector.hands is None
        ):

            return (
                "?",
                0.0,
                {
                    "hand_detected": False,
                    "hand_count": 0,
                    "detection_progress": 0.0,
                    "should_auto_detect": False,
                },
            )

        try:

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            results = detector.hands.process(
                frame_rgb
            )

            detector.last_results = (
                results
            )

            hands = list(
                results.multi_hand_landmarks
                or []
            )

            if not hands:

                detector.last_prediction = None
                detector.prediction_counter = 0

                return (
                    "?",
                    0.0,
                    {
                        "hand_detected": False,
                        "hand_count": 0,
                        "detection_progress": 0.0,
                        "should_auto_detect": False,
                        "cooldown_active":
                            detector.cooldown_counter > 0,
                    },
                )

            detected_char, confidence = (
                detector.predict(hands)
            )

            should_detect, progress = (
                detector.should_auto_detect(
                    detected_char,
                    confidence,
                )
            )

            return (
                detected_char,
                float(confidence),
                {
                    "hand_detected": True,
                    "hand_count": len(hands),
                    "detection_progress":
                        float(progress),
                    "should_auto_detect":
                        bool(should_detect),
                    "cooldown_active":
                        detector.cooldown_counter > 0,
                },
            )

        except Exception as exc:

            print(
                "[UNIFIED] Frame processing error:",
                repr(exc),
            )

            traceback.print_exc()

            return (
                "?",
                0.0,
                {
                    "hand_detected": False,
                    "hand_count": 0,
                    "detection_progress": 0.0,
                    "should_auto_detect": False,
                    "error": str(exc),
                },
            )

    # --------------------------------------------------------
    # DRAW LANDMARKS
    # --------------------------------------------------------

    def draw_landmarks(
        self,
        frame,
    ):

        detector = (
            self.current_detector
        )

        if (
            detector is None
            or detector.last_results is None
        ):

            return frame

        for hand in (
            detector.last_results
            .multi_hand_landmarks
            or []
        ):

            mp_drawing.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
            )

        return frame

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    def get_detector_info(self):

        detector = (
            self.current_detector
        )

        if detector is None:

            return {
                "language": self.language,
                "model_loaded": False,
            }

        info = {
            "language":
                self.language,

            "model_path":
                getattr(
                    detector,
                    "model_path",
                    None,
                ),

            "model_exists":
                os.path.isfile(
                    getattr(
                        detector,
                        "model_path",
                        "",
                    )
                ),

            "model_loaded":
                bool(
                    detector.model_loaded
                ),

            "model_type":
                (
                    type(
                        detector.model
                    ).__name__
                    if detector.model
                    is not None
                    else None
                ),

            "confidence_threshold":
                detector.confidence_threshold,

            "consistent_frames_required":
                detector.required_consistent_frames,

            "input_features":
                84,

            "hands_supported":
                1
                if self.language == "ASL"
                else 2,
        }

        if (
            self.language == "ASL"
        ):

            info["input_shape"] = "(1, 84)"
            info["classes"] = len(
                detector.labels_dict
            )
            info["supported_characters"] = list(
                detector.labels_dict.values()
            )

        elif (
            self.language == "ISL"
        ):

            info["input_shape"] = "(1, 1, 84)"
            info["classes"] = len(
                detector.labels_dict
            )
            info["supported_characters"] = list(
                detector.labels_dict.values()
            )

        elif (
            self.language == "TSL"
        ):

            info["input_shape"] = "(1, 1, 84)"
            info["classes"] = len(
                detector.labels_dict
            )
            info["supported_characters"] = list(
                detector.labels_dict.values()
            )

        return info

    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    def get_instructions(self):

        if (
            self.current_detector
            is None
        ):

            return []

        return (
            self.current_detector
            .get_instructions()
        )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        if self.current_detector:

            self.current_detector.reset_detection_state()

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    def cleanup(self):

        self.asl_detector.cleanup()
        self.isl_detector.cleanup()
        self.tsl_detector.cleanup()