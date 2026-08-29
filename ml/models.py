# ============================================================
# models.py
# Unified Sign Language Detection Models
#
# Supports:
#   ASL
#   ISL
#   TSL / TAMIL
#
# Render-safe model paths
# 84-feature Random Forest support
# MediaPipe Hands
# Automatic stable-frame detection
# ============================================================

import os
import cv2
import pickle
import traceback
import numpy as np
import mediapipe as mp


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def model_path(filename):
    """
    Always create an absolute path relative to models.py.

    This is important on Render because the current working
    directory is not something the application should depend on.
    """
    return os.path.join(BASE_DIR, filename)


# ============================================================
# COMMON 35-CLASS LABELS
# ============================================================
#
# Your verified RandomForest model has:
#
#   n_features_in_ = 84
#   classes_ = 0 ... 34
#
# The supplied detector code uses:
#
#   0-8   -> 1-9
#   9-34  -> A-Z
#   35    -> space
#
# However, your actual RandomForest has only 35 classes:
# 0-34.
#
# Therefore the 35 model classes represented here are:
#
# 0-8   = 1-9
# 9-34  = A-Z
#
# There is NO class 35 in this RandomForest.
#
# ============================================================

RANDOM_FOREST_LABELS = {
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
# MEDIAPIPE CONFIGURATION
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# ASL DETECTOR
# ============================================================

class ASLDetector:

    def __init__(self):

        self.model = None

        self.hands = None

        self.last_results = None

        self.model_loaded = False

        # ----------------------------------------------------
        # Render-safe path
        # ----------------------------------------------------

        self.model_path = model_path(
            "random_forest_isl_model.pkl"
        )

        # ----------------------------------------------------
        # 84 features
        #
        # 21 landmarks × 2 coordinates = 42
        # two hands = 84
        # ----------------------------------------------------

        self.expected_features = 84

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        self.labels_dict = RANDOM_FOREST_LABELS.copy()

        # ----------------------------------------------------
        # Detection state
        # ----------------------------------------------------

        self.last_prediction = None

        self.prediction_counter = 0

        self.required_consistent_frames = 30

        self.last_added_char = None

        self.cooldown_counter = 0

        self.cooldown_frames = 15

        self.confidence_threshold = 60.0

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        try:

            print("=" * 60)
            print("Loading ASL Random Forest model")
            print("=" * 60)

            print(
                "ASL model path:",
                self.model_path
            )

            # ------------------------------------------------
            # Check model file
            # ------------------------------------------------

            if not os.path.isfile(self.model_path):

                print(
                    "ERROR: ASL model file does not exist:"
                )

                print(
                    self.model_path
                )

                self.model_loaded = False

                return False

            print(
                "ASL model file found."
            )

            # ------------------------------------------------
            # Load pickle
            # ------------------------------------------------

            with open(
                self.model_path,
                "rb"
            ) as f:

                loaded_model = pickle.load(f)

            # ------------------------------------------------
            # Some projects save:
            #
            # pickle.dump(model)
            #
            # Others save:
            #
            # pickle.dump({"model": model})
            #
            # Support both.
            # ------------------------------------------------

            if isinstance(
                loaded_model,
                dict
            ):

                if "model" in loaded_model:

                    self.model = (
                        loaded_model["model"]
                    )

                elif "classifier" in loaded_model:

                    self.model = (
                        loaded_model["classifier"]
                    )

                else:

                    raise ValueError(
                        "Pickle dictionary does not contain "
                        "'model' or 'classifier'."
                    )

            else:

                self.model = loaded_model

            # ------------------------------------------------
            # Verify model
            # ------------------------------------------------

            print(
                "Loaded ASL model type:",
                type(self.model)
            )

            # ------------------------------------------------
            # Verify feature count
            # ------------------------------------------------

            actual_features = getattr(
                self.model,
                "n_features_in_",
                None
            )

            print(
                "Model features:",
                actual_features
            )

            if (
                actual_features is not None
                and actual_features != self.expected_features
            ):

                raise ValueError(
                    "ASL model expects "
                    f"{actual_features} features, "
                    f"but this detector creates "
                    f"{self.expected_features}."
                )

            # ------------------------------------------------
            # Verify classes
            # ------------------------------------------------

            classes = getattr(
                self.model,
                "classes_",
                None
            )

            print(
                "Model classes:",
                classes
            )

            if classes is not None:

                print(
                    "Number of classes:",
                    len(classes)
                )

            # ------------------------------------------------
            # MediaPipe
            # ------------------------------------------------

            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.reset_detection_state()

            self.model_loaded = True

            print(
                "ASL Random Forest loaded successfully."
            )

            print("=" * 60)

            return True

        except Exception as exc:

            print(
                "ERROR loading ASL model:"
            )

            print(
                repr(exc)
            )

            traceback.print_exc()

            self.model = None

            self.model_loaded = False

            return False

    # ========================================================
    # RESET
    # ========================================================

    def reset_detection_state(self):

        self.last_prediction = None

        self.prediction_counter = 0

        self.last_added_char = None

        self.cooldown_counter = 0

        self.last_results = None

    # ========================================================
    # LANDMARK PROCESSING
    # ========================================================

    def process_landmarks(
        self,
        multi_hand_landmarks
    ):

        """
        Converts MediaPipe hand landmarks to exactly 84 features.

        Hand 1:
            21 × x,y = 42

        Hand 2:
            21 × x,y = 42

        Total:
            84

        Final RandomForest input:
            shape = (1, 84)
        """

        first_hand = np.zeros(
            42,
            dtype=np.float32
        )

        second_hand = np.zeros(
            42,
            dtype=np.float32
        )

        if multi_hand_landmarks:

            for hand_idx, hand_landmarks in enumerate(
                multi_hand_landmarks
            ):

                values = []

                for landmark in hand_landmarks.landmark:

                    values.append(
                        float(landmark.x)
                    )

                    values.append(
                        float(landmark.y)
                    )

                values = np.asarray(
                    values[:42],
                    dtype=np.float32
                )

                if values.size < 42:

                    values = np.pad(
                        values,
                        (
                            0,
                            42 - values.size
                        ),
                        mode="constant"
                    )

                if hand_idx == 0:

                    first_hand = values

                elif hand_idx == 1:

                    second_hand = values

        features = np.concatenate(
            [
                first_hand,
                second_hand
            ]
        )

        if features.size != 84:

            raise ValueError(
                "Expected exactly 84 features, "
                f"got {features.size}"
            )

        return features.reshape(
            1,
            84
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        multi_hand_landmarks
    ):

        try:

            if (
                self.model is None
                or not self.model_loaded
            ):

                return "?", 0.0

            features = self.process_landmarks(
                multi_hand_landmarks
            )

            # ------------------------------------------------
            # RandomForest prediction
            # ------------------------------------------------

            predicted_index = int(
                self.model.predict(
                    features
                )[0]
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = 0.0

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model.predict_proba(
                        features
                    )[0]
                )

                if len(probabilities) > 0:

                    confidence = (
                        float(
                            np.max(
                                probabilities
                            )
                        )
                        * 100.0
                    )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            detected_char = (
                self.labels_dict.get(
                    predicted_index,
                    "?"
                )
            )

            if (
                confidence
                >= self.confidence_threshold
            ):

                return (
                    detected_char,
                    confidence
                )

            return (
                "?",
                confidence
            )

        except Exception as exc:

            print(
                "ASL prediction error:",
                repr(exc)
            )

            return (
                "?",
                0.0
            )

    # ========================================================
    # AUTO DETECTION
    # ========================================================

    def should_auto_detect(
        self,
        detected_char,
        confidence
    ):

        if self.cooldown_counter > 0:

            self.cooldown_counter -= 1

            return (
                False,
                0.0
            )

        if (
            detected_char == "?"
            or confidence < self.confidence_threshold
        ):

            self.last_prediction = None

            self.prediction_counter = 0

            return (
                False,
                0.0
            )

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
            1.0
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

            return (
                True,
                1.0
            )

        return (
            False,
            progress
        )

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    def get_instructions(self):

        return [
            "ASL Detection - Place hands inside the detection area",
            "Hold the same sign for 2-3 seconds",
            "The character is automatically accepted when stable",
            "Press ENTER to add the captured character",
            "Press Q when the sentence is complete",
        ]

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(self):

        try:

            if self.hands is not None:

                self.hands.close()

        except Exception:
            pass

        self.hands = None

        self.last_results = None


# ============================================================
# ISL DETECTOR
# ============================================================

class ISLDetector:

    def __init__(self):

        self.model = None

        self.hands = None

        self.last_results = None

        self.model_loaded = False

        # ----------------------------------------------------
        # Use the same verified 84-feature Random Forest.
        #
        # Change this filename ONLY if you have a separate
        # trained ISL model.
        # ----------------------------------------------------

        self.model_path = model_path(
            "random_forest_isl_model.pkl"
        )

        self.expected_features = 84

        self.isl_labels_dict = {
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

        self.last_prediction = None

        self.prediction_counter = 0

        self.required_consistent_frames = 30

        self.last_added_char = None

        self.cooldown_counter = 0

        self.cooldown_frames = 15

        self.confidence_threshold = 60.0

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        try:

            print("=" * 60)
            print("Loading ISL Random Forest model")
            print("=" * 60)

            print(
                "ISL model path:",
                self.model_path
            )

            if not os.path.isfile(
                self.model_path
            ):

                print(
                    "ERROR: ISL model file not found:"
                )

                print(
                    self.model_path
                )

                self.model_loaded = False

                return False

            with open(
                self.model_path,
                "rb"
            ) as f:

                loaded_model = pickle.load(f)

            if isinstance(
                loaded_model,
                dict
            ):

                if "model" in loaded_model:

                    self.model = (
                        loaded_model["model"]
                    )

                elif "classifier" in loaded_model:

                    self.model = (
                        loaded_model["classifier"]
                    )

                else:

                    raise ValueError(
                        "ISL pickle dictionary does not "
                        "contain 'model' or 'classifier'."
                    )

            else:

                self.model = loaded_model

            print(
                "Loaded ISL model type:",
                type(self.model)
            )

            actual_features = getattr(
                self.model,
                "n_features_in_",
                None
            )

            print(
                "ISL model features:",
                actual_features
            )

            if (
                actual_features is not None
                and actual_features != 84
            ):

                raise ValueError(
                    f"ISL model expects {actual_features} "
                    "features instead of 84."
                )

            classes = getattr(
                self.model,
                "classes_",
                None
            )

            print(
                "ISL model classes:",
                classes
            )

            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.reset_detection_state()

            self.model_loaded = True

            print(
                "ISL Random Forest loaded successfully."
            )

            print("=" * 60)

            return True

        except Exception as exc:

            print(
                "ERROR loading ISL model:",
                repr(exc)
            )

            traceback.print_exc()

            self.model = None

            self.model_loaded = False

            return False

    # ========================================================
    # RESET
    # ========================================================

    def reset_detection_state(self):

        self.last_prediction = None

        self.prediction_counter = 0

        self.last_added_char = None

        self.cooldown_counter = 0

        self.last_results = None

    # ========================================================
    # LANDMARK PROCESSING
    # ========================================================

    def process_isl_landmarks(
        self,
        multi_hand_landmarks
    ):

        first_hand = np.zeros(
            42,
            dtype=np.float32
        )

        second_hand = np.zeros(
            42,
            dtype=np.float32
        )

        if multi_hand_landmarks:

            for hand_idx, hand_landmarks in enumerate(
                multi_hand_landmarks
            ):

                values = []

                for landmark in hand_landmarks.landmark:

                    values.append(
                        float(landmark.x)
                    )

                    values.append(
                        float(landmark.y)
                    )

                values = np.asarray(
                    values[:42],
                    dtype=np.float32
                )

                if values.size < 42:

                    values = np.pad(
                        values,
                        (
                            0,
                            42 - values.size
                        ),
                        mode="constant"
                    )

                if hand_idx == 0:

                    first_hand = values

                elif hand_idx == 1:

                    second_hand = values

        features = np.concatenate(
            [
                first_hand,
                second_hand
            ]
        )

        if features.size != 84:

            raise ValueError(
                f"Expected 84 features, "
                f"got {features.size}"
            )

        return features.reshape(
            1,
            84
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        multi_hand_landmarks
    ):

        try:

            if (
                self.model is None
                or not self.model_loaded
            ):

                return "?", 0.0

            features = (
                self.process_isl_landmarks(
                    multi_hand_landmarks
                )
            )

            predicted_index = int(
                self.model.predict(
                    features
                )[0]
            )

            confidence = 0.0

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model.predict_proba(
                        features
                    )[0]
                )

                if len(probabilities) > 0:

                    confidence = (
                        float(
                            np.max(
                                probabilities
                            )
                        )
                        * 100.0
                    )

            detected_char = (
                self.isl_labels_dict.get(
                    predicted_index,
                    "?"
                )
            )

            if (
                confidence
                >= self.confidence_threshold
            ):

                return (
                    detected_char,
                    confidence
                )

            return (
                "?",
                confidence
            )

        except Exception as exc:

            print(
                "ISL prediction error:",
                repr(exc)
            )

            return (
                "?",
                0.0
            )

    # ========================================================
    # AUTO DETECTION
    # ========================================================

    def should_auto_detect(
        self,
        detected_char,
        confidence
    ):

        if self.cooldown_counter > 0:

            self.cooldown_counter -= 1

            return (
                False,
                0.0
            )

        if (
            detected_char == "?"
            or confidence < self.confidence_threshold
        ):

            self.last_prediction = None

            self.prediction_counter = 0

            return (
                False,
                0.0
            )

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
            1.0
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

            return (
                True,
                1.0
            )

        return (
            False,
            progress
        )

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    def get_instructions(self):

        return [
            "ISL Detection - Place hands in the detection area",
            "Hold the same sign for 2-3 seconds",
            "The character is automatically accepted when stable",
            "Press ENTER to add the captured character",
            "Press Q when the sentence is complete",
        ]

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(self):

        try:

            if self.hands is not None:

                self.hands.close()

        except Exception:
            pass

        self.hands = None

        self.last_results = None


# ============================================================
# TSL / TAMIL DETECTOR
# ============================================================

class TSLDetector:

    def __init__(self):

        self.model = None

        self.hands = None

        self.last_results = None

        self.model_loaded = False

        # ----------------------------------------------------
        # Optional TSL model
        #
        # If you have a separate TSL model, put it in ml/
        # and change this filename.
        # ----------------------------------------------------

        self.model_path = model_path(
            "random_forest_tsl_model.pkl"
        )

        self.expected_features = 84

        self.labels_dict = {}

        self.last_prediction = None

        self.prediction_counter = 0

        self.required_consistent_frames = 30

        self.last_added_char = None

        self.cooldown_counter = 0

        self.cooldown_frames = 15

        self.confidence_threshold = 60.0

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        try:

            print("=" * 60)
            print("Loading TSL / Tamil model")
            print("=" * 60)

            print(
                "TSL model path:",
                self.model_path
            )

            if not os.path.isfile(
                self.model_path
            ):

                print(
                    "TSL model file not found."
                )

                print(
                    "TSL is not configured yet."
                )

                self.model_loaded = False

                return False

            with open(
                self.model_path,
                "rb"
            ) as f:

                loaded_model = pickle.load(f)

            if isinstance(
                loaded_model,
                dict
            ):

                if "model" in loaded_model:

                    self.model = (
                        loaded_model["model"]
                    )

                elif "classifier" in loaded_model:

                    self.model = (
                        loaded_model["classifier"]
                    )

                else:

                    raise ValueError(
                        "TSL pickle does not contain "
                        "'model' or 'classifier'."
                    )

            else:

                self.model = loaded_model

            actual_features = getattr(
                self.model,
                "n_features_in_",
                None
            )

            print(
                "TSL model features:",
                actual_features
            )

            if (
                actual_features is not None
                and actual_features != 84
            ):

                raise ValueError(
                    f"TSL model expects {actual_features} "
                    "features instead of 84."
                )

            classes = getattr(
                self.model,
                "classes_",
                None
            )

            if classes is not None:

                print(
                    "TSL classes:",
                    classes
                )

                # --------------------------------------------
                # Default numeric mapping.
                #
                # Replace with your actual Tamil label mapping
                # if your trained model uses one.
                # --------------------------------------------

                self.labels_dict = {
                    int(c): str(c)
                    for c in classes
                }

            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.reset_detection_state()

            self.model_loaded = True

            print(
                "TSL model loaded successfully."
            )

            print("=" * 60)

            return True

        except Exception as exc:

            print(
                "ERROR loading TSL model:",
                repr(exc)
            )

            traceback.print_exc()

            self.model = None

            self.model_loaded = False

            return False

    # ========================================================
    # RESET
    # ========================================================

    def reset_detection_state(self):

        self.last_prediction = None

        self.prediction_counter = 0

        self.last_added_char = None

        self.cooldown_counter = 0

        self.last_results = None

    # ========================================================
    # FEATURES
    # ========================================================

    def process_landmarks(
        self,
        multi_hand_landmarks
    ):

        first_hand = np.zeros(
            42,
            dtype=np.float32
        )

        second_hand = np.zeros(
            42,
            dtype=np.float32
        )

        if multi_hand_landmarks:

            for hand_idx, hand_landmarks in enumerate(
                multi_hand_landmarks
            ):

                values = []

                for landmark in hand_landmarks.landmark:

                    values.extend(
                        [
                            float(landmark.x),
                            float(landmark.y)
                        ]
                    )

                values = np.asarray(
                    values[:42],
                    dtype=np.float32
                )

                if values.size < 42:

                    values = np.pad(
                        values,
                        (
                            0,
                            42 - values.size
                        ),
                        mode="constant"
                    )

                if hand_idx == 0:

                    first_hand = values

                elif hand_idx == 1:

                    second_hand = values

        features = np.concatenate(
            [
                first_hand,
                second_hand
            ]
        )

        return features.reshape(
            1,
            84
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        multi_hand_landmarks
    ):

        try:

            if (
                self.model is None
                or not self.model_loaded
            ):

                return "?", 0.0

            features = (
                self.process_landmarks(
                    multi_hand_landmarks
                )
            )

            predicted_index = int(
                self.model.predict(
                    features
                )[0]
            )

            confidence = 0.0

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model.predict_proba(
                        features
                    )[0]
                )

                confidence = (
                    float(
                        np.max(
                            probabilities
                        )
                    )
                    * 100.0
                )

            detected_char = (
                self.labels_dict.get(
                    predicted_index,
                    str(predicted_index)
                )
            )

            if (
                confidence
                >= self.confidence_threshold
            ):

                return (
                    detected_char,
                    confidence
                )

            return (
                "?",
                confidence
            )

        except Exception as exc:

            print(
                "TSL prediction error:",
                repr(exc)
            )

            return (
                "?",
                0.0
            )

    # ========================================================
    # AUTO DETECTION
    # ========================================================

    def should_auto_detect(
        self,
        detected_char,
        confidence
    ):

        if self.cooldown_counter > 0:

            self.cooldown_counter -= 1

            return (
                False,
                0.0
            )

        if (
            detected_char == "?"
            or confidence < self.confidence_threshold
        ):

            self.last_prediction = None

            self.prediction_counter = 0

            return (
                False,
                0.0
            )

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
            1.0
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

            return (
                True,
                1.0
            )

        return (
            False,
            progress
        )

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    def get_instructions(self):

        return [
            "Tamil Sign Language Detection",
            "Place your hands inside the detection area",
            "Hold the same sign for 2-3 seconds",
        ]

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(self):

        try:

            if self.hands is not None:

                self.hands.close()

        except Exception:
            pass

        self.hands = None

        self.last_results = None


# ============================================================
# UNIFIED SIGN LANGUAGE DETECTOR
# ============================================================

class UnifiedSignLanguageDetector:

    def __init__(self):

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # These classes MUST exist before they are instantiated.
        #
        # This fixes your Render error:
        #
        # NameError:
        # name 'ISLDetector' is not defined
        # ----------------------------------------------------

        self.asl_detector = ASLDetector()

        self.isl_detector = ISLDetector()

        self.tsl_detector = TSLDetector()

        self.current_detector = None

        self.language = None

        self.cap = None

        self.session_start_time = None

    # ========================================================
    # INITIALIZE
    # ========================================================

    def initialize(
        self,
        language="ASL"
    ):

        self.language = (
            str(language)
            .upper()
            .strip()
        )

        print(
            f"Initializing detector for: "
            f"{self.language}"
        )

        # ----------------------------------------------------
        # ASL
        # ----------------------------------------------------

        if self.language == "ASL":

            self.current_detector = (
                self.asl_detector
            )

            return (
                self.asl_detector.load_model()
            )

        # ----------------------------------------------------
        # ISL
        # ----------------------------------------------------

        if self.language == "ISL":

            self.current_detector = (
                self.isl_detector
            )

            return (
                self.isl_detector.load_model()
            )

        # ----------------------------------------------------
        # Tamil / TSL
        # ----------------------------------------------------

        if self.language in (
            "TSL",
            "TAMIL",
            "TAMIL_SIGN_LANGUAGE"
        ):

            self.current_detector = (
                self.tsl_detector
            )

            return (
                self.tsl_detector.load_model()
            )

        # ----------------------------------------------------
        # Unsupported
        # ----------------------------------------------------

        print(
            f"Unsupported language: "
            f"{self.language}"
        )

        self.current_detector = None

        return False

    # ========================================================
    # DETECTOR INFO
    # ========================================================

    def get_detector_info(self):

        if self.current_detector is None:

            return {}

        detector = self.current_detector

        # ----------------------------------------------------
        # ASL
        # ----------------------------------------------------

        if self.language == "ASL":

            return {

                "language":
                    "ASL",

                "model_type":
                    "Random Forest Classifier",

                "model_path":
                    self.asl_detector.model_path,

                "model_exists":
                    os.path.isfile(
                        self.asl_detector.model_path
                    ),

                "model_loaded":
                    self.asl_detector.model_loaded,

                "hands_supported":
                    "Two hands",

                "classes":
                    len(
                        self.asl_detector
                        .labels_dict
                    ),

                "supported_characters":
                    list(
                        self.asl_detector
                        .labels_dict
                        .values()
                    ),

                "input_features":
                    84,

                "input_shape":
                    "(1, 84)",

                "auto_detection":
                    True,

                "confidence_threshold":
                    self.asl_detector
                    .confidence_threshold,

                "consistent_frames_required":
                    self.asl_detector
                    .required_consistent_frames,

            }

        # ----------------------------------------------------
        # ISL
        # ----------------------------------------------------

        if self.language == "ISL":

            return {

                "language":
                    "ISL",

                "model_type":
                    "Random Forest Classifier",

                "model_path":
                    self.isl_detector.model_path,

                "model_exists":
                    os.path.isfile(
                        self.isl_detector.model_path
                    ),

                "model_loaded":
                    self.isl_detector.model_loaded,

                "hands_supported":
                    "Two hands",

                "classes":
                    len(
                        self.isl_detector
                        .isl_labels_dict
                    ),

                "supported_characters":
                    list(
                        self.isl_detector
                        .isl_labels_dict
                        .values()
                    ),

                "input_features":
                    84,

                "input_shape":
                    "(1, 84)",

                "auto_detection":
                    True,

                "confidence_threshold":
                    self.isl_detector
                    .confidence_threshold,

                "consistent_frames_required":
                    self.isl_detector
                    .required_consistent_frames,

            }

        # ----------------------------------------------------
        # TSL
        # ----------------------------------------------------

        if self.language in (
            "TSL",
            "TAMIL",
            "TAMIL_SIGN_LANGUAGE"
        ):

            return {

                "language":
                    "TSL / Tamil",

                "model_type":
                    "Random Forest Classifier",

                "model_path":
                    self.tsl_detector.model_path,

                "model_exists":
                    os.path.isfile(
                        self.tsl_detector.model_path
                    ),

                "model_loaded":
                    self.tsl_detector.model_loaded,

                "hands_supported":
                    "Two hands",

                "classes":
                    len(
                        self.tsl_detector
                        .labels_dict
                    ),

                "supported_characters":
                    list(
                        self.tsl_detector
                        .labels_dict
                        .values()
                    ),

                "input_features":
                    84,

                "input_shape":
                    "(1, 84)",

                "auto_detection":
                    True,

                "confidence_threshold":
                    self.tsl_detector
                    .confidence_threshold,

                "consistent_frames_required":
                    self.tsl_detector
                    .required_consistent_frames,

            }

        return {}

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame
    ):

        if self.current_detector is None:

            return (
                "?",
                0.0,
                {
                    "hand_detected": False,
                    "detection_progress": 0.0,
                    "should_auto_detect": False,
                    "error":
                        "Detector is not initialized"
                }
            )

        if (
            not self.current_detector.model_loaded
            or self.current_detector.hands is None
        ):

            return (
                "?",
                0.0,
                {
                    "hand_detected": False,
                    "detection_progress": 0.0,
                    "should_auto_detect": False,
                    "error":
                        "Model is not loaded"
                }
            )

        try:

            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # MediaPipe
            # ------------------------------------------------

            results = (
                self.current_detector
                .hands
                .process(
                    frame_rgb
                )
            )

            self.current_detector.last_results = (
                results
            )

            # ------------------------------------------------
            # No hand
            # ------------------------------------------------

            if not results.multi_hand_landmarks:

                self.current_detector.last_prediction = (
                    None
                )

                self.current_detector.prediction_counter = (
                    0
                )

                return (
                    "?",
                    0.0,
                    {
                        "hand_detected": False,
                        "detection_progress": 0.0,
                        "should_auto_detect": False,
                        "hand_count": 0,
                    }
                )

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            detected_char, confidence = (
                self.current_detector.predict(
                    results.multi_hand_landmarks
                )
            )

            # ------------------------------------------------
            # Stable-frame detection
            # ------------------------------------------------

            should_detect, progress = (
                self.current_detector
                .should_auto_detect(
                    detected_char,
                    confidence
                )
            )

            return (
                detected_char,
                confidence,
                {
                    "hand_detected": True,

                    "hand_count":
                        len(
                            results
                            .multi_hand_landmarks
                        ),

                    "detection_progress":
                        progress,

                    "should_auto_detect":
                        should_detect,

                    "cooldown_active":
                        self.current_detector
                        .cooldown_counter > 0,
                }
            )

        except Exception as exc:

            print(
                "Frame processing error:",
                repr(exc)
            )

            traceback.print_exc()

            return (
                "?",
                0.0,
                {
                    "hand_detected": False,
                    "detection_progress": 0.0,
                    "should_auto_detect": False,
                    "error": str(exc)
                }
            )

    # ========================================================
    # DRAW LANDMARKS
    # ========================================================

    def draw_landmarks(
        self,
        frame
    ):

        if self.current_detector is None:

            return frame

        results = (
            self.current_detector
            .last_results
        )

        if (
            results is None
            or not results.multi_hand_landmarks
        ):

            return frame

        for hand_landmarks in (
            results.multi_hand_landmarks
        ):

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,

                mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2,
                    circle_radius=4,
                ),

                mp_drawing.DrawingSpec(
                    color=(255, 255, 255),
                    thickness=2,
                ),
            )

        return frame

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    def get_instructions(self):

        if self.current_detector:

            return (
                self.current_detector
                .get_instructions()
            )

        return []

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        if self.current_detector:

            self.current_detector.reset_detection_state()

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(self):

        try:

            if self.cap is not None:

                self.cap.release()

        except Exception:
            pass

        self.cap = None

        try:

            if self.asl_detector:

                self.asl_detector.cleanup()

        except Exception:
            pass

        try:

            if self.isl_detector:

                self.isl_detector.cleanup()

        except Exception:
            pass

        try:

            if self.tsl_detector:

                self.tsl_detector.cleanup()

        except Exception:
            pass

        try:

            cv2.destroyAllWindows()

        except Exception:
            pass


# ============================================================
# TEST WHEN RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Unified Sign Language Detector - Model Test")
    print("=" * 70)

    print(
        "BASE_DIR:",
        BASE_DIR
    )

    print(
        "ASL/ISL model:",
        model_path(
            "random_forest_isl_model.pkl"
        )
    )

    print(
        "ASL/ISL model exists:",
        os.path.isfile(
            model_path(
                "random_forest_isl_model.pkl"
            )
        )
    )

    detector = UnifiedSignLanguageDetector()

    print()
    print("Testing ASL initialization...")

    asl_ok = detector.initialize(
        "ASL"
    )

    print(
        "ASL loaded:",
        asl_ok
    )

    if asl_ok:

        print(
            detector.get_detector_info()
        )

    detector.cleanup()

    print()
    print("=" * 70)
    print("Model test complete")
    print("=" * 70)