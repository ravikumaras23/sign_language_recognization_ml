import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

from keras.models import load_model


# ============================================================
# ASL DETECTOR
# ============================================================

class ASLDetector:

    def __init__(self):

        self.model = None

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = None

        # ----------------------------------------------------
        # ASL LABELS
        # ----------------------------------------------------

        self.asl_labels_dict = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
            5: "F",
            6: "G",
            7: "H",
            8: "I",
            9: "J",
            10: "K",
            11: "L",
            12: "M",
            13: "N",
            14: "O",
            15: "P",
            16: "Q",
            17: "R",
            18: "S",
            19: "T",
            20: "U",
            21: "V",
            22: "W",
            23: "X",
            24: "Y",
            25: "Z",
        }

        # ----------------------------------------------------
        # MODEL PATH
        #
        # models.py is inside:
        #
        # ml/
        #   models.py
        #   random_forest_asl_model.pkl
        #
        # ----------------------------------------------------

        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.model_path = os.path.join(
            base_dir,
            "random_forest_asl_model.pkl",
        )

        # ----------------------------------------------------
        # AUTO DETECTION
        # ----------------------------------------------------

        self.last_prediction = None

        self.prediction_counter = 0

        # Around 1.5-2 seconds depending on FPS
        self.required_consistent_frames = 25

        self.last_added_char = None

        self.cooldown_counter = 0

        # Around 1 second at ~15 FPS
        self.cooldown_frames = 15

        # Minimum confidence
        self.confidence_threshold = 60.0

        self.last_results = None

    # ========================================================
    # LOAD ASL MODEL
    # ========================================================

    def load_model(self):

        try:

            # ------------------------------------------------
            # Check model file
            # ------------------------------------------------

            if not os.path.exists(
                self.model_path
            ):

                print(
                    "ASL model file not found:"
                )

                print(
                    self.model_path
                )

                return False

            print(
                "=========================================="
            )

            print(
                "Loading ASL Random Forest model:"
            )

            print(
                self.model_path
            )

            # ------------------------------------------------
            # Load pickle
            # ------------------------------------------------

            with open(
                self.model_path,
                "rb",
            ) as f:

                loaded_model = pickle.load(f)

            # ------------------------------------------------
            # Support both:
            #
            # Direct model:
            # RandomForestClassifier(...)
            #
            # OR:
            #
            # {
            #     "model": RandomForestClassifier(...)
            # }
            # ------------------------------------------------

            if (
                isinstance(
                    loaded_model,
                    dict,
                )
                and "model" in loaded_model
            ):

                self.model = (
                    loaded_model["model"]
                )

            else:

                self.model = loaded_model

            # ------------------------------------------------
            # Verify model
            # ------------------------------------------------

            if self.model is None:

                print(
                    "ASL model loaded as None."
                )

                return False

            print(
                "ASL model loaded successfully."
            )

            print(
                "ASL model type:",
                type(self.model).__name__,
            )

            # ------------------------------------------------
            # Print model classes if available
            # ------------------------------------------------

            if hasattr(
                self.model,
                "classes_",
            ):

                print(
                    "ASL model classes:",
                    self.model.classes_,
                )

            # ------------------------------------------------
            # MediaPipe Hands
            #
            # ASL uses one hand
            # ------------------------------------------------

            self.hands = (
                self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )

            self.reset_detection_state()

            print(
                "ASL MediaPipe initialized."
            )

            print(
                "=========================================="
            )

            return True

        except Exception as exc:

            print(
                "Error loading ASL model:",
                exc,
            )

            return False

    # ========================================================
    # RESET AUTO DETECTION
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

    def process_asl_landmarks(
        self,
        hand_landmarks,
    ):

        """
        Convert MediaPipe hand landmarks into
        the 42-feature format used by the ASL
        Random Forest model.

        21 landmarks × x,y = 42 features.
        """

        if hand_landmarks is None:

            raise ValueError(
                "No ASL hand landmarks provided."
            )

        data_aux = []

        x_ = []
        y_ = []

        # ----------------------------------------------------
        # Extract coordinates
        # ----------------------------------------------------

        for landmark in (
            hand_landmarks.landmark
        ):

            x_.append(
                float(landmark.x)
            )

            y_.append(
                float(landmark.y)
            )

        # ----------------------------------------------------
        # Normalize using minimum x/y
        #
        # This matches the common MediaPipe
        # landmark preprocessing used by the
        # Random Forest ASL classifier.
        # ----------------------------------------------------

        min_x = min(x_)
        min_y = min(y_)

        for landmark in (
            hand_landmarks.landmark
        ):

            data_aux.append(
                float(landmark.x) - min_x
            )

            data_aux.append(
                float(landmark.y) - min_y
            )

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        data_aux = data_aux[:42]

        while len(data_aux) < 42:

            data_aux.append(0.0)

        features = np.asarray(
            data_aux,
            dtype=np.float32,
        )

        if features.size != 42:

            raise ValueError(
                f"Expected 42 ASL features, "
                f"got {features.size}"
            )

        return features

    # ========================================================
    # PREDICT ASL
    # ========================================================

    def predict(
        self,
        hand_landmarks,
    ):

        try:

            if self.model is None:

                return "?", 0.0

            # ------------------------------------------------
            # Create features
            # ------------------------------------------------

            features = (
                self.process_asl_landmarks(
                    hand_landmarks
                )
            )

            input_data = (
                features.reshape(
                    1,
                    -1,
                )
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = (
                self.model.predict(
                    input_data
                )
            )

            predicted_value = prediction[0]

            # ------------------------------------------------
            # Determine predicted class
            #
            # Some sklearn models return numpy
            # integer labels.
            # ------------------------------------------------

            try:

                predicted_index = int(
                    predicted_value
                )

            except Exception:

                # If model returns string labels
                detected_char = str(
                    predicted_value
                )

                confidence = 0.0

                if hasattr(
                    self.model,
                    "predict_proba",
                ):

                    probabilities = (
                        self.model.predict_proba(
                            input_data
                        )
                    )

                    if (
                        probabilities is not None
                        and len(probabilities) > 0
                    ):

                        confidence = (
                            float(
                                np.max(
                                    probabilities[0]
                                )
                            )
                            * 100.0
                        )

                return (
                    detected_char,
                    confidence,
                )

            # ------------------------------------------------
            # Convert class index to letter
            # ------------------------------------------------

            detected_char = (
                self.asl_labels_dict.get(
                    predicted_index,
                    "?",
                )
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = 0.0

            if hasattr(
                self.model,
                "predict_proba",
            ):

                try:

                    probabilities = (
                        self.model.predict_proba(
                            input_data
                        )
                    )

                    if (
                        probabilities is not None
                        and len(probabilities) > 0
                    ):

                        probabilities = (
                            np.asarray(
                                probabilities[0]
                            )
                        )

                        if (
                            probabilities.size > 0
                        ):

                            confidence = (
                                float(
                                    np.max(
                                        probabilities
                                    )
                                )
                                * 100.0
                            )

                except Exception as exc:

                    print(
                        "ASL confidence error:",
                        exc,
                    )

            # ------------------------------------------------
            # If the Random Forest does not provide
            # predict_proba(), use a fallback confidence.
            # ------------------------------------------------

            if confidence <= 0.0:

                confidence = 85.0

            # ------------------------------------------------
            # Confidence threshold
            # ------------------------------------------------

            if (
                confidence
                >= self.confidence_threshold
            ):

                return (
                    detected_char,
                    confidence,
                )

            return (
                "?",
                confidence,
            )

        except Exception as exc:

            print(
                "ASL prediction error:",
                exc,
            )

            return (
                "?",
                0.0,
            )

    # ========================================================
    # AUTO DETECTION
    # ========================================================

    def should_auto_detect(
        self,
        detected_char,
        confidence,
    ):

        """
        Automatically accepts a character only when
        the same prediction remains stable for the
        required number of frames.
        """

        # ----------------------------------------------------
        # Cooldown
        # ----------------------------------------------------

        if self.cooldown_counter > 0:

            self.cooldown_counter -= 1

            return (
                False,
                0.0,
            )

        # ----------------------------------------------------
        # Invalid prediction
        # ----------------------------------------------------

        if (
            detected_char == "?"
            or confidence
            < self.confidence_threshold
        ):

            self.last_prediction = None

            self.prediction_counter = 0

            return (
                False,
                0.0,
            )

        # ----------------------------------------------------
        # Same prediction
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        progress = min(
            self.prediction_counter
            / self.required_consistent_frames,
            1.0,
        )

        # ----------------------------------------------------
        # Character accepted
        # ----------------------------------------------------

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
                1.0,
            )

        return (
            False,
            progress,
        )

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    def get_instructions(self):

        return [
            "ASL Detection - Place one hand in green box",
            "Hold the same sign for 1.5-2 seconds",
            "The detected letter is added automatically",
            "Press ENTER to add the word to the sentence",
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

        self.mp_hands = (
            mp.solutions.hands
        )

        self.mp_drawing = (
            mp.solutions.drawing_utils
        )

        self.hands = None

        # ----------------------------------------------------
        # ISL LABEL ORDER
        # ----------------------------------------------------

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

            35: " ",
        }

        # ----------------------------------------------------
        # MODEL PATH
        # ----------------------------------------------------

        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.model_path = os.path.join(
            base_dir,
            "final_lstm_hand_model11.keras",
        )

        # ----------------------------------------------------
        # AUTO DETECTION
        # ----------------------------------------------------

        self.last_prediction = None

        self.prediction_counter = 0

        # Around 2 seconds at ~15 FPS
        self.required_consistent_frames = 30

        self.last_added_char = None

        self.cooldown_counter = 0

        # Around 1 second at 15 FPS
        self.cooldown_frames = 15

        # Minimum confidence
        self.confidence_threshold = 60.0

        self.last_results = None

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        try:

            if not os.path.exists(
                self.model_path
            ):

                print(
                    "ISL model file not found:"
                )

                print(
                    self.model_path
                )

                return False

            print(
                "=========================================="
            )

            print(
                "Loading ISL model:"
            )

            print(
                self.model_path
            )

            self.model = load_model(
                self.model_path,
                compile=False,
            )

            print(
                "ISL model loaded successfully."
            )

            print(
                "Model input shape:",
                self.model.input_shape,
            )

            print(
                "Model output shape:",
                self.model.output_shape,
            )

            # ------------------------------------------------
            # MediaPipe Hands
            # ------------------------------------------------

            self.hands = (
                self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )

            self.reset_detection_state()

            print(
                "ISL MediaPipe initialized."
            )

            print(
                "=========================================="
            )

            return True

        except Exception as exc:

            print(
                "Error loading ISL model:",
                exc,
            )

            return False

    # ========================================================
    # RESET AUTO DETECTION
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
        multi_hand_landmarks,
    ):

        """
        ISL feature format:

        First hand:
            21 landmarks × x,y = 42

        Second hand:
            21 landmarks × x,y = 42

        Total:
            84 features

        Final model input:
            (1, 1, 84)
        """

        # ----------------------------------------------------
        # First hand
        # ----------------------------------------------------

        first_hand = np.zeros(
            21 * 2,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Second hand
        # ----------------------------------------------------

        second_hand = np.zeros(
            21 * 2,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Process detected hands
        # ----------------------------------------------------

        if multi_hand_landmarks:

            for (
                hand_idx,
                hand_landmarks
            ) in enumerate(
                multi_hand_landmarks
            ):

                values = []

                for landmark in (
                    hand_landmarks.landmark
                ):

                    values.append(
                        float(landmark.x)
                    )

                    values.append(
                        float(landmark.y)
                    )

                values = np.asarray(
                    values[:42],
                    dtype=np.float32,
                )

                if values.size < 42:

                    values = np.pad(
                        values,
                        (
                            0,
                            42 - values.size,
                        ),
                    )

                if hand_idx == 0:

                    first_hand = values

                elif hand_idx == 1:

                    second_hand = values

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        features = np.concatenate(
            [
                first_hand,
                second_hand,
            ]
        )

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        if features.size != 84:

            raise ValueError(
                f"Expected 84 features, "
                f"got {features.size}"
            )

        # ----------------------------------------------------
        # LSTM input
        # ----------------------------------------------------

        features = features.reshape(
            1,
            1,
            84,
        )

        return features

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        multi_hand_landmarks,
    ):

        try:

            if self.model is None:

                return (
                    "?",
                    0.0,
                )

            # ------------------------------------------------
            # Process landmarks
            # ------------------------------------------------

            features = (
                self.process_isl_landmarks(
                    multi_hand_landmarks
                )
            )

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            prediction = (
                self.model.predict(
                    features,
                    verbose=0,
                )
            )

            prediction = np.asarray(
                prediction
            )

            if prediction.ndim == 2:

                probabilities = (
                    prediction[0]
                )

            else:

                probabilities = (
                    prediction.flatten()
                )

            # ------------------------------------------------
            # Safety
            # ------------------------------------------------

            if probabilities.size == 0:

                return (
                    "?",
                    0.0,
                )

            # ------------------------------------------------
            # Get predicted class
            # ------------------------------------------------

            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = (
                float(
                    probabilities[
                        predicted_index
                    ]
                )
                * 100.0
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            detected_char = (
                self.isl_labels_dict.get(
                    predicted_index,
                    "?",
                )
            )

            # ------------------------------------------------
            # Threshold
            # ------------------------------------------------

            if (
                confidence
                >= self.confidence_threshold
            ):

                return (
                    detected_char,
                    confidence,
                )

            return (
                "?",
                confidence,
            )

        except Exception as exc:

            print(
                "ISL prediction error:",
                exc,
            )

            return (
                "?",
                0.0,
            )

    # ========================================================
    # AUTO DETECTION
    # ========================================================

    def should_auto_detect(
        self,
        detected_char,
        confidence,
    ):

        """
        Automatically accepts a character only when
        the same prediction remains stable.
        """

        # ----------------------------------------------------
        # Cooldown
        # ----------------------------------------------------

        if self.cooldown_counter > 0:

            self.cooldown_counter -= 1

            return (
                False,
                0.0,
            )

        # ----------------------------------------------------
        # Invalid prediction
        # ----------------------------------------------------

        if (
            detected_char == "?"
            or confidence
            < self.confidence_threshold
        ):

            self.last_prediction = None

            self.prediction_counter = 0

            return (
                False,
                0.0,
            )

        # ----------------------------------------------------
        # Same prediction
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        progress = min(
            self.prediction_counter
            / self.required_consistent_frames,
            1.0,
        )

        # ----------------------------------------------------
        # Accepted
        # ----------------------------------------------------

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
                1.0,
            )

        return (
            False,
            progress,
        )

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    def get_instructions(self):

        return [
            "ISL Detection - Place hands in green box",
            "Hold the same sign for 2-3 seconds",
            "Press ENTER to add the captured word to the sentence",
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
# UNIFIED SIGN LANGUAGE DETECTOR
# ============================================================

class UnifiedSignLanguageDetector:

    def __init__(self):

        # ----------------------------------------------------
        # Create detectors
        # ----------------------------------------------------

        self.isl_detector = (
            ISLDetector()
        )

        self.asl_detector = (
            ASLDetector()
        )

        self.current_detector = None

        self.language = None

        self.cap = None

        self.session_start_time = None

    # ========================================================
    # INITIALIZE
    # ========================================================

    def initialize(
        self,
        language="ISL",
    ):

        self.language = (
            language.upper()
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
        # Unsupported
        # ----------------------------------------------------

        print(
            f"{self.language} is not configured."
        )

        return False

    # ========================================================
    # INFO
    # ========================================================

    def get_detector_info(self):

        # ----------------------------------------------------
        # ISL information
        # ----------------------------------------------------

        if (
            self.language == "ISL"
            and self.current_detector
        ):

            return {

                "language":
                    "ISL (Indian Sign Language)",

                "model_type":
                    "LSTM Neural Network",

                "model_path":
                    self.isl_detector.model_path,

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
                    "(1, 1, 84)",
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
        # ASL information
        # ----------------------------------------------------

        if (
            self.language == "ASL"
            and self.current_detector
        ):

            return {

                "language":
                    "ASL (American Sign Language)",

                "model_type":
                    "Random Forest",

                "model_path":
                    self.asl_detector.model_path,

                "hands_supported":
                    "One hand",

                "classes":
                    len(
                        self.asl_detector
                        .asl_labels_dict
                    ),

                "supported_characters":
                    list(
                        self.asl_detector
                        .asl_labels_dict
                        .values()
                    ),

                "input_features":
                    42,

                "input_shape":
                    "(1, 42)",

                "auto_detection":
                    True,

                "confidence_threshold":
                    self.asl_detector
                    .confidence_threshold,

                "consistent_frames_required":
                    self.asl_detector
                    .required_consistent_frames,

            }

        return {}

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame,
    ):

        # ----------------------------------------------------
        # No detector
        # ----------------------------------------------------

        if (
            self.current_detector
            is None
        ):

            return (
                "?",
                0.0,
                {
                    "hand_detected": False,
                    "detection_progress": 0.0,
                    "should_auto_detect": False,
                    "hand_count": 0,
                },
            )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        results = (
            self.current_detector
            .hands
            .process(frame_rgb)
        )

        self.current_detector.last_results = (
            results
        )

        # ----------------------------------------------------
        # No hands
        # ----------------------------------------------------

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
                },
            )

        # ====================================================
        # ASL
        # ====================================================

        if self.language == "ASL":

            # ------------------------------------------------
            # ASL uses first detected hand
            # ------------------------------------------------

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            detected_char, confidence = (
                self.asl_detector.predict(
                    hand_landmarks
                )
            )

            # ------------------------------------------------
            # Auto detection
            # ------------------------------------------------

            should_detect, progress = (
                self.asl_detector
                .should_auto_detect(
                    detected_char,
                    confidence,
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
                        self.asl_detector
                        .cooldown_counter > 0,
                },
            )

        # ====================================================
        # ISL
        # ====================================================

        if self.language == "ISL":

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            detected_char, confidence = (
                self.isl_detector.predict(
                    results.multi_hand_landmarks
                )
            )

            # ------------------------------------------------
            # Auto detection
            # ------------------------------------------------

            should_detect, progress = (
                self.isl_detector
                .should_auto_detect(
                    detected_char,
                    confidence,
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
                        self.isl_detector
                        .cooldown_counter > 0,
                },
            )

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        return (
            "?",
            0.0,
            {
                "hand_detected": True,
                "hand_count":
                    len(
                        results.multi_hand_landmarks
                    ),
                "detection_progress": 0.0,
                "should_auto_detect": False,
            },
        )

    # ========================================================
    # DRAW LANDMARKS
    # ========================================================

    def draw_landmarks(
        self,
        frame,
    ):

        if (
            self.current_detector
            is None
        ):

            return

        results = (
            self.current_detector
            .last_results
        )

        if (
            results is None
            or not results.multi_hand_landmarks
        ):

            return

        # ----------------------------------------------------
        # Draw every detected hand
        # ----------------------------------------------------

        for hand_landmarks in (
            results.multi_hand_landmarks
        ):

            self.current_detector.mp_drawing.draw_landmarks(

                frame,

                hand_landmarks,

                self.current_detector
                .mp_hands
                .HAND_CONNECTIONS,

                self.current_detector
                .mp_drawing
                .DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2,
                    circle_radius=4,
                ),

                self.current_detector
                .mp_drawing
                .DrawingSpec(
                    color=(255, 255, 255),
                    thickness=2,
                ),
            )

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

        # ----------------------------------------------------
        # Cleanup both detectors
        # ----------------------------------------------------

        try:

            self.isl_detector.cleanup()

        except Exception:
            pass

        try:

            self.asl_detector.cleanup()

        except Exception:
            pass

        cv2.destroyAllWindows()