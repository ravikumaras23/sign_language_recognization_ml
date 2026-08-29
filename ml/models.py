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
        # IMPORTANT
        #
        # Your trained Random Forest has:
        #
        # classes = 0 ... 34
        # features = 84
        #
        # Therefore the feature format MUST remain 84.
        # ----------------------------------------------------

        self.asl_labels_dict = {
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

        # ----------------------------------------------------
        # MODEL PATH
        #
        # Actual trained model:
        #
        # ml/
        #   models.py
        #   random_forest_isl_model.pkl
        #
        # ----------------------------------------------------

        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.model_path = os.path.join(
            base_dir,
            "random_forest_isl_model.pkl",
        )

        # ----------------------------------------------------
        # AUTO DETECTION
        # ----------------------------------------------------

        self.last_prediction = None

        self.prediction_counter = 0

        # Hold sign for approximately 1.5-2 seconds
        self.required_consistent_frames = 25

        self.last_added_char = None

        self.cooldown_counter = 0

        # Approximately 1 second at 15 FPS
        self.cooldown_frames = 15

        # Minimum confidence
        self.confidence_threshold = 60.0

        self.last_results = None

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        try:

            print(
                "=========================================="
            )

            print(
                "Loading ASL Random Forest model"
            )

            print(
                "Model path:",
                self.model_path
            )

            # ------------------------------------------------
            # Check model
            # ------------------------------------------------

            if not os.path.isfile(
                self.model_path
            ):

                print(
                    "ERROR: ASL model file not found"
                )

                print(
                    "Expected:",
                    self.model_path
                )

                return False

            # ------------------------------------------------
            # Load pickle
            # ------------------------------------------------

            with open(
                self.model_path,
                "rb"
            ) as file:

                loaded_model = pickle.load(file)

            # ------------------------------------------------
            # Support:
            #
            # RandomForestClassifier
            #
            # OR
            #
            # {
            #     "model": RandomForestClassifier(...)
            # }
            # ------------------------------------------------

            if (
                isinstance(
                    loaded_model,
                    dict
                )
                and "model" in loaded_model
            ):

                self.model = loaded_model["model"]

            else:

                self.model = loaded_model

            # ------------------------------------------------
            # Validate model
            # ------------------------------------------------

            if self.model is None:

                print(
                    "ERROR: Loaded ASL model is None"
                )

                return False

            # ------------------------------------------------
            # Print model information
            # ------------------------------------------------

            print(
                "ASL model loaded successfully."
            )

            print(
                "Model type:",
                type(self.model).__name__
            )

            if hasattr(
                self.model,
                "classes_"
            ):

                print(
                    "Classes:",
                    self.model.classes_
                )

                print(
                    "Number of classes:",
                    len(self.model.classes_)
                )

            if hasattr(
                self.model,
                "n_features_in_"
            ):

                print(
                    "Number of input features:",
                    self.model.n_features_in_
                )

                # ------------------------------------------------
                # IMPORTANT
                #
                # Your trained model must have 84 features.
                # ------------------------------------------------

                if self.model.n_features_in_ != 84:

                    raise ValueError(
                        "ASL Random Forest expects "
                        f"{self.model.n_features_in_} features, "
                        "but this application is configured "
                        "for 84 features."
                    )

            # ------------------------------------------------
            # MediaPipe
            #
            # We use one hand.
            #
            # The 84 feature vector is:
            #
            # first hand = 42
            # second hand = 42 zeros
            #
            # ------------------------------------------------

            self.hands = self.mp_hands.Hands(

                static_image_mode=False,

                max_num_hands=1,

                min_detection_confidence=0.5,

                min_tracking_confidence=0.5,
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
                "ERROR loading ASL model:",
                repr(exc)
            )

            self.model = None

            return False

    # ========================================================
    # RESET
    # ========================================================

    def reset_detection_state(self):

        self.last_prediction = None

        self.prediction_counter = 0

        self.last_added_char = None

        self.cooldown_counter = 0

    # ========================================================
    # LANDMARK PROCESSING
    # ========================================================

    def process_asl_landmarks(
        self,
        hand_landmarks,
    ):

        """
        Convert MediaPipe landmarks into
        the 84-feature representation required
        by the trained Random Forest.

        21 landmarks × x,y = 42

        First hand:
            42 features

        Second hand:
            42 features

        Total:
            84 features
        """

        if hand_landmarks is None:

            raise ValueError(
                "No ASL hand landmarks provided."
            )

        # ----------------------------------------------------
        # First hand = 42
        # ----------------------------------------------------

        first_hand = []

        x_values = []
        y_values = []

        for landmark in (
            hand_landmarks.landmark
        ):

            x_values.append(
                float(landmark.x)
            )

            y_values.append(
                float(landmark.y)
            )

        # ----------------------------------------------------
        # Normalize relative to minimum x/y
        #
        # This matches the preprocessing used
        # by the existing Random Forest code.
        # ----------------------------------------------------

        min_x = min(x_values)
        min_y = min(y_values)

        for landmark in (
            hand_landmarks.landmark
        ):

            first_hand.append(
                float(landmark.x) - min_x
            )

            first_hand.append(
                float(landmark.y) - min_y
            )

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        first_hand = first_hand[:42]

        while len(first_hand) < 42:

            first_hand.append(0.0)

        # ----------------------------------------------------
        # Second hand
        #
        # ASL uses one hand in this detector,
        # therefore the remaining 42 features
        # are zero.
        # ----------------------------------------------------

        second_hand = [
            0.0
        ] * 42

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        features = (
            first_hand
            + second_hand
        )

        # ----------------------------------------------------
        # Convert numpy
        # ----------------------------------------------------

        features = np.asarray(
            features,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if features.size != 84:

            raise ValueError(
                "Expected 84 ASL features, "
                f"got {features.size}"
            )

        return features

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        hand_landmarks,
    ):

        try:

            if self.model is None:

                return (
                    "?",
                    0.0
                )

            # ------------------------------------------------
            # Create 84 features
            # ------------------------------------------------

            features = (
                self.process_asl_landmarks(
                    hand_landmarks
                )
            )

            # ------------------------------------------------
            # Random Forest expects:
            #
            # (1, 84)
            # ------------------------------------------------

            input_data = (
                features.reshape(
                    1,
                    84
                )
            )

            # ------------------------------------------------
            # Safety validation
            # ------------------------------------------------

            if hasattr(
                self.model,
                "n_features_in_"
            ):

                expected = (
                    self.model.n_features_in_
                )

                if expected != 84:

                    raise ValueError(
                        "Model expects "
                        f"{expected} features, "
                        "but detector generated 84."
                    )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = (
                self.model.predict(
                    input_data
                )
            )

            predicted_value = (
                prediction[0]
            )

            # ------------------------------------------------
            # Convert class
            # ------------------------------------------------

            predicted_index = int(
                predicted_value
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            detected_char = (
                self.asl_labels_dict.get(
                    predicted_index,
                    "?"
                )
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = 0.0

            if hasattr(
                self.model,
                "predict_proba"
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
                        repr(exc)
                    )

            # ------------------------------------------------
            # Fallback
            # ------------------------------------------------

            if confidence <= 0.0:

                confidence = 85.0

            # ------------------------------------------------
            # Threshold
            # ------------------------------------------------

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
        confidence,
    ):

        # ----------------------------------------------------
        # Cooldown
        # ----------------------------------------------------

        if self.cooldown_counter > 0:

            self.cooldown_counter -= 1

            return (
                False,
                0.0
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
                0.0
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
            1.0
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

            "ASL Detection - Place one hand in the green box",

            "Hold the same sign for 2-3 seconds",

            "The character will be detected automatically",

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
        # ISL
        # ----------------------------------------------------

        self.isl_detector = (
            ISLDetector()
        )

        # ----------------------------------------------------
        # ASL
        # ----------------------------------------------------

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
            language or "ISL"
        ).upper()

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

        self.current_detector = None

        return False

    # ========================================================
    # INFO
    # ========================================================

    def get_detector_info(self):

        # ----------------------------------------------------
        # ISL
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
        # ASL
        # ----------------------------------------------------

        if (
            self.language == "ASL"
            and self.current_detector
        ):

            return {

                "language":
                    "ASL",

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

        return {}

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame,
    ):

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
                }
            )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
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
                }
            )

        # ====================================================
        # ASL
        # ====================================================

        if self.language == "ASL":

            # ------------------------------------------------
            # First hand
            # ------------------------------------------------

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            # ------------------------------------------------
            # Prediction
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
                        self.asl_detector
                        .cooldown_counter > 0,
                }
            )

        # ====================================================
        # ISL
        # ====================================================

        detected_char, confidence = (
            self.isl_detector.predict(
                results.multi_hand_landmarks
            )
        )

        should_detect, progress = (
            self.isl_detector
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
                    self.isl_detector
                    .cooldown_counter > 0,
            }
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

        if self.current_detector:

            self.current_detector.cleanup()

        cv2.destroyAllWindows()