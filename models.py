
import os
import cv2
import mediapipe as mp
import numpy as np

from keras.models import load_model


# ============================================================
# ISL DETECTOR
# ============================================================

class ISLDetector:

    def __init__(self):

        self.model = None

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = None

        # ----------------------------------------------------
        # IMPORTANT:
        # This is the class order used by your old ISL code.
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
        # CURRENT MODEL
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

        # Camera is deliberately processed around 15 FPS.
        # 30 consistent frames ~= 2 seconds.
        self.required_consistent_frames = 30

        self.last_added_char = None

        self.cooldown_counter = 0

        # About 1 second cooldown at 15 FPS.
        self.cooldown_frames = 15

        # Minimum confidence.
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
                    f"ISL model file not found: "
                    f"{self.model_path}"
                )

                return False

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

            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.reset_detection_state()

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
        IMPORTANT:

        This reproduces the feature format used by
        your old ISL application.

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
        # Empty arrays for two hands
        # ----------------------------------------------------

        first_hand = np.zeros(
            21 * 2,
            dtype=np.float32,
        )

        second_hand = np.zeros(
            21 * 2,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Process detected hands
        # ----------------------------------------------------

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

                return "?", 0.0

            features = (
                self.process_isl_landmarks(
                    multi_hand_landmarks
                )
            )

            prediction = self.model.predict(
                features,
                verbose=0,
            )

            prediction = np.asarray(
                prediction
            )

            if prediction.ndim == 2:

                probabilities = prediction[0]

            else:

                probabilities = prediction.flatten()

            if probabilities.size == 0:

                return "?", 0.0

            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )

            confidence = (
                float(
                    probabilities[
                        predicted_index
                    ]
                )
                * 100.0
            )

            detected_char = (
                self.isl_labels_dict.get(
                    predicted_index,
                    "?",
                )
            )

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
            "ISL Detection - Place hands in green box",
            "Hold the same sign for 2-3 seconds for automatic detection",
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
# UNIFIED DETECTOR
# ============================================================

class UnifiedSignLanguageDetector:

    def __init__(self):

        self.isl_detector = (
            ISLDetector()
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
        # Current model is ISL.
        # ----------------------------------------------------

        if self.language == "ISL":

            self.current_detector = (
                self.isl_detector
            )

            return (
                self.isl_detector.load_model()
            )

        print(
            f"{self.language} is not configured "
            f"with the current best_lstm_model.keras."
        )

        return False

    # ========================================================
    # INFO
    # ========================================================

    def get_detector_info(self):

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
                },
            )

        # ----------------------------------------------------
        # BGR → RGB
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
        # No hand
        # ----------------------------------------------------

        if not results.multi_hand_landmarks:

            self.current_detector.last_prediction = None

            self.current_detector.prediction_counter = 0

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
        # ISL prediction
        # ----------------------------------------------------

        detected_char, confidence = (
            self.current_detector.predict(
                results.multi_hand_landmarks
            )
        )

        # ----------------------------------------------------
        # Auto detection
        # ----------------------------------------------------

        should_detect, progress = (
            self.current_detector
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
                        results.multi_hand_landmarks
                    ),
                "detection_progress":
                    progress,
                "should_auto_detect":
                    should_detect,
                "cooldown_active":
                    self.current_detector
                    .cooldown_counter > 0,
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
            self.current_detector is None
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
