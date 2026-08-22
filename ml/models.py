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

        # ====================================================
        # ISL LABELS
        # 9 numbers + 26 alphabets = 35 classes
        # ====================================================

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

        # ====================================================
        # MODEL PATH
        # ====================================================

        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        # IMPORTANT:
        # This is the model that was successfully tested.
        self.model_path = os.path.join(
            base_dir,
            "final_lstm_hand_model.keras",
        )

        # ====================================================
        # AUTO DETECTION
        # ====================================================

        self.last_prediction = None

        self.prediction_counter = 0

        # 30 stable frames ~= 2 seconds at ~15 FPS
        self.required_consistent_frames = 30

        self.last_added_char = None

        self.cooldown_counter = 0

        # ~1 second cooldown at 15 FPS
        self.cooldown_frames = 15

        # Minimum prediction confidence
        self.confidence_threshold = 60.0

        self.last_results = None

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        try:

            # ------------------------------------------------
            # Check model file
            # ------------------------------------------------

            if not os.path.exists(self.model_path):

                print(
                    f"ISL model file not found: "
                    f"{self.model_path}"
                )

                return False

            print()
            print("==========================================")
            print("Loading ISL model")
            print("==========================================")
            print("Model path:")
            print(self.model_path)

            # ------------------------------------------------
            # Load Keras model
            # ------------------------------------------------

            self.model = load_model(
                self.model_path,
                compile=False,
            )

            # ------------------------------------------------
            # Validate model output classes
            # ------------------------------------------------

            expected_classes = len(
                self.isl_labels_dict
            )

            actual_classes = self.model.output_shape[-1]

            print(
                "Expected classes:",
                expected_classes,
            )

            print(
                "Model output classes:",
                actual_classes,
            )

            if actual_classes != expected_classes:

                raise ValueError(
                    f"ISL model class mismatch: "
                    f"model has {actual_classes} outputs, "
                    f"but labels contain {expected_classes} classes."
                )

            # ------------------------------------------------
            # Validate input shape
            # ------------------------------------------------

            expected_input_features = 84

            actual_input_shape = self.model.input_shape

            print(
                "Model input shape:",
                actual_input_shape,
            )

            print(
                "Model output shape:",
                self.model.output_shape,
            )

            if (
                actual_input_shape[-1]
                != expected_input_features
            ):

                raise ValueError(
                    f"ISL model input mismatch: "
                    f"expected 84 features, "
                    f"but model expects {actual_input_shape[-1]}."
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

            # ------------------------------------------------
            # Reset detection state
            # ------------------------------------------------

            self.reset_detection_state()

            print()
            print("==========================================")
            print("ISL MODEL LOADED SUCCESSFULLY")
            print("==========================================")
            print(
                f"Classes: {expected_classes}"
            )
            print(
                f"Input: {actual_input_shape}"
            )
            print(
                f"Output: {self.model.output_shape}"
            )
            print("Hands: 2")
            print("Features: 84")
            print("==========================================")
            print()

            return True

        except Exception as exc:

            print()
            print("==========================================")
            print("ERROR LOADING ISL MODEL")
            print("==========================================")
            print(exc)
            print("==========================================")
            print()

            self.model = None

            if self.hands is not None:

                try:
                    self.hands.close()
                except Exception:
                    pass

            self.hands = None

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
        Convert MediaPipe hand landmarks into
        the 84-feature format expected by the model.

        Hand 1:
            21 landmarks × 2 = 42

        Hand 2:
            21 landmarks × 2 = 42

        Total:
            84 features

        Final input:
            (1, 1, 84)
        """

        # ----------------------------------------------------
        # Empty arrays for two hands
        # ----------------------------------------------------

        first_hand = np.zeros(
            42,
            dtype=np.float32,
        )

        second_hand = np.zeros(
            42,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Process detected hands
        # ----------------------------------------------------

        if multi_hand_landmarks:

            for hand_idx, hand_landmarks in enumerate(
                multi_hand_landmarks
            ):

                # Only support first two hands
                if hand_idx >= 2:
                    break

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

                # Safety padding
                if values.size < 42:

                    values = np.pad(
                        values,
                        (
                            0,
                            42 - values.size,
                        ),
                        mode="constant",
                    )

                if hand_idx == 0:

                    first_hand = values

                elif hand_idx == 1:

                    second_hand = values

        # ----------------------------------------------------
        # Combine both hands
        # ----------------------------------------------------

        features = np.concatenate(
            [
                first_hand,
                second_hand,
            ]
        )

        # ----------------------------------------------------
        # Validate features
        # ----------------------------------------------------

        if features.size != 84:

            raise ValueError(
                f"Expected 84 features, "
                f"got {features.size}"
            )

        # ----------------------------------------------------
        # Convert to LSTM input
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

            # ------------------------------------------------
            # Prepare features
            # ------------------------------------------------

            features = self.process_isl_landmarks(
                multi_hand_landmarks
            )

            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            prediction = self.model.predict(
                features,
                verbose=0,
            )

            prediction = np.asarray(
                prediction
            )

            # ------------------------------------------------
            # Extract probabilities
            # ------------------------------------------------

            if prediction.ndim == 2:

                probabilities = prediction[0]

            else:

                probabilities = prediction.flatten()

            if probabilities.size == 0:

                return "?", 0.0

            # ------------------------------------------------
            # Get predicted class
            # ------------------------------------------------

            predicted_index = int(
                np.argmax(probabilities)
            )

            confidence = (
                float(
                    probabilities[predicted_index]
                )
                * 100.0
            )

            # ------------------------------------------------
            # Convert index → character
            # ------------------------------------------------

            detected_char = (
                self.isl_labels_dict.get(
                    predicted_index,
                    "?",
                )
            )

            # ------------------------------------------------
            # Confidence filter
            # ------------------------------------------------

            if confidence >= self.confidence_threshold:

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
        Automatically accepts a character when
        the same prediction remains stable for
        the required number of frames.
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
            or confidence < self.confidence_threshold
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

        if self.last_prediction == detected_char:

            self.prediction_counter += 1

        else:

            self.last_prediction = detected_char

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

            self.last_added_char = detected_char

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

        self.isl_detector = ISLDetector()

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

        self.language = language.upper()

        # ----------------------------------------------------
        # ISL
        # ----------------------------------------------------

        if self.language == "ISL":

            self.current_detector = (
                self.isl_detector
            )

            return self.isl_detector.load_model()

        print(
            f"{self.language} is not configured "
            f"with the current ISL model."
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

                "model_output_shape":
                    str(
                        self.isl_detector
                        .model.output_shape
                    )
                    if self.isl_detector.model
                    else None,

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

        if self.current_detector is None:

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

        self.current_detector.last_results = results

        # ----------------------------------------------------
        # No hand detected
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
        # Prediction
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

        if self.current_detector is None:

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