````markdown
# SingLang ML

SingLang ML is a machine-learning backend for real-time sign-language recognition using MediaPipe hand landmarks, an LSTM neural network, OpenCV, and FastAPI.

The current implementation supports **Indian Sign Language (ISL)** character recognition using a trained LSTM model.

---

## Features

- Real-time browser-camera sign recognition
- Indian Sign Language (ISL) detection
- MediaPipe hand landmark extraction
- Two-hand support
- 84-feature landmark representation
- LSTM neural-network inference
- 35 ISL output classes
- Automatic character detection using consistent-frame validation
- Confidence thresholding
- Word buffer and sentence buffer support
- FastAPI REST API
- Browser-camera integration
- GitHub Actions CI support

---

## Current ISL Model

The production model currently used by the project is:

```text
final_lstm_hand_model.keras
````

Model configuration:

| Property             | Value              |
| -------------------- | ------------------ |
| Framework            | TensorFlow / Keras |
| Model type           | LSTM               |
| Input shape          | `(None, 1, 84)`    |
| Runtime input        | `(1, 1, 84)`       |
| Output shape         | `(None, 35)`       |
| Classes              | 35                 |
| Hands                | Up to 2            |
| Features             | 84                 |
| Confidence threshold | 60%                |
| Consistent frames    | 30                 |

The 35 classes are:

```text
1 2 3 4 5 6 7 8 9
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
```

---

## Landmark Representation

The model expects 84 features.

For the first detected hand:

```text
21 landmarks × 2 coordinates = 42 features
```

For the second detected hand:

```text
21 landmarks × 2 coordinates = 42 features
```

Total:

```text
42 + 42 = 84 features
```

The final model input is:

```text
(1, 1, 84)
```

The current preprocessing uses the MediaPipe landmark `x` and `y` coordinates.

---

## Project Structure

A recommended repository structure is:

```text
sign_language_recognization_ml/
│
├── api.py
├── models.py
├── final_lstm_hand_model.keras
├── requirements.txt
├── README.md
├── .gitignore
│
└── .github/
    └── workflows/
        └── python-app.yml
```

Additional project files may be included depending on the frontend and application structure.

---

## Requirements

The current development environment uses:

```text
Python 3.11
TensorFlow 2.16.2
Keras 3.3.3
NumPy 1.26.4
OpenCV
MediaPipe
FastAPI
Uvicorn
```

Install dependencies with:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Windows Setup

Create a virtual environment:

```powershell
python -m venv env
```

Activate it:

```powershell
.\env\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Verify the Model

Run:

```powershell
python -c "from keras.models import load_model; m=load_model('final_lstm_hand_model.keras', compile=False); print('MODEL LOADED'); print('INPUT:', m.input_shape); print('OUTPUT:', m.output_shape)"
```

Expected output:

```text
MODEL LOADED
INPUT: (None, 1, 84)
OUTPUT: (None, 35)
```

---

## Verify the ISL Detector

Run:

```powershell
python -c "from models import ISLDetector; d=ISLDetector(); print('PATH:', d.model_path); print('LABELS:', len(d.isl_labels_dict)); print('LOAD:', d.load_model())"
```

Expected configuration:

```text
LABELS: 35
Expected classes: 35
Model output classes: 35
Model input shape: (None, 1, 84)
Model output shape: (None, 35)
LOAD: True
```

---

## Start the API

Run:

```powershell
python api.py
```

The API starts at:

```text
http://127.0.0.1:8000
```

---

## API Endpoints

### Start detection

```http
POST /start_detection
```

Example:

```json
{
  "language": "ISL"
}
```

PowerShell:

```powershell
Invoke-RestMethod `
  -Method POST `
  -Uri http://127.0.0.1:8000/start_detection `
  -ContentType "application/json" `
  -Body '{"language":"ISL"}'
```

---

### Detection status

```http
GET /detection_status
```

PowerShell:

```powershell
Invoke-RestMethod `
  http://127.0.0.1:8000/detection_status |
  ConvertTo-Json -Depth 10
```

Example response:

```json
{
  "active": true,
  "language": "ISL",
  "word_buffer": "",
  "sentence_buffer": "",
  "last_detected_char": "A",
  "confidence": 82.4,
  "completed": false,
  "detection_progress": 0.53,
  "auto_detection_enabled": true
}
```

---

### Process camera frame

```http
POST /process_frame
```

The browser frontend sends camera frames to the ML API.

The backend:

```text
Browser Camera
      ↓
OpenCV / Image Decode
      ↓
MediaPipe Hands
      ↓
84 Landmark Features
      ↓
LSTM Model
      ↓
35-Class Prediction
      ↓
Confidence Filtering
      ↓
Stable Frame Detection
      ↓
Character / Word / Sentence
```

---

### Stop detection

```http
POST /stop_detection
```

---

### Enter / accept character

```http
POST /enter
```

---

### Backspace

```http
POST /backspace
```

---

### Get detected text

```http
GET /get_detected_text
```

---

### Clear session

```http
DELETE /clear_session
```

---

## Automatic Detection

The detector does not immediately accept a prediction.

A prediction must:

1. Have confidence of at least 60%.
2. Remain the same character for 30 consecutive processed frames.
3. Reach the required stability threshold.
4. Pass the automatic detection logic.

Current configuration:

```python
confidence_threshold = 60.0
required_consistent_frames = 30
cooldown_frames = 15
```

This reduces accidental character insertion caused by short-lived prediction changes.

---

## Example

If the camera recognizes:

```text
A
```

with:

```text
Confidence: 82.4%
Progress: 0.53
```

the system continues tracking the same prediction.

When the stable-frame requirement reaches 100%, the character can be automatically accepted.

A sequence such as:

```text
A → B → C
```

can therefore become:

```text
ABC
```

in the word buffer.

---

## GitHub Actions

The repository can use:

```text
.github/workflows/python-app.yml
```

The workflow should:

* Check out the repository
* Install Python 3.11
* Install `requirements.txt`
* Compile Python files
* Verify TensorFlow and Keras
* Import the ISL detector

GitHub Actions is intended for CI verification. It does not provide a permanent camera-processing server.

---

## Git LFS

The Keras model may be too large for normal Git storage.

Recommended:

```powershell
git lfs install
git lfs track "*.keras"
```

Then:

```powershell
git add .gitattributes
git add final_lstm_hand_model.keras
git commit -m "Add ISL LSTM model using Git LFS"
git push
```

Check tracked files with:

```powershell
git lfs ls-files
```

The production model should be tracked using Git LFS if it exceeds normal GitHub file-size limits.

---

## Running the Complete System

### 1. Clone the repository

```powershell
git clone https://github.com/ravikumaras23/sign_language_recognization_ml.git
cd sign_language_recognization_ml
```

### 2. Create the virtual environment

```powershell
python -m venv env
```

### 3. Activate it

```powershell
.\env\Scripts\activate
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Verify the model

```powershell
python -c "from models import ISLDetector; d=ISLDetector(); print(d.load_model())"
```

### 6. Start the API

```powershell
python api.py
```

### 7. Connect the browser frontend

Configure the frontend to communicate with:

```text
http://127.0.0.1:8000
```

---

## Troubleshooting

### Model cannot be loaded

Verify that:

```text
final_lstm_hand_model.keras
```

exists in the expected project directory.

Then check:

```powershell
python -c "from keras.models import load_model; m=load_model('final_lstm_hand_model.keras', compile=False); print(m.input_shape, m.output_shape)"
```

Expected:

```text
(None, 1, 84)
(None, 35)
```

---

### Class mismatch

The current ISL detector expects:

```text
35 classes
```

The model must therefore produce:

```text
(None, 35)
```

A model producing 36, 248, or another number of outputs is not compatible with the current ISL label mapping.

---

### Camera prediction is `?`

Check:

* Browser camera permission
* MediaPipe hand detection
* Camera frame requests
* `/process_frame`
* Confidence threshold
* Landmark preprocessing

---

### Prediction is correct but not automatically accepted

The same character must remain stable for the configured number of frames:

```text
30 frames
```

Hold the sign steadily until detection progress reaches:

```text
1.0
```

---

## Development

Run Python syntax checking:

```powershell
python -m compileall .
```

Check TensorFlow/Keras:

```powershell
python -c "import tensorflow as tf; import keras; print(tf.__version__); print(keras.__version__)"
```

Check the detector:

```powershell
python -c "from models import ISLDetector; d=ISLDetector(); print(d.model_path); print(len(d.isl_labels_dict)); print(d.load_model())"
```

---

## Current Status

The ISL ML pipeline has been tested successfully with:

```text
Model loading       ✓
35-class output     ✓
84-feature input    ✓
MediaPipe detection ✓
LSTM inference      ✓
Character detection ✓
Confidence scoring  ✓
Auto detection      ✓
Word buffering      ✓
Sentence handling   ✓
FastAPI backend     ✓
Browser camera      ✓
```

Example verified prediction:

```text
Character: A
Confidence: 82.4%
```

---

## License

Add the project's intended license here before distributing the repository.

---

## Author

Ravi Kumar A S

GitHub:

https://github.com/ravikumaras23

Repository:

https://github.com/ravikumaras23/sign_language_recognization_ml

```
```
