# ExoDetector Model Training & Dataset Feeding Guide

This guide details the requirements, dataset schemas, and commands to train and fine-tune the **ExoDetector** machine learning models on a new light curve dataset.

The system uses a two-stage deep learning architecture:
1. **Light Curve Autoencoder**: Denoises raw light curves (trained in an unsupervised fashion on flux windows).
2. **CNN Transit Classifier**: Identifies exoplanets, eclipsing binaries, stellar spots, and false positives (trained using a 1D Convolutional Neural Network branch + a multi-layer perceptron branch for scalar orbital features).

---

## 1. Dataset Schema Requirements (CSV format)

To feed a new dataset into the training pipeline, compile your labeled light curves into a single CSV file.

### Required CSV Columns
| Column Name | Type | Description |
| :--- | :--- | :--- |
| **`flux`** | `string` | A comma-separated or semicolon-separated sequence of normalized flux readings (e.g. `"1.002,0.998,0.997,1.001"`). |
| **`label`** *or* **`class`** | `string` | The target classification. Must be one of the four supported labels (see below). |
| **`period`** | `float` | The orbital period in days (e.g., `4.5682`). Used to phase-fold the light curve. |
| **`time`** *(optional)* | `string` | A comma- or semicolon-separated sequence of times in days corresponding to the flux points. If omitted, the pipeline defaults to indices (`0, 1, 2, ...`). |

### Supported Labels (`label` Column)
The CNN classifier maps targets to one of four indices. Any label not matching these exact strings will be ignored:
* **`PLANET`**: True exoplanetary transit signal (repeating dip in light curve).
* **`ECLIPSING_BINARY`**: A binary star system where stars eclipse each other.
* **`FALSE_POSITIVE`**: Instrument glitches, backgrounds, or random noise anomalies.
* **`STARSPOT`**: Periodic dips caused by starspots rotating on the stellar surface.

### Example CSV Row
```csv
time,flux,period,label
"0.1;0.2;0.3;0.4","1.0002;0.9995;0.9912;1.0001",4.5204,PLANET
```

---

## 2. Hardware & Environment Requirements

Training can be run on CPU or GPU. The pipeline will automatically select the CUDA-enabled GPU if available.

### Dependencies (Local Virtual Environment)
If training outside Docker, ensure these are installed (`requirements.txt`):
* `torch` (PyTorch v2.0+)
* `numpy`
* `pandas`
* `astropy`
* `wotan` (for detrending)
* `scipy`
* `pydantic-settings`

---

## 3. How to Train the Model (Command-by-Command)

There are three ways to launch model training:

### Method A: Local execution (CLI)
Inside your virtual environment, run the training script by passing the path to your training CSV:
```bash
# 1. Activate your virtual environment (.venv)
.venv\Scripts\activate

# 2. Set Python path to backend
$env:PYTHONPATH="backend"

# 3. Execute training
python backend/app/ml/train.py data/processed/training_set.csv --epochs 100 --batch-size 64
```

### Method B: Docker execution (Recommended)
If your application is running inside Docker compose:
```bash
docker compose exec backend python app/ml/train.py data/processed/training_set.csv --epochs 100 --batch-size 64
```

### Method C: VIA API / Dashboard (Asynchronous Celery Task)
You can trigger training remotely via the frontend dashboard or a `POST` request to the backend API.
* **Endpoint**: `POST /api/train`
* **Headers**: `X-API-Key: <your_api_key>` (matches `API_KEY` in `.env`)
* **Request Body**:
  ```json
  {
    "dataset_path": "data/processed/training_set.csv",
    "epochs": 100,
    "batch_size": 64
  }
  ```
* **cURL Command**:
  ```bash
  curl -X POST http://localhost:8000/api/train \
    -H "Content-Type: application/json" \
    -H "X-API-Key: change-me" \
    -d '{"dataset_path": "data/processed/training_set.csv", "epochs": 100, "batch_size": 64}'
  ```

---

## 4. What Happens During Training?

When the training job is launched:
1. **Autoencoder Unsupervised Warmup**:
   * Extracts raw flux rows from the training set.
   * Segments flux curves into sliding windows of length `512` (overlap = `0.5`).
   * Trains the convolutional encoder/decoder layers for 50 epochs.
   * Saves the weights to `models/autoencoder_best.pt`.
2. **CNN Transit Classifier Preprocessing**:
   * Performs data augmentation on training curves (Gaussian noise addition, time flipping, flux scaling).
   * Phase-folds the light curve using the provided `period` and `time` columns into exactly `128` phase bins.
   * Extracts 4 engineering scalar features:
     * `depth_ppt` (transit depth in parts-per-thousand)
     * `duration_period_ratio` (transit duration ratio)
     * `secondary_flag` (presence of secondary eclipses)
     * `odd_even` (odd/even transit depth differences)
3. **CNN Training**:
   * Evaluates validation accuracy at each epoch.
   * Saves the model state with the highest validation accuracy to `models/classifier_best.pt`.
4. **Post-Calibration**:
   * Computes a logit scaling temperature parameter and writes it to `models/temperature.json` (to align probability confidence ratings).
   * Saves evaluation statistics (precision, recall, F1, confusion matrix, test accuracy) to `models/training_metadata.json`.
5. **Reload**:
   * The FastAPI server automatically reloads the new best weights (`classifier_best.pt`) on next API calls or restarts.
