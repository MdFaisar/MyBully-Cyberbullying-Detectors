# MyBully — Cyberbullying Detector

A Flask web app that detects cyberbullying in text messages using a BERT (Bidirectional Encoder Representations from Transformers) model. The app supports user feedback for continuous model improvement.

## Project Stack

- Python 3.8+
- Flask — lightweight web server and REST API
- PyTorch & Transformers — BERT model implementation
- pandas — CSV read/write and dataset manipulation
- scikit-learn — data splitting and metrics
- HTML/CSS/JavaScript — front-end UI

## Project Structure

```
.
├── app.py                     # Flask app and API endpoints
├── model.py                   # BullyingDetector class using BERT
├── train_model.py             # Script to train BERT model
├── requirements.txt           # Python dependencies
├── dataset/
│   ├── cyberbullying_dataset.csv    # Training dataset
│   ├── cyberbullying_dataset_test.csv # Test dataset
│   └── feedback.csv          # User-submitted feedback
├── models/
│   └── bert_cyberbullying/   # BERT model files
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       └── special_tokens_map.json
├── static/                   # Front-end assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── templates/
│   ├── index.html           # Landing page
│   └── dashboard.html       # Results page with correction buttons
└── README.md
```

## What this project does

- Trains a TF-IDF + Logistic Regression classifier to detect bullying vs non-bullying messages.
- Serves a REST API (`/api/analyze`) to analyze a text and return prediction, confidence, and severity.
- Lets users provide feedback (Correct/Wrong). Feedback is saved to `dataset/feedback.csv` and used to update the model. The update routine combines the original dataset and recent feedback so the model keeps base knowledge while learning from user corrections.

## Getting Started

### Option 1: Train on Google Colab (Recommended)

1. Upload these files/folders to Google Colab:
   - `train_model.py`
   - `dataset/` folder
   - `models/` folder (empty folder)

2. Create a new Colab notebook and run:
   ```python
   !python train_model.py
   ```

3. After training, download the following files from `models`:
   - `config.json`
   - `pytorch_model.bin`
   - `tokenizer_config.json`
   - `vocab.txt`
   - `special_tokens_map.json`

4. Place these files in your local project's `models` directory

5. Run the web app:
    ```powershell
    python app.py
    ```

### Option 2: Local Setup

1. Create and activate a virtual environment:
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```

2. Prepare your dataset:
    - Place training data in `dataset/cyberbullying_dataset.csv`
    - Format: CSV with columns `text` and `label` (0 = Non-Bullying, 1 = Bullying)
    - Optional: Add test data in `dataset/cyberbullying_dataset_test.csv`

3. Train the model (if not in google collab):
    ```powershell
    python train_model.py
    ```

4. Run the web app:
    ```powershell
    python app.py
    ```

Visit http://127.0.0.1:5000/ in your browser.

## API endpoints

- POST /api/analyze
  - Input: JSON { "text": "..." }
  - Output: prediction JSON with keys: `prediction`, `confidence`, `severity`, `is_bullying`, and `probabilities`.

- POST /api/correct
  - Input: JSON { "text": "...", "prediction": "Bullying|Non-Bullying", "correction": "Bullying|Non-Bullying" }
  - Behavior: saves feedback to `dataset/feedback.csv` and schedules model update (threaded). The retraining routine combines feedback with `dataset/cyberbullying_dataset.csv` (feedback duplicated to increase its weight) and writes updated model files.

Check `app.py` for implementation details and any additional endpoints exposed by the UI.

## Model Details

### BERT Architecture
- Based on `bert-base-uncased`
- Fine-tuned for cyberbullying classification
- Uses attention mechanisms for better context understanding
- Processes up to 128 tokens per text

### Training Process
- Trains on combined dataset (original + feedback)
- Uses AdamW optimizer with learning rate 2e-5
- Implements early stopping based on validation loss
- Supports CPU and CUDA (GPU) training

### Feedback Integration
- Saves user feedback to `dataset/feedback.csv`
- Combines feedback with original dataset
- Duplicates feedback samples for stronger influence
- Retrains incrementally to maintain performance

## Troubleshooting

### Common Issues

1. CUDA/GPU errors
   - Solution: Model falls back to CPU automatically

2. Memory issues during training
   - Solution: Reduce batch size in `train_model.py`
   - Alternative: Use Google Colab with GPU runtime

3. Import errors with protobuf/tensorflow
   - Solution: Ensure correct protobuf version:
     ```powershell
     pip install --upgrade protobuf==4.23.4
     ```

### Performance Optimization

- Use GPU for training when available
- Adjust batch size based on available memory
- Consider using smaller BERT variants for faster inference

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request


## OCR Image Upload (New)

This project now includes an OCR feature that allows you to upload an image (PNG/JPEG) and extract text using Tesseract OCR, then analyze the extracted text with the model.

Requirements:

- Python packages: `Pillow`, `pytesseract` (already added to `requirements.txt`).
- System dependency: the Tesseract OCR engine must be installed separately on your OS.

Install Tesseract on Windows (example):

```powershell
choco install tesseract -y
```

Or download from: https://github.com/tesseract-ocr/tesseract

After installing the Tesseract binary, make sure it is on your PATH so `pytesseract` can call it.

API endpoint:

- POST /api/ocr
   - Input: multipart/form-data with key `image` containing an image file (PNG/JPEG)
   - Output: JSON { "extracted_text": "...", "analysis": { <same keys as /api/analyze> }, "success": true }
   - Behavior: Extracts text from the uploaded image and runs the same text analysis pipeline, returning both the extracted text and the model's prediction.