# 🩸 BloodSense AI: Blood Group Classification from Infrared Hand Images

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)

An experimental deep learning project that classifies ABO/Rh blood group (8 classes) from infrared hand images using transfer learning, served through a Flask web app.

> ⚠️ **Research and educational project only.** Predicting blood group from images is not an established clinical method. Results below are on this project's dataset and have not been externally validated. Always confirm blood type with a certified laboratory test.

---

## Results

| Metric | VGG16 (v1) | EfficientNetV2S (v2) |
| --- | --- | --- |
| Validation accuracy | ~85% | 99%+ |
| Top-2 accuracy | ~95% | ~100% |
| Inference time | ~1.5s | ~0.9s |
| Model size | 98 MB | 85 MB |

See `training_validation_accuracy.png` for training curves.

### Known limitations

- Reported figures are validation accuracy; a separate held-out test set and subject-level splitting are needed to confirm generalization
- High validation accuracy on a small dataset can reflect overfitting or split leakage; independent validation is needed before drawing conclusions

---

## Modeling Approach

### v2: EfficientNetV2S (transfer learning)

```
Input (224×224×3)
  └─ EfficientNetV2S backbone (ImageNet pre-trained)
     └─ GlobalAveragePooling2D
        └─ BatchNormalization
           └─ Dense(512, swish) + Dropout(0.4)
              └─ Dense(256, swish) + Dropout(0.3)
                 └─ Dense(8, softmax)
```

**Training techniques:**
- Two-phase training: frozen backbone first, then full fine-tuning
- Data augmentation: flip, rotation, zoom, contrast, brightness
- Mixup augmentation and label smoothing (0.05 → 0.03)
- Cosine learning-rate decay with linear warmup, AdamW with weight decay
- Test-time augmentation (TTA) at inference

### v1: VGG16 baseline

Original transfer-learning baseline (`model.py`), kept for comparison.

---

## Application Features

- 8 blood types: A+, A−, B+, B−, AB+, AB−, O+, O−
- Full probability distribution across classes, not just the top prediction
- SQLite-backed history with search and filter
- Donor/recipient compatibility chart per result
- Printable report
- TFLite model export (`convert_model.py`) for a lighter deployment

---

## Quick Start

```bash
git clone https://github.com/sujal112002/blood-group-detection.git
cd blood-group-detection
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                 # open http://localhost:5000
```

Production:

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

### Retrain

```
dataset_folder/
  train/A+/  train/A-/  train/B+/  ...
  val/A+/    val/A-/    val/B+/    ...
```

```bash
python model_v2.py
```

---

## API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Dashboard |
| `/predict` | POST | Submit an image for classification |
| `/history` | GET | Prediction history |
| `/history/delete/<id>` | POST | Delete a record |
| `/api/stats` | GET | JSON statistics |
| `/health` | GET | Health check |

`POST /predict` form fields: `image` (file), `temperature` (float, °C), `patient_name`, `patient_age` (optional), `patient_gender` (optional), `notes` (optional).

---

## Project Structure

```
├── app.py                    # Flask application
├── model_v2.py               # EfficientNetV2S training
├── model.py                  # VGG16 baseline training
├── preprocess.py             # image preprocessing
├── split_data.py             # train/val split
├── evaluate.py               # evaluation
├── convert_model.py          # TFLite conversion
├── templates/                # dashboard, result, history pages
├── static/                   # assets
├── requirements.txt
├── Procfile / render.yaml    # deployment config
└── runtime.txt
```

## Deployment

Configured for Render (`render.yaml`). Environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `BLOOD_GROUP_MODEL` | `blood_group_model_vgg16.keras` | Path to model file |
| `CLASS_INDICES_PATH` | `class_indices.pkl` | Class mapping file |
| `MAX_UPLOAD_MB` | `8` | Max upload size |
| `PORT` | `5000` | Server port |
