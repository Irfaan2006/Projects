# Skin Cancer Classification — Flask App

Serves the `rexnet_150` model trained in
`skin-cancer-classification-project-with-pytorch.ipynb` behind a small
Flask web app with an upload UI and a JSON API.

## 1. Get your trained weights

In the notebook, training saves the best checkpoint to:

```
saved_models/cancer_best_model.pth
```

Copy that file into this project's `saved_models/` folder so the final
path is:

```
flask_app/saved_models/cancer_best_model.pth
```

If the file isn't there, the app still starts, but predictions will be
disabled and the homepage will show a warning.

## 2. Install dependencies

```bash
cd flask_app
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser and upload an
image.

## API

### `POST /predict`
Multipart form upload, field name `file` (jpg/png).

```bash
curl -X POST -F "file=@lesion.jpg" http://localhost:5000/predict
```

Response:
```json
{
  "top_prediction": {"code": "nv", "name": "Melanocytic nevus", "risk": "low", "probability": 87.42},
  "predictions": [
    {"code": "nv", "name": "Melanocytic nevus", "risk": "low", "probability": 87.42},
    {"code": "bkl", "name": "Benign keratosis-like lesion", "risk": "low", "probability": 6.11},
    ...
  ]
}
```

### `GET /health`
Returns model load status — useful for checking the weights loaded
correctly.

## Classes

| Code  | Full name |
|-------|-----------|
| mel   | Melanoma |
| nv    | Melanocytic nevus |
| bkl   | Benign keratosis-like lesion |
| bcc   | Basal cell carcinoma |
| akiec | Actinic keratosis / intraepithelial carcinoma |
| vasc  | Vascular lesion |
| df    | Dermatofibroma |

## Notes

- Preprocessing (resize to 224×224, ImageNet mean/std normalization)
  matches the notebook's `ts_tfs` test-time transform exactly.
- This is a demo tool, not a medical device — the UI includes a
  disclaimer accordingly.
