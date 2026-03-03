# Predicting Rent Price Project

A web application that estimates monthly apartment rent prices in Vienna based on property features.

The project includes:
- **Backend** built with **FastAPI** (PyTorch model inference).
- **Frontend** built with plain **HTML/CSS/JavaScript** (input form + prediction output).
- Utility modules for data preprocessing and model definition.

---

## Features

- Apartment feature input:
  - area in square meters;
  - number of rooms;
  - district (1–23);
  - balcony/terrace flags;
  - furnished flag;
  - social housing flag.
- Rent prediction through `POST /predict`.
- Frontend served directly by FastAPI via `StaticFiles`.

---

## Tech Stack

- **Python 3.11**
- **FastAPI + Uvicorn**
- **PyTorch**
- **Pandas / NumPy / scikit-learn / joblib**
- **HTML, CSS, JavaScript**
- **Docker** (`Dockerfile.dev`)

---

## Project Structure

```text
.
├── backend
│   ├── app
│   │   └── main.py                # API, model loading, frontend static serving
│   └── model
│       ├── model.py               # PricePredictor neural network architecture
│       ├── data_processing.py     # Data cleaning and preprocessing
│       ├── prediction.py          # Local inference example
│       ├── data_visualisation.py  # Data visualization example
│       ├── custom_dataset.py      # PyTorch Dataset implementation
│       └── data
│           ├── vienna_apartments.csv
│           └── losses.txt
├── frontend
│   ├── index.html
│   ├── style.css
│   └── script.js
├── Dockerfile.dev
└── requirements.txt
```

---

## Quick Start (Local)

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API server:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

4. Open in your browser:

```text
http://127.0.0.1:8000/
```

---

## Run with Docker

Build image:

```bash
docker build -f Dockerfile.dev -t rent-predictor:dev .
```

Run container:

```bash
docker run --rm -p 8000:8000 rent-predictor:dev
```

Then open:

```text
http://127.0.0.1:8000/
```

---

## API

### `POST /predict`

Example request body:

```json
{
  "area_sqm": 50.0,
  "rooms": 2,
  "has_balcony": true,
  "has_terrace": false,
  "is_furnished": false,
  "is_social_housing": false,
  "district": 10
}
```

Example response:

```json
{
  "predicted_rent_price": 1234
}
```

---

## Important Model Artifact Note

The app expects trained artifacts in `backend/model/data/`:
- `model.pth`
- `scaler_X.pkl`
- `scaler_Y.pkl`

If these files are missing, the backend will fail at startup while loading the model/scalers.

---

## Data and Preprocessing

Main dataset: `backend/model/data/vienna_apartments.csv`.

In `data_processing.py`, the pipeline performs:
- duplicate removal and dropping critical `NaN` values;
- rent filtering (`400 <= price <= 5000`);
- district extraction from address;
- log transform of target variable `price`;
- one-hot encoding for district;
- train/test split.

---
