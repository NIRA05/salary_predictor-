# Random Forest Regressor Web App

This project wraps your uploaded `Random_Forest_Regressor_Model.pkl` inside a simple Flask application with a polished frontend.

## Features

- Responsive frontend with a clean prediction form
- Flask API for model inference
- Automatic one-hot reconstruction for the model's encoded feature vector
- Ready to push to GitHub as a starter project

## Project Structure

```text
random-forest-regressor-webapp/
├── app.py
├── model/
│   └── Random_Forest_Regressor_Model.pkl
├── model_metadata.json
├── requirements.txt
├── static/
│   ├── index.html
│   ├── script.js
│   └── styles.css
└── README.md
```

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Notes

- The form fields were inferred from the model's `feature_names_in_` metadata.
- The target label is shown as **Predicted Value** because the `.pkl` file does not include a human-readable target name.
- If you know the output is salary, house price, etc., rename the heading in `static/index.html` and the note in `app.py`.

## Deployment Ideas

- Render
- Railway
- Fly.io
- Any VPS with Python 3.10+
