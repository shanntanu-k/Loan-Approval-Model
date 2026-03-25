# Loan Approval Predictor

A Flask web application that predicts whether a loan will be approved using a machine learning pipeline trained on applicant data.

## What This Project Demonstrates
- End-to-end ML workflow (data prep, training, evaluation, serving)
- Production-style preprocessing with `Pipeline` and `ColumnTransformer`
- Input validation and API-based inference in Flask
- Frontend form integration with backend prediction API
- Free cloud deployment with Render

## Project Structure
- `main.py`: Flask app and prediction endpoint
- `train.py`: Training pipeline, metrics, and model artifact generation
- `templates/index.html`: UI form and prediction popup
- `static/style.css`: Styling
- `tests/test_app.py`: Basic API tests
- `render.yaml`: Free deployment configuration for Render

## Dataset Schema
The model uses these input features:
- `Income` (numeric)
- `Credit_Score` (numeric)
- `Loan_Amount` (numeric)
- `DTI_Ratio` (numeric)
- `Employment_Status` (`employed` or `unemployed`)

Target column:
- `Approval` (`Approved` or `Rejected`)

## Local Setup
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/Loan-Prediction.git
cd Loan-Prediction
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python train.py
```

5. Run the app:
```bash
python main.py
```

6. Open:
```text
http://127.0.0.1:8080
```

## Run Tests
```bash
pytest -q
```

## API Usage
Endpoint:
```text
POST /predict
Content-Type: application/json
```

Example request body:
```json
{
  "Income": 85000,
  "Credit_Score": 730,
  "Loan_Amount": 25000,
  "DTI_Ratio": 22.5,
  "Employment_Status": "employed"
}
```

## Free Deployment on Render
This repository already includes `render.yaml`, so deployment is simple.

1. Push your code to GitHub.
2. Go to [Render](https://render.com/) and sign in.
3. Click `New +` -> `Blueprint`.
4. Connect your GitHub repo.
5. Render will detect `render.yaml` and create the web service.
6. Wait for build and deploy to complete.
7. Open the generated live URL and add it to your CV.

### Build and Start Commands (already configured)
- Build: `pip install -r requirements.txt && python train.py`
- Start: `gunicorn main:app`

## CV Bullet Ideas
- Built and deployed an end-to-end loan approval prediction web app using Flask and scikit-learn.
- Designed a robust ML pipeline with one-hot encoding, scaling, and model validation metrics.
- Implemented REST-style inference API with input validation and frontend integration.
- Deployed on Render (free tier) with reproducible cloud build and startup workflow.
