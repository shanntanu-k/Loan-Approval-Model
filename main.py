from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# ✅ Load model (NO TRAINING HERE)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [
            float(request.form['Income']),
            float(request.form['Credit_Score']),
            float(request.form['Loan_Amount']),
            float(request.form['DTI_Ratio']),
            int(request.form['Employment_Status'])
        ]

        inputs_scaled = scaler.transform([inputs])

        prediction = model.predict(inputs_scaled)[0]
        probability = model.predict_proba(inputs_scaled)[0][prediction]

        confidence = round(probability * 100, 2)

        result = "✅ Loan Approved!" if prediction == 1 else "❌ Loan Rejected"
        result += f" (Confidence: {confidence}%)"

    except Exception as e:
        result = f"Error: {e}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)