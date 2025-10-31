    
from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#placeholder taki apan reuse kar pae 
model = None
scaler = None

# astart training
def train_model():
    global scaler

    df = pd.read_csv("loan_data.csv")

    # Drop rows with khali values 
    df.dropna(subset=['Approval', 'Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio', 'Employment_Status'], inplace=True)

    # Map Approval column 
    df['Approval'] = df['Approval'].map({'Approved': 1, 'Rejected': 0})

    # Encode Employment_Status to numbers
    df['Employment_Status'] = df['Employment_Status'].astype('category').cat.codes

    # Define features and label
    X = df[['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio', 'Employment_Status']]
    y = df['Approval']

    # Fit scaler and scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the model with class weight balancing 
    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model

# Train once
model = train_model()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read input values
        inputs = [
            float(request.form['Income']),
            float(request.form['Credit_Score']),
            float(request.form['Loan_Amount']),
            float(request.form['DTI_Ratio']),
            int(request.form['Employment_Status'])  # Encoded value
        ]

        # Scale the input using the same scaler as training
        inputs_scaled = scaler.transform([inputs])

        # Predict
        prediction = model.predict(inputs_scaled)[0]
        result = "✅ Loan Approved!" if prediction == 1 else "❌ Loan Rejected"

    except Exception as e:
        result = f"Error: {e}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
