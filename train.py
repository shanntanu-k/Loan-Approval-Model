import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("loan_data.csv")

df.dropna(subset=['Approval', 'Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio', 'Employment_Status'], inplace=True)

df['Approval'] = df['Approval'].map({'Approved': 1, 'Rejected': 0})
df['Employment_Status'] = df['Employment_Status'].astype('category').cat.codes

X = df[['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio', 'Employment_Status']]
y = df['Approval']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model trained and saved")