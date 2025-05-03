import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("dataset.csv")

# Binary target: accident occurred (1 for real entries, 0 for dummy/no accident rows)
# For demo, we simulate using 'Severity' > 0 to indicate accident occurrence
df = df.dropna(subset=['Severity', 'Start_Time'])
df['Accident_Occurred'] = df['Severity'].apply(lambda x: 1 if x > 0 else 0)

# Extract hour and rush hour flag
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Hour'] = df['Start_Time'].dt.hour
df['Is_Rush_Hour'] = df['Hour'].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)

# Features and Labels
features = ['Hour', 'Is_Rush_Hour']
X = df[features]
y = df['Accident_Occurred']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Evaluation
print("=== Random Forest Classification Report ===")
print(classification_report(y_test, rf_preds))
print("
=== XGBoost Classification Report ===")
print(classification_report(y_test, xgb_preds))
