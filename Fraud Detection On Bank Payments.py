import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv("bank_transaction_data.csv")

# Data preprocessing
# Handle missing values
data.fillna(data.mean(), inplace=True)

# Encode categorical variables
data = pd.get_dummies(data)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('is_fraud', axis=1), data['is_fraud'], test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = IsolationForest(contamination=0.01)  # 1% of data assumed to be fraudulent
model.fit(X_train_scaled)

# Model evaluation
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
