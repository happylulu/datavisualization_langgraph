import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('NSCLC_Clinical_Trials_Data_UTF8.csv')

# Data Cleaning
# Fill missing values with the median for numerical features
numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
for feature in numerical_features:
    data[feature].fillna(data[feature].median(), inplace=True)

# Encode categorical variables
categorical_features = data.select_dtypes(include=['object']).columns
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature].astype(str))
    label_encoders[feature] = le

# Feature Scaling
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Define features and target variable
features = data.drop(columns=['Response'])
target = data['Response']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the models
rf_model = RandomForestClassifier(random_state=42)
logistic_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the Random Forest model
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Train the Logistic Regression model
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions, average='weighted')
rf_recall = recall_score(y_test, rf_predictions, average='weighted')
rf_f1 = f1_score(y_test, rf_predictions, average='weighted')

# Evaluate the Logistic Regression model
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_precision = precision_score(y_test, logistic_predictions, average='weighted')
logistic_recall = recall_score(y_test, logistic_predictions, average='weighted')
logistic_f1 = f1_score(y_test, logistic_predictions, average='weighted')

# Output the evaluation metrics
rf_results = {
    'Accuracy': rf_accuracy,
    'Precision': rf_precision,
    'Recall': rf_recall,
    'F1 Score': rf_f1
}

logistic_results = {
    'Accuracy': logistic_accuracy,
    'Precision': logistic_precision,
    'Recall': logistic_recall,
    'F1 Score': logistic_f1
}

rf_results, logistic_results