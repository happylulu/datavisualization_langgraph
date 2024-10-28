import pandas as pd

# Load the dataset
file_path = 'NSCLC_Clinical_Trials_Data_UTF8.csv'
data = pd.read_csv(file_path)

# Data Cleaning and Preprocessing
# 1. Drop duplicate rows
cleaned_data = data.drop_duplicates()

# 2. Handle missing values
# Fill numeric columns with the mean and categorical columns with the mode
for column in cleaned_data.columns:
    if cleaned_data[column].dtype in ['int64', 'float64']:
        cleaned_data[column].fillna(cleaned_data[column].mean(), inplace=True)
    else:
        cleaned_data[column].fillna(cleaned_data[column].mode()[0], inplace=True)

# 3. Normalize numeric features
numeric_columns = cleaned_data.select_dtypes(include=['int64', 'float64']).columns
cleaned_data[numeric_columns] = (cleaned_data[numeric_columns] - cleaned_data[numeric_columns].mean()) / cleaned_data[numeric_columns].std()

# Save the cleaned and preprocessed data to a new CSV file
cleaned_data.to_csv('NSCLC_Clinical_Trials_Data_Cleaned.csv', index=False)

# Display the first few rows of the cleaned data
cleaned_data.head()