import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
file_path = 'NSCLC_Clinical_Trials_Data_UTF8.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head()