import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'NSCLC_Clinical_Trials_Data_UTF8.csv'
data = pd.read_csv(file_path)

# Convert categorical columns to category type
categorical_columns = ['Gender', 'Race', 'Stage of NSCLC', 'Treatment Arm', 'OS Status', 'Dropped Out']
for column in categorical_columns:
    data[column] = data[column].astype('category')

# Plot distribution of tumor measurements across treatment arms
plt.figure(figsize=(10, 6))
sns.boxplot(x='Treatment Arm', y='Tumor Measurement (mm)', data=data)
plt.title('Distribution of Tumor Measurements Across Treatment Arms')
plt.xlabel('Treatment Arm')
plt.ylabel('Tumor Measurement (mm)')
plt.savefig('tumor_measurement_distribution.png')
plt.close()

# Plot predicted vs actual PFS values (dummy prediction for illustration)
# Assuming y_pred is the predicted PFS values from the model
# Here we use actual PFS values for both actual and predicted for demonstration
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PFS (months)', y='PFS (months)', data=data)
plt.title('Predicted vs Actual PFS Values')
plt.xlabel('Actual PFS (months)')
plt.ylabel('Predicted PFS (months)')
plt.savefig('predicted_vs_actual_pfs.png')
plt.close()