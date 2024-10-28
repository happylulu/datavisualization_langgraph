import matplotlib.pyplot as plt
import numpy as np

# Model evaluation results
rf_results = {'Accuracy': 0.85, 'Precision': 0.80, 'Recall': 0.75, 'F1 Score': 0.77}
logistic_results = {'Accuracy': 0.82, 'Precision': 0.78, 'Recall': 0.74, 'F1 Score': 0.76}

# Bar chart data
labels = list(rf_results.keys())
rf_scores = list(rf_results.values())
logistic_scores = list(logistic_results.values())

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

# Create bar chart
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest')
rects2 = ax.bar(x + width/2, logistic_scores, width, label='Logistic Regression')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value labels on top of the bars
for rect in rects1:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

for rect in rects2:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Save the figure
plt.tight_layout()
plt.savefig('model_performance_comparison.png')
plt.show()