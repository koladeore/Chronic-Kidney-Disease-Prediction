import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, f1_score, accuracy_score, confusion_matrix, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('heart.csv')

# Features and target
X = df.drop(columns=['Class'])  # Multiple features used for prediction
y = df['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert predictions to binary (0 or 1) based on a threshold (0.5)
y_pred_binary = np.where(y_pred >= 0.5, 1, 0)

# Calculate metrics
precision = precision_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
accuracy = accuracy_score(y_test, y_pred_binary)
mse = mean_squared_error(y_test, y_pred)

# Save the metrics to a dictionary
metrics = {
    'precision': precision,
    'f1_score': f1,
    'accuracy': accuracy,
    'mse': mse
}

# Save the metrics dictionary to a file
joblib.dump(metrics, 'metrics.pkl')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')

# Loss vs. Accuracy Plot (For regression, we're using loss vs. predicted values)
plt.figure(figsize=(10, 6))
plt.plot(y_pred, label='Predicted Values')
plt.plot(y_test.values, label='Actual Values')
plt.title('Loss vs. Predicted Values')
plt.xlabel('Test Data Points')
plt.ylabel('Values')
plt.legend()
plt.savefig('loss_vs_accuracy.png')

# Save the trained model
joblib.dump(model, 'ckd_model.pkl')
