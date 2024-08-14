import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib  # Ensure this is imported

# Load the dataset
df = pd.read_csv('heart.csv')

# Features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'ckd_model.pkl')  # This line saves the model to 'ckd_model.pkl'

# (Optional) Output some metrics or results
