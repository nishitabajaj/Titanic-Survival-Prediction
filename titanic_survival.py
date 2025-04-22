import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv(r"C:\Users\Nishita Bajaj\OneDrive\Desktop\tested.csv")

# Handle missing values and preprocessing
data["Age"] = data["Age"].fillna(int(data["Age"].mean()))
data.drop(columns=["Cabin"], inplace=True)
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

# Convert categorical columns to dummy variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Drop unnecessary columns
data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

# Standardize the Age and Fare columns
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Define feature matrix and target variable
X = data.drop(columns='Survived')
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Save the model using pickle
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)