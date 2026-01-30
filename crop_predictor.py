import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings

# Ignore future warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset
# Ensure 'crop_data.csv' is in the same directory as this script
data = pd.read_csv("crop_data.csv")
print(data.head())

# Visualizations
plt.ion() # Allows code to continue after plotting
plt.figure(figsize=(8, 5))
sns.countplot(x="crop", data=data, hue="crop", palette="Set2", legend=False)
plt.title("Crop Frequency")
plt.show()

# Split features and target
X = data.drop("crop", axis=1)
y = data["crop"]

# Train-Test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model for future use
joblib.dump(model, "crop_model.pkl")

# Prediction function
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Load the saved model [cite: 1146]
    model = joblib.load("crop_model.pkl")
    
    # Create a DataFrame for the sample input [cite: 1147-1148]
    sample = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                          columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    
    # Make prediction [cite: 1149]
    prediction = model.predict(sample)
    return prediction[0]

# Example prediction
if __name__ == "__main__":
    # Example values for Nitrogen, Phosphorus, Potassium, etc. [cite: 1152]
    result = recommend_crop(80, 50, 45, 25, 65, 6.8, 150)
    print("\n🌱 Recommended Crop:", result)
