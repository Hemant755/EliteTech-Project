# Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, jsonify

# Step 1: Generate Random Data
def generate_random_data():
    np.random.seed(42)
    n_samples = 1000
    data = {
        "Age": np.random.randint(18, 65, size=n_samples),
        "Annual_Income": np.random.randint(20000, 120000, size=n_samples),
        "Browsing_Time_Mins": np.random.randint(5, 120, size=n_samples),
        "Purchased": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 70% did not purchase, 30% did
    }
    df = pd.DataFrame(data)
    df.to_csv("random_data.csv", index=False)
    print("Random data generated and saved as 'random_data.csv'.")

# Step 2: Preprocess and Train the Model
def train_and_save_model():
    # Load Data
    df = pd.read_csv("random_data.csv")

    # Data Preprocessing
    X = df.drop("Purchased", axis=1)  # Features
    y = df["Purchased"]  # Target

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

    # Save Model
    joblib.dump(model, "model.pkl")
    print("Model saved as 'model.pkl'.")

# Step 3: Create Flask App for Deployment
def create_flask_app():
    app = Flask(__name__)

    # Load the trained model
    model = joblib.load("model.pkl")

    @app.route("/")
    def home():
        return "Welcome to the Random Data Prediction API!"

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json()  # Expecting JSON input
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})

    return app

# Step 4: Main Execution Flow
if __name__ == "__main__":
    # Generate Data
    generate_random_data()

    # Train and Save Model
    train_and_save_model()

    # Deploy the Flask App
    app = create_flask_app()
    app.run(debug=True)
