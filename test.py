import joblib

vectorizer =  joblib.load("models/vectorizer.pkl")

sleeve_model = joblib.load("models/sleeve_model.pkl")

print("Loaded successfully")