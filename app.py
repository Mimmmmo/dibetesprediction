from flask import Flask, render_template, request
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model/diabetes_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        inputs = [float(request.form.get(key)) for key in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]]
        scaled_inputs = scaler.transform([inputs])
        prediction = model.predict(scaled_inputs)[0]
        result = "Diabetic" if prediction >= 0.5 else "Non-Diabetic"
        return render_template("result.html", prediction=result)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
