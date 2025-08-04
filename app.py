from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None

    if request.method == 'POST':
        # Get form data
        Pclass = int(request.form['Pclass'])
        Sex = int(request.form['Sex'])
        Age = float(request.form['Age'])
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])
        Embarked = int(request.form['Embarked'])

        features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
        scaled_features = scaler.transform(features)

        result = model.predict(scaled_features)[0]
        confidence = model.predict_proba(scaled_features)[0][result]

        prediction = "Yes ✅ (Survived)" if result == 1 else "No ❌ (Did not survive)"
        probability = f"{confidence * 100:.2f}%"

    return render_template("form.html", prediction=prediction, probability=probability)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()

        features = np.array([[
            int(data['Pclass']),
            int(data['Sex']),
            float(data['Age']),
            int(data['SibSp']),
            int(data['Parch']),
            float(data['Fare']),
            int(data['Embarked'])
        ]])
        scaled_features = scaler.transform(features)
        result = model.predict(scaled_features)[0]
        confidence = model.predict_proba(scaled_features)[0][result]

        return jsonify({
            "prediction": "Yes" if result == 1 else "No",
            "confidence": f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

