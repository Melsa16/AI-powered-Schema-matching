import joblib
from flask import Flask, request, app, jsonify, url_for, render_template
import os

app=Flask(__name__)

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved artifacts using absolute paths
model = joblib.load(os.path.join(BASE_DIR, 'xgboost_schema_mapper.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(f"Received data: {data}")
    
    # Convert data to a list of column names
    column_names = data
    
    # Preprocess and predict
    X_new = vectorizer.transform(column_names)
    predictions = model.predict(X_new)
    standardized = label_encoder.inverse_transform(predictions)
    
    # Create a response dictionary
    response = dict(zip(column_names, standardized))
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)