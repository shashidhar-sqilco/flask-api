from flask import Flask, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)

# Load the pickled model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return 'Flask server is running!'

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.json
    
    # Extract numerical features from the input dictionary
    age = data['age']
    bmi = data['bmi']
    children = data['children']
    
    # Convert categorical features to numerical values
    # Assuming 'sex', 'smoker', and 'region' are categorical features
    sex = 1 if data['sex'] == 'male' else 0
    smoker = 1 if data['smoker'] == 'yes' else 0
    region = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}[data['region']]
    
    # Create an array containing all features
    features = [age, bmi, children, sex, smoker, region]
    
    # Add default values for missing features if necessary
    # For example, if your model expects 11 features, you can add default values for the remaining 5 features
    features.extend([0] * 5)  # Assuming remaining features are numerical and set to 0
    
    # Make prediction using the loaded model
    result = model.predict([features])
    
    # Convert NumPy array to Python list
    result_list = result.tolist()
    return jsonify({'prediction': result_list})


    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)