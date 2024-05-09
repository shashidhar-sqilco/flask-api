from flask import Flask, request, jsonify
from pymongo import MongoClient
from flask_cors import CORS
import pickle
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)
mongo_uri = os.getenv("MONGO_URI")

# Create a new client and connect to the server
try:
    client = MongoClient(mongo_uri)
    db = client.myDatabase
    collection = db["predictions"]
except Exception as e:
    print("Error connecting to MongoDB:", e)
    exit(1)

# Load the pickled model
try:
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print("Error loading the model:", e)
    exit(1)


@app.route('/')
def index():
    return 'Flask server is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.json
        
        # Extract numerical features from the input dictionary
        age = data.get('age')
        bmi = data.get('bmi')
        children = data.get('children')
        
        # Convert categorical features to numerical values
        # Assuming 'sex', 'smoker', and 'region' are categorical features
        sex = 1 if data.get('sex') == 'male' else 0
        smoker = 1 if data.get('smoker') == 'yes' else 0
        region = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}.get(data.get('region'))
        
        if None in [age, bmi, children, sex, smoker, region]:
            raise ValueError("One or more required fields are missing.")
        
        # Create an array containing all features
        features = [age, bmi, children, sex, smoker, region]
        
        # Add default values for missing features if necessary
        # For example, if your model expects 11 features, you can add default values for the remaining 5 features
        features.extend([0] * 5)  # Assuming remaining features are numerical and set to 0
        
        # Make prediction using the loaded model
        result = model.predict([features])
        
        # Convert NumPy array to Python list
        result_list = result.tolist()
        
        # Store request and prediction data in MongoDB
        collection.insert_one({'request': data, 'prediction': result_list})
        
        return jsonify({'request': data, 'prediction': result_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/all_data', methods=['GET'])
def get_all_data():
    try:
        # Fetch all data from MongoDB collection
        all_data = list(collection.find({}, {'_id': 0}))
        return jsonify(all_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
