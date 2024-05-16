from flask import Flask, request, jsonify
from pymongo import MongoClient
from flask_cors import CORS
import pickle
from dotenv import load_dotenv
import os
import snowflake.connector

app = Flask(__name__)
CORS(app)

mongo_uri = os.getenv("MONGO_URI")

# Connect to Snowflake account
conn = snowflake.connector.connect(
    account='ixzsmyp-jr08929',
    user='anujsqilco',
    password='Sqilco@1',
    warehouse='AMALWEBFORMULER',
    database='WEBFORMULER',
    schema='WEBFORMULER'
)
cur = conn.cursor()

# Create the FACT_PREDICTIONS table in Snowflake
cur.execute('''
    CREATE TABLE IF NOT EXISTS FACT_PREDICTIONS (
        PREDICTION_ID NUMBER(38,0) AUTOINCREMENT,
        AGE NUMBER(3,0),
        SEX VARCHAR(5),
        BMI NUMBER(5,2),
        CHILDREN NUMBER(2,0),
        SMOKER BOOLEAN,
        REGION VARCHAR(10),
        PREDICTION NUMBER(10,2),
        CREATED_AT TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
        PRIMARY KEY (PREDICTION_ID)
    )
''')

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
        data = request.json
        # Extract features from user input
        age = data.get('age')
        bmi = data.get('bmi')
        children = data.get('children')
        sex = 1 if data.get('sex') == 'male' else 0
        smoker = 1 if data.get('smoker') == 'yes' else 0
        region = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}.get(data.get('region'))

        if None in [age, bmi, children, sex, smoker, region]:
            raise ValueError("One or more required fields are missing.")

        # Create an array containing all features
        features = [age, bmi, children, sex, smoker, region]

        # Add default values for missing features if necessary
        features.extend([0] * 5)  # Assuming remaining features are numerical and set to 0

        # Make prediction using the loaded model
        result = model.predict([features])

        # Store user input and prediction in MongoDB
        prediction_data = {
            'age': age,
            'bmi': bmi,
            'children': children,
            'sex': 'male' if sex == 1 else 'female',
            'smoker': 'yes' if smoker == 1 else 'no',
            'region': data.get('region'),
            'prediction': result[0]
        }
        collection.insert_one(prediction_data)

        # Store user input and prediction in Snowflake
        cur.execute('''
            INSERT INTO FACT_PREDICTIONS (AGE, SEX, BMI, CHILDREN, SMOKER, REGION, PREDICTION)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (age, 'male' if sex == 1 else 'female', bmi, children, bool(smoker), data.get('region'), result[0]))
        conn.commit()

        return jsonify({'prediction': result[0]})

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