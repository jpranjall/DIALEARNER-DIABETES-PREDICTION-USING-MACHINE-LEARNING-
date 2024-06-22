from flask import Flask, render_template, request
import numpy as np
# import pickle
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# model = pickle.load(open("diabetes_pred.sav", 'rb'))
# scaler = pickle.load(open("scaler.sav",'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            print("Hell1")
            # Get form data
            glucose = float(request.form['glucose'])
            blood_pressure = float(request.form['blood_pressure'])
            skin_thickness = float(request.form['skin_thickness'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])
            print(age)
            
            # Create input array
            user_data = np.array([[glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            user_data_scaled = scaler.transform(user_data)
            
            # Make prediction
            prediction = model.predict(user_data_scaled)
            print(prediction)

            print("Hell2")
            
            # Interpret the result
            if prediction[0] == 1:
                result = "Based on the provided data, you have a high chance of having diabetes."
            else:
                result = "Based on the provided data, you have a low chance of having diabetes."
        except Exception as e:
            result = f"Error in processing input: {str(e)}"
        
        return render_template('index.html', result=result)
    
    return render_template('index.html', result='')

if __name__ == '__main__':
    app.run(debug=True)
