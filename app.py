from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
model_path = "artifacts/model.pkl"
model = pickle.load(open(model_path, "rb"))

# Load the preprocessor
preprocessor_path = "artifacts/preprocessor.pkl"
preprocessor = pickle.load(open(preprocessor_path, "rb"))


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        # Extract features from form data
        date_dep = request.form["Dep_Time"]
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").month)
        Dep_hour = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").minute)
        
        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").minute)
        
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)
        
        Total_stops = int(request.form["stops"])
        
        airline = request.form['airline']
        Air_India = 1 if airline == 'Air India' else 0
        GoAir = 1 if airline == 'GoAir' else 0
        IndiGo = 1 if airline == 'IndiGo' else 0
        Jet_Airways = 1 if airline == 'Jet Airways' else 0
        Jet_Airways_Business = 1 if airline == 'Jet Airways Business' else 0
        Multiple_carriers = 1 if airline == 'Multiple carriers' else 0
        Multiple_carriers_Premium_economy = 1 if airline == 'Multiple carriers Premium economy' else 0
        SpiceJet = 1 if airline == 'SpiceJet' else 0
        Vistara = 1 if airline == 'Vistara' else 0
        Vistara_Premium_economy = 1 if airline == 'Vistara Premium economy' else 0
        
        Source = request.form["Source"]
        s_Chennai = 1 if Source == 'Chennai' else 0
        s_Delhi = 1 if Source == 'Delhi' else 0
        s_Kolkata = 1 if Source == 'Kolkata' else 0
        s_Mumbai = 1 if Source == 'Mumbai' else 0
        
        Destination = request.form["Destination"]
        d_Cochin = 1 if Destination == 'Cochin' else 0
        d_Delhi = 1 if Destination == 'Delhi' else 0
        d_Hyderabad = 1 if Destination == 'Hyderabad' else 0
        d_Kolkata = 1 if Destination == 'Kolkata' else 0
        d_New_Delhi = 1 if Destination == 'New_Delhi' else 0

        # Create a DataFrame with the input features
        input_data = {
            'Total_Stops': Total_stops,
            'Journey_day': Journey_day,
            'Journey_month': Journey_month,
            'Dep_hour': Dep_hour,
            'Dep_min': Dep_min,
            'Arrival_hour': Arrival_hour,
            'Arrival_min': Arrival_min,
            'Duration_hours': dur_hour,
            'Duration_mins': dur_min,
            'Airline_Air India': Air_India,
            'Airline_GoAir': GoAir,
            'Airline_IndiGo': IndiGo,
            'Airline_Jet Airways': Jet_Airways,
            'Airline_Jet Airways Business': Jet_Airways_Business,
            'Airline_Multiple carriers': Multiple_carriers,
            'Airline_Multiple carriers Premium economy': Multiple_carriers_Premium_economy,
            'Airline_SpiceJet': SpiceJet,
            'Airline_Vistara': Vistara,
            'Airline_Vistara Premium economy': Vistara_Premium_economy,
            'Source_Chennai': s_Chennai,
            'Source_Delhi': s_Delhi,
            'Source_Kolkata': s_Kolkata,
            'Source_Mumbai': s_Mumbai,
            'Destination_Cochin': d_Cochin,
            'Destination_Delhi': d_Delhi,
            'Destination_Hyderabad': d_Hyderabad,
            'Destination_Kolkata': d_Kolkata,
            'Destination_New Delhi': d_New_Delhi
        }
        
        # Convert input_data to DataFrame
        df_input = pd.DataFrame([input_data])

        # Preprocess the input data
        preprocessed_input = preprocessor.transform(df_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        output = round(prediction[0], 2)

        return render_template('home.html', prediction_text=f'Your Flight price is Rs. {output}')

    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=7000)
