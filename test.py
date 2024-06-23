import os
import pandas as pd
from src.utilis import load_object

# preprocessor = load_object(os.path.join('artifacts', 'preprocessor.pkl'))

# # Example new data (it should have the same structure as the training data)
# new_data = pd.DataFrame({
#     'Total_Stops': [2],
#     'Journey_day': [15],
#     'Journey_month': [6],
#     'Dep_hour': [14],
#     'Dep_min': [35],
#     'Arrival_hour': [16],
#     'Arrival_min': [20],
#     'Duration_hours': [2],
#     'Duration_mins': [45],
#     'Airline_Air India': [0],
#     'Airline_GoAir': [1],
#     'Airline_IndiGo': [0],
#     'Airline_Jet Airways': [0],
#     'Airline_Jet Airways Business': [0],
#     'Airline_Multiple carriers': [0],
#     'Airline_Multiple carriers Premium economy': [0],
#     'Airline_SpiceJet': [0],
#     'Airline_Vistara': [0],
#     'Airline_Vistara Premium economy': [0],
#     'Source_Chennai': [0],
#     'Source_Delhi': [1],
#     'Source_Kolkata': [0],
#     'Source_Mumbai': [0],
#     'Destination_Cochin': [0],
#     'Destination_Delhi': [0],
#     'Destination_Hyderabad': [0],
#     'Destination_Kolkata': [0],
#     'Destination_New Delhi': [1]
# })

# # Transform the new data
# transformed_data = preprocessor.transform(new_data)
# print(transformed_data)

# # transformed_data is now ready to be fed into the model for prediction



import os
import numpy as np
import pandas as pd
from src.utilis import load_object

# Load the pre-trained model
model_path = os.path.join("artifacts", "model.pkl")
model = load_object(model_path)

# Load the preprocessor
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
preprocessor = load_object(preprocessor_path)

# Define the column names as per your preprocessing requirements
numerical_columns = [
    'Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
    'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
    'Duration_mins'
]
categorical_columns = [
    'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
    'Airline_Jet Airways Business', 'Airline_Multiple carriers', 'Airline_Multiple carriers Premium economy',
    'Airline_SpiceJet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
    'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai', 'Destination_Cochin',
    'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata', 'Destination_New Delhi'
]
all_columns = numerical_columns + categorical_columns

# Input data from user
input_data = list(int(num) for num in input("Enter the elements for input data: ").strip().split(","))

# Convert input data to numpy array
np_input_data = np.asarray(input_data)

# Reshape the numpy array
np_input_data_reshape = np_input_data.reshape(1, -1)

# Convert the numpy array to a pandas DataFrame
df_input_data = pd.DataFrame(np_input_data_reshape, columns=all_columns)

# Apply the preprocessor to the input data
preprocessed_input_data = preprocessor.transform(df_input_data)

# Make prediction using the model
prediction = model.predict(preprocessed_input_data)

# Print the prediction
print(prediction)

# Check and print the prediction
if prediction[0]:
    print(f"The price will be {prediction[0]}")
else:
    print("none")
